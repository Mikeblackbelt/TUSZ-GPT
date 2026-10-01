# EEG-GPT Model Architecture

Source: `eeg_gpt/models/gpt.py`, `eeg_gpt/data/tokenizer.py`. Based on Mahdi & Baghdadi 2026 (arXiv:2603.26821), with electrode-wise modeling as in BrainGPT/EEGPT.

## Overview

```
raw EEG (C ch × T samples)
  └─ AmplitudeTokenizer ──► tokens (C × T), 512 levels
       └─ EEGGPT backbone (per channel, causal)      ← pretrained with next-token LM
            └─ WindowEncoder (pool over time, transformer over channels + CLS) ─► 1 vector / block
                 └─ TemporalContext (optional, causal over N blocks) ─► newest-block vector
                      ├─ binary_head  (preictal / horizon)
                      └─ type_head    (seizure type)
```

## 1. Tokenizer: `AmplitudeTokenizer`

| Step | Formula |
|---|---|
| z-score | `z = (x − μ) / σ` (per channel per block by default; `stats="session"` uses per-recording stats) |
| clip | `z_clip = clip(z, −k, k)`, k = 5 |
| scale | `u = (z_clip + k) / 2k ∈ [0,1]` |
| quantize | `token = floor(u · (L−1))`, L = 512 |
| dequantize | `ẑ = level/(L−1) · 2k − k` |

Unlike the paper, each channel gets **its own token stream**. The paper interleaves channels into one stream. Channel identity goes into the model as an embedding instead.

## 2. Backbone: `EEGGPT`

A pre-LN, decoder-only causal transformer that runs over the amplitude tokens of **one channel**.

- **Input embedding** = `tok_emb(token) + pos_emb(t) + ch_emb(channel)`, followed by dropout
- **Blocks** (×`layers`): `x + Attn(LN(x))`, then `x + FFN(LN(x))`
  - Attention uses `F.scaled_dot_product_attention(is_causal=True)`, which enables flash/memory-efficient kernels and needs no T×T mask
  - FFN: Linear → GELU → Dropout → Linear → Dropout
- **Final LN** leads into one or more **LM heads**. There is one `Linear(d, 512, bias=False)` per prediction offset (`pred_offsets`, e.g. `1,8,32` for multi-token prediction)
- Init: normal(0, 0.02); biases are set to zero

**Default hyperparameters (pretrain.py):**

| param | default |
|---|---|
| n_levels | 512 |
| d | 128 |
| layers | 4 |
| heads | 4 |
| ffn | 512 |
| block_size | 1024 tokens (4 s @ 256 Hz) |
| dropout | 0.1 |
| n_channels | 20 |

### Pretraining loss: `gpt_loss`

- **Cross-entropy** for every offset `o`: the logits at position t predict `tokens[t+o]`
- **Dual MSE** (Mahdi §III-E), offset-1 head only: `ẑ = Σ softmax(logits) · level_values`, then `MSE(ẑ, z_true)`, weighted by `lambda_mse` (default 0.1)
- `loss = Σ_o CE@o + λ·MSE`

Optimizer: AdamW (β = 0.9/0.95, wd 0.05; embeddings and 1-D params are not decayed), lr 3e-4 with warmup plus decay, and AMP.

## 3. Fine-tuning: `SeizureModel`

### `WindowEncoder`: (B, C, T) → (B, d)
1. It runs the backbone's `hidden()` on every channel sequence. Channels are processed in chunks of `seq_chunk` = 256 to bound activation memory.
2. It pools over time. `mean` is the default (Mahdi eq. 17); `last` and `meanmax` are the alternatives.
3. It adds learned channel-position embeddings and prepends a learned **CLS** token.
4. It runs `agg_layers` (default 1) **non-causal** transformer blocks over the 1+C tokens (4 heads, ffn = 4d).
5. The output is LN(CLS).

### `TemporalContext`: (B, N, d) → (B, d), optional
This module is used only when `context_blocks > 1`. It adds learned block-position embeddings, then applies 2 **causal** transformer blocks over the N windows (oldest → newest) and LN. It returns the newest position.

### Heads: `Head`
`LayerNorm → GELU → Dropout → Linear(d,d) → GELU → Dropout → Linear(d, n_out)` (Mahdi §III-F 2), with dropout 0.2.
- `binary_head`: 1 logit (preictal vs background, or the seizure-within-horizon variant; default horizon is 30 s)
- `type_head`: `n_types` logits (multi-type seizure classification)

### Forward
`tokens (B, C, N, T)` goes through the encoder once per block to give `(B, N, d)`. If `context_blocks > 1`, TemporalContext runs next; otherwise the model takes the last block. The model returns `{binary_logit, type_logits, embedding}`.

The fine-tuning defaults use separate learning rates: the backbone is fine-tuned at 3e-5 and the heads at 3e-4. Batch size is 16 with ×4 gradient accumulation, the window is 4 s, and positive weights are capped at 20.
