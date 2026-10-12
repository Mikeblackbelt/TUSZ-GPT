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
| clip | `z_clip = clip(z, −k, k)`, k = **4.0** (tighter than before → more dynamic range per bin) |
| scale | `u = (z_clip + k) / 2k ∈ [0,1]` |
| quantize | `token = floor(u · (L−1))`, L = 512 |
| dequantize | `ẑ = level/(L−1) · 2k − k` |

Unlike the paper, each channel gets **its own token stream**. The paper interleaves channels into one stream. Channel identity goes into the model as an embedding instead.

## 2. Backbone: `EEGGPT`

A pre-LN, decoder-only causal transformer that runs over the amplitude tokens of **one channel**.

- **Input embedding** = `tok_emb(token) + val_proj(fourier(amplitude of token)) + ch_emb(channel)`, followed by dropout. The Fourier term (`value_emb`) gives the 512 ordered levels a smooth code so neighbouring levels start out similar.
- **Positions**: rotary (RoPE) on q/k inside attention (`rope`, default for new pretraining runs); with `rope=False` a learned absolute `pos_emb(t)` is added to the input instead (old checkpoints).
- **Blocks** (×`layers`): `x + Attn(LN(x))`, then `x + FFN(LN(x))`
  - Attention uses `F.scaled_dot_product_attention(is_causal=True)`, which enables flash/memory-efficient kernels and needs no T×T mask
  - FFN: Linear → GELU → Dropout → Linear → Dropout
- **Final LN** leads into one or more **LM heads**. There is one `Linear(d, 512, bias=False)` per prediction offset (`pred_offsets`, default `1,8,32` for multi-token prediction); the offset-1 head shares its weight with `tok_emb` (`tie_weights`)
- Init: normal(0, 0.02); **residual projections** (`attn.proj`, last FFN linear) scaled by `1/√(2·L)` (GPT-2 style) to prevent hidden-state variance growth with depth

**Default hyperparameters (pretrain.py):**

| param | default | note |
|---|---|---|
| n_levels | 512 | |
| d | **256** | was 128 |
| layers | **6** | was 4 |
| heads | 4 | |
| ffn | **1024** | was 512 |
| block_size | 1024 tokens (4 s @ 256 Hz) | |
| dropout | **0.0** | single pass over the corpus, nothing to overfit |
| rope / value_emb / tie_weights | **on** | `--no-rope` etc. give the old trunk |
| n_channels | 20 | |
| pred_offsets | **1,8,32** | multi-timescale (4 ms, 31 ms, 125 ms) |
| clip_k | **4.0** | tighter clipping |

### Pretraining loss: `gpt_loss`

- **Cross-entropy** for every offset `o`: the logits at position t predict `tokens[t+o]`. Label smoothing is **off** by default (uniform smoothing over 512 ordered levels floors each head at ~0.95 nats)
- **Dual MSE** (Mahdi §III-E), offset-1 head only: `ẑ = Σ softmax(logits) · level_values`, then `MSE(ẑ, z_true)`, weighted by `lambda_mse` (**default 0.5**, up from 0.1)
- `loss = CE@1 + aux_weight · Σ_{o>1} CE@o + λ·MSE` (`aux_weight` 0.5)
- The logged `ce@o` are always plain cross-entropies; `ce@1` is the number to compare between runs and what `best.pt` is selected on. It has a tokenizer-set floor: a next-sample residual of r bins RMS costs ≈ ln(4.13·r) nats, so 2.0 nats ≈ 1.8 bins ≈ 0.03 σ.

Optimizer: fused AdamW (β = 0.9/0.95, wd 0.05; embeddings and 1-D params are not decayed), lr 3e-4 with **1000-step warmup** → cosine decay to **1e-5** (not 1e-6). Mixed precision is on by default on CUDA (bf16 where supported); `--compile` adds `torch.compile`.

Data: `PretrainSessions` samples from a pool of `--session-pool` (16) recordings per worker; each pool slot keeps its recording for `--samples-per-session` (256) draws, so a full-parquet read is amortized over hundreds of samples instead of being paid on nearly every one.

## 3. Fine-tuning: `SeizureModel`

### `WindowEncoder`: (B, C, T) → (B, d)
1. It runs the backbone's `hidden()` on every channel sequence. Channels are processed in chunks of `seq_chunk` = 256 to bound activation memory.
2. It pools over time. **`meanmax`** is the new default (average + peak captures both sustained and transient ictal patterns); `mean` and `last` are the alternatives.
3. It adds learned channel-position embeddings and prepends a learned **CLS** token.
4. It runs `agg_layers` (**default 2**, up from 1) **non-causal** transformer blocks over the 1+C tokens (4 heads, ffn = 4d).
5. The output is LN(CLS).

### `TemporalContext`: (B, N, d) → (B, d), optional
This module is used only when `context_blocks > 1`. It adds learned block-position embeddings, then applies 2 **causal** transformer blocks over the N windows (oldest → newest) and LN. It returns the newest position.

### Heads: `Head`
`LayerNorm → GELU → Dropout → Linear(d, 2d) → GELU → Dropout → Linear(2d, n_out)` (wider than paper's d→d), dropout 0.2.
- `binary_head`: 1 logit (preictal vs background, or the seizure-within-horizon variant; default horizon is 30 s)
- `type_head`: `n_types` logits (multi-type seizure classification)

### Binary loss: `focal_bce`
The binary preictal/background head now uses **focal BCE** (Lin et al. 2017) with γ = 2.0:

```
focal_bce(logit, y) = (1 - p_t)^γ · BCE(logit, y)
```

This downweights easy background windows (p_t ≫ 0) and concentrates gradient on hard/uncertain preictal examples, directly addressing low sensitivity.

### Forward
`tokens (B, C, N, T)` goes through the encoder once per block to give `(B, N, d)`. If `context_blocks > 1`, TemporalContext runs next; otherwise the model takes the last block. The model returns `{binary_logit, type_logits, embedding}`.

### Fine-tuning defaults

| param | default | change |
|---|---|---|
| context_blocks | **4** (16 s) | was 1 |
| pool | **meanmax** | was mean |
| pos_weight_cap | **50** | was 20 |
| focal_gamma | **2.0** | new |
| head_lr | **1e-3** | was 3e-4 |
| warmup | **5 % of optimizer steps** (`--warmup-frac`) | was a fixed 200 |
| eval batch | **4× train batch** (`--eval-batch-size`) | was 1× |

`--freeze-backbone` runs the trunk in eval mode under `no_grad` (no activation graph, no dropout in frozen layers).

### Evaluation metrics
In addition to AUROC and threshold-sweep balanced accuracy, the evaluation now reports **sensitivity at fixed specificity** (80%, 90%, 95%), which is the clinically relevant metric for seizure prediction with a bounded false-alarm rate.

## 4. Patient-Specific Training & Candidate Sweeps

Source: `eeg_gpt/training/patient_sweep.py`, `eeg_gpt/training/finetune.py`.

### Patient-Specific Initialization Modes
1. **Pretrained-only trunk** (`--patient-base pretrained`):
   - Loads Stage 1 autoregressive next-token `EEGGPT` trunk (`--pretrained`).
   - Re-initializes spatial aggregation (`WindowEncoder`), `TemporalContext`, and classification heads fresh.
   - Learns patient-specific aggregation and decision boundaries from the pretrained EEG dynamics.
2. **Non-patient-specific general model** (`--patient-base general`):
   - Loads full Stage 2 `SeizureModel` (`--general-checkpoint`), which was already trained across the cross-patient population.
   - Adapts both representation and prediction heads to the target patient, capitalizing on general seizure forecasting features.
3. **Comparative A/B** (`--patient-base both`):
   - Evaluates both initializations on the candidate patient and reports the empirical transfer delta: $\text{AUROC}_{\text{general}} - \text{AUROC}_{\text{pretrained}}$.

### 5-Fold Cross-Validation
- `--cv-folds 5` partitions patient recordings into 5 folds.
- `--cv-strategy auto`:
  - When a patient has $\ge 5$ recording sessions with seizures across sessions, splits by whole session to guarantee zero cross-recording temporal leakage.
  - When sessions $< 5$ or seizures are concentrated, splits by stratified windows to balance preictal and background representations across all 5 evaluation folds.
- Aggregates mean ± std for AUROC, balanced accuracy, and sensitivity at fixed specificities (80%, 90%, 95%).

### Candidate Patient Sweeping
- `--sweep-patients` scans all patients in `master_full.csv`.
- Analyzes session counts, preictal window volume, background duration, and class balance.
- Identifies **ideal candidates** (minimum sessions $\ge 2$, preictal windows $\ge 5$, background windows $\ge 20$, and viable 5-fold partitioning).
- Computes composite suitability scores, ranks candidate patients, and outputs structured profile and benchmark tables (`candidates_profile.csv`, `patient_sweep_results.csv`, `patient_sweep_results.json`).
