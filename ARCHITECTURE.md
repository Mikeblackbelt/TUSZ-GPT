# EEG-GPT Model Architecture

This document provides the formal architectural specification for `eeg_gpt`. The system processes multi-channel continuous scalp EEG through a hierarchical pipeline: channel-independent autoregressive foundation modeling followed by spatial channel integration, temporal context aggregation, and multi-task classification.

```
Input: Raw EEG (C channels × N blocks × T samples)
  │
  ▼
[AmplitudeTokenizer] ────────────────────────────────────────────────────────┐
  Quantizes continuous samples into L=512 discrete levels per channel         │
  Tokens: (B·C·N, T)                                                          │
  │                                                                           │
  ▼                                                                           │
[EEGGPT Backbone] (Stage 1 Pretrained Trunk)                                  │
  Channel-independent causal transformer with RoPE + Fourier value embeddings │
  Hidden states: (B·C·N, T, d)                                                │
  │                                                                           │
  ▼                                                                           │
[WindowEncoder]                                                               │
  1. Temporal pooling across samples T -> (B, N, C, d)                        │
  2. Channel positional embeddings + learned [CLS] token                      │
  3. Bidirectional spatial transformer over C channels                        │
  Output: Single spatial embedding per block -> (B, N, d)                     │
  │                                                                           │
  ▼                                                                           │
[TemporalContext] (Optional, when N > 1)                                      │
  Causal transformer or delta integration across N consecutive blocks         │
  Context embedding: (B, d)                                                   │
  │                                                                           │
  ▼                                                                           │
[Feature Fusion Branch] (Optional, data/features.py) ◄────────────────────────┘
  Concatenates 24 hand-crafted spectral/energy features per channel
  Fused representation: (B, d + d_feat)
  │
  ├─► [Binary Head] ────────► Seizure forecast logit (preictal vs. background)
  └─► [Multi-Type Head] ────► Seizure type logits (focal, generalized, etc.)
```

---

## 1. Amplitude Tokenizer (`AmplitudeTokenizer`)

Raw amplitude sequences are quantized into categorical token streams independently per channel. Categorical tokenization prevents representation collapse toward the running mean—a failure mode typical of autoregressive mean-squared-error predictors on oscillatory signals.

### Quantization Equations

Given input channel vector $x \in \mathbb{R}^T$:

1. **Normalization ($z$-score):**
   $$z = \frac{x - \mu}{\sigma + \epsilon}$$
   where $\mu, \sigma$ are computed per block (default), per session, or from a lead-4s causal window (`--norm-scope`).

2. **Symmetric Clipping:**
   $$z_{\text{clip}} = \text{clip}(z, -k, k), \quad k = 4.0$$
   Clipping at $\pm 4.0\sigma$ bounds extreme artifacts while dedicating maximum categorical resolution to physiological rhythms.

3. **Linear Quantization:**
   $$u = \frac{z_{\text{clip}} + k}{2k} \in [0, 1]$$
   $$\text{token} = \min\left(\lfloor u \cdot (L - 1) \rfloor, L - 1\right), \quad L = 512$$

4. **Dequantization (Continuous Reconstruction):**
   $$\hat{z} = \frac{\text{token}}{L - 1} \cdot 2k - k$$

### Channel-Independent Stream Formulation
Unlike models that interleave all channels into a single sequence (which fragments temporal rhythm into millisecond slices across electrodes), `eeg_gpt` processes each channel as an independent continuous sequence of 1024 samples (4.0 seconds at 256 Hz). Spatial identity is injected via a learned channel embedding.

---

## 2. Autoregressive Trunk (`EEGGPT`)

The backbone is a pre-LayerNorm, causal, decoder-only transformer running over individual channel sequences.

### Input Embedding
For a token $t \in \{0, \dots, L-1\}$ from channel $c \in \{0, \dots, C-1\}$:

$$h_0 = \text{Embedding}(t) + \text{ValueProj}\left(\text{Fourier}(\hat{z})\right) + \text{ChannelEmbedding}(c)$$

- **Discrete Token Embedding:** Standard lookup table $\mathbb{R}^{L \times d}$.
- **Fourier Value Embedding (`value_emb`):** Projects sinusoidal features of the dequantized amplitude $\hat{z}$ through a linear layer. This imparts a smooth inductive bias across adjacent numerical bins so neighboring amplitude levels start with correlated representations.
- **Channel Embedding:** Learned embedding $\mathbb{R}^{C \times d}$ identifying the electrode.

### Positional Encodings
Rotary Position Embeddings (RoPE) are applied directly to queries and keys within each self-attention head:
$$\mathbf{q}_m = \mathbf{R}_{\Theta, m} \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n} \mathbf{W}_k \mathbf{x}_n$$
RoPE enables relative temporal distance awareness and natural length generalization. (Learned absolute positional embeddings are supported as an optional backward-compatibility fallback).

### Transformer Block
Each of the $M$ layers applies causal multi-head self-attention followed by an MLP:
$$\mathbf{x}' = \mathbf{x} + \text{CausalSelfAttention}(\text{LN}(\mathbf{x}))$$
$$\mathbf{x}_{\text{out}} = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$$

- **Attention Mechanism:** Implemented using PyTorch `scaled_dot_product_attention(is_causal=True)`, dispatching to FlashAttention / memory-efficient kernels on supported hardware.
- **FFN:** $\text{Linear}(d \to 4d) \to \text{GELU} \to \text{Dropout} \to \text{Linear}(4d \to d)$.
- **Residual Initialization:** Attention projection and output MLP layers are initialized with standard normal scaling $\mathcal{N}(0, 0.02 / \sqrt{2M})$ following GPT-2 to prevent variance explosion with depth.

### Multi-Horizon Prediction Heads
Rather than predicting only the immediate next sample ($t+1$), the trunk features multiple linear prediction heads operating across distinct temporal offsets:
$$\mathcal{O} = \{1, 8, 32\}$$
At 256 Hz, these offsets correspond to:
- Offset 1: $\Delta t = 3.9\text{ ms}$ (local sample transition)
- Offset 8: $\Delta t = 31.2\text{ ms}$ (fast oscillatory phase / beta-gamma)
- Offset 32: $\Delta t = 125.0\text{ ms}$ (theta rhythm / wave morphology)

The offset-1 head weight is tied with the token embedding table: $\mathbf{W}_{\text{head}, 1} = \mathbf{W}_{\text{tok}}$.

### Pretraining Objective (`gpt_loss`)

$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{CE}@1} + \alpha_{\text{aux}} \sum_{o \in \{8, 32\}} \mathcal{L}_{\text{CE}@o} + \lambda_{\text{mse}} \mathcal{L}_{\text{MSE}}$$

1. **Cross-Entropy Loss:** Standard negative log-likelihood on discrete next tokens:
   $$\mathcal{L}_{\text{CE}@o} = -\frac{1}{T-o} \sum_{t=1}^{T-o} \log p\left(\text{token}_{t+o} \mid \mathbf{h}_t\right)$$
2. **Dual Continuous MSE Loss:** Computes the expected normalized amplitude from the offset-1 softmax distribution and measures squared error against the ground-truth continuous normalized sample:
   $$\hat{z}_t = \sum_{l=0}^{L-1} p\left(l \mid \mathbf{h}_t\right) \cdot \text{level\_val}_l$$
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left(\hat{z}_t - z_{t+1}\right)^2$$
   Defaults: $\alpha_{\text{aux}} = 0.5$, $\lambda_{\text{mse}} = 0.5$.

---

## 3. Spatial Aggregation (`WindowEncoder`)

For downstream classification, the representation of an entire multi-channel window $(C \times T)$ must be condensed into a single vector $\mathbf{e}_{\text{block}} \in \mathbb{R}^d$.

```
Per-channel sequences (C × T) 
  ──► EEGGPT Trunk (hidden states) 
  ──► Temporal Pooling (meanmax / tail / attn) 
  ──► C Channel Vectors (C × d) 
  ──► Prepend [CLS] Token + Channel Position Embeddings
  ──► Non-Causal Spatial Transformer (2 layers, 4 heads)
  ──► LayerNorm([CLS] Output) ──► Block Vector (d)
```

1. **Temporal Pooling over Samples:**
   - **`meanmax` (Default):** Concatenates mean pooling and max pooling across time, then projects back to $d$:
     $$\mathbf{v}_c = \mathbf{W}_p \left[ \frac{1}{T}\sum_t \mathbf{h}_{c,t} \;\|\; \max_t \mathbf{h}_{c,t} \right]$$
     This preserves both sustained baseline background activity and transient, high-amplitude epileptiform spikes.
   - **`tail`:** Averages hidden states over the final fraction (e.g. last 25%) of the causal block.
   - **`attn`:** Multi-query cross-attention pooling over time.
2. **Channel Integration:**
   - A learned $\text{[CLS]}$ token is prepended to the $C$ pooled channel vectors: $\mathbf{Z}_0 = [\mathbf{e}_{\text{CLS}}, \mathbf{v}_1, \dots, \mathbf{v}_C] \in \mathbb{R}^{(1+C) \times d}$.
   - Learned channel spatial position embeddings are added.
   - A 2-layer bidirectional (non-causal) transformer applies self-attention across the electrodes, allowing the model to learn cross-montage spatial correlations (e.g. left vs. right hemisphere asymmetry, anterior-posterior gradients).
   - The normalized output at the $\text{[CLS]}$ position serves as the block embedding: $\mathbf{e}_{\text{block}} = \text{LN}(\mathbf{Z}_{M, 0})$.

---

## 4. Hierarchical Temporal Context (`TemporalContext`)

Seizure forecasting benefits from observing trend evolution across consecutive windows rather than isolated 4-second snapshots.

When `--context-blocks N` is set ($N > 1$, typically $N=4$ for 16 seconds or $N=8$ for 32 seconds):
1. The $N$ blocks are encoded sequentially by `WindowEncoder` into a sequence $[\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_N] \in \mathbb{R}^{N \times d}$.
2. Block temporal positional encodings are added.
3. A 2-layer **causal** transformer processes the sequence from oldest to newest block.
4. The output vector at the newest position $\mathbf{e}_N$ is returned as the final temporal representation.

*Alternative Mode (`--context-mode delta`):* Directly computes $\mathbf{e}_{\text{final}} = \mathbf{e}_N + \mathbf{W}_\Delta \left(\mathbf{e}_N - \frac{1}{N-1}\sum_{j=1}^{N-1}\mathbf{e}_j\right)$, offering an efficient parameter-light difference signal.

---

## 5. Hand-Crafted Feature Branch (`data/features.py`)

Because standard $z$-score normalization discards absolute energy and amplitude variance, a parallel feature branch computes 24 domain-standard physiological signal features per channel on raw (un-normalized) EEG:

| Index | Feature | Description |
|---|---|---|
| `0` | Log Absolute Variance | Re-introduces signal power stripped by per-block normalization |
| `1-5` | Log Relative Band Power | Classical EEG bands: $\delta$ (0.5–4 Hz), $\theta$ (4–8 Hz), $\alpha$ (8–13 Hz), $\beta$ (13–30 Hz), $\gamma$ (30–45 Hz) |
| `6` | Spectral Edge Frequency 90% | Frequency below which 90% of spectral power resides |
| `7` | Line Length | Waveform complexity: $\log \frac{1}{T}\sum |x_t - x_{t-1}|$ |
| `8-9` | Hjorth Mobility & Complexity | Signal dynamics based on first and second derivative variance |
| `10-11` | Skewness & Kurtosis | 3rd and 4th standardized statistical moments |
| `12-13` | Spectral Edge 50% & 75% | Median and 75% spectral power cutoffs |
| `14` | Spectral Entropy | Normalized Shannon entropy of the power spectrum |
| `15` | Spectral Flatness | Wiener entropy (geometric mean / arithmetic mean of spectrum) |
| `16-17` | Spectral Centroid & Spread | Center frequency and spectral standard deviation |
| `18` | Slow / Fast Energy Ratio | $\log \frac{\text{power}(\delta + \theta)}{\text{power}(\alpha + \beta)}$ |
| `19` | $\theta / \alpha$ Ratio | Relative rhythmic deceleration index |
| `20` | Zero-Crossing Rate | Rate of sign changes of mean-centered waveform |
| `21` | Log Teager-Kaiser Energy | Non-linear instantaneous energy operator |
| `22` | 2nd Difference Line Length | High-frequency curvature measure |
| `23` | Log Peak-to-Peak Amplitude | $\log (\max(x) - \min(x))$ |

When `--feat-branch full` is enabled, the 24 features per channel are computed across context blocks, projected through an MLP, and concatenated with the transformer latent representation.

---

## 6. Downstream Heads & Optimization Recipe

### Prediction Heads
- **Binary Forecasting Head:** $\text{Linear}(d \to 2d) \to \text{GELU} \to \text{Dropout}(0.2) \to \text{Linear}(2d \to 1)$.
- **Multi-Type Seizure Head:** $\text{Linear}(d \to 2d) \to \text{GELU} \to \text{Dropout}(0.2) \to \text{Linear}(2d \to K)$, where $K$ is the number of target seizure categories (e.g. focal vs. generalized).

### Prior Log-Odds Initialization
To stabilize training under severe class imbalance (where preictal windows represent <3% of the corpus), the final linear layer of the binary head is initialized with weights set to 0 and bias set to the empirical log-odds:
$$b_0 = \log\left(\frac{\pi}{1 - \pi}\right)$$
This prevents destructive early gradient spikes by ensuring the model predicts the prior distribution at step 0.

### Focal Binary Cross-Entropy
$$\mathcal{L}_{\text{focal}} = -(1 - p_t)^\gamma \log(p_t)$$
where $p_t = p$ for positive instances and $1-p$ for negative instances. With $\gamma = 2.0$, easy background windows are suppressed, focusing gradient updates on hard, ambiguous preictal boundaries.

### Linear-Probe-Then-Fine-Tune (LP-FT) with LLRD
1. **Warmup Phase (`--freeze-epochs 3`):** The pretrained backbone is frozen while the spatial aggregator, temporal context, and heads are trained with AdamW ($\text{lr} = 10^{-3}$).
2. **Fine-Tuning Phase:** The backbone is unfrozen using Layer-wise Learning Rate Decay (LLRD, `--llrd 0.65`). Deeper layers receive smaller learning rates:
   $$\eta_l = \eta_{\text{base}} \cdot \alpha^{M - 1 - l}$$
3. **Weight Averaging:** An Exponential Moving Average (EMA, `--ema 0.999`) tracks weights throughout fine-tuning, providing robust checkpoints for validation and test evaluation.

---

## 7. Hyperparameter Summary

| Module | Parameter | Default Value | Description |
|---|---|---|---|
| **Tokenizer** | Quantization Levels ($L$) | `512` | Discrete amplitude levels |
| | Clip Threshold ($k$) | `4.0` | $z$-score clipping range $(\pm 4.0\sigma)$ |
| | Block Size ($T$) | `1024` | Samples per window (4.0 s @ 256 Hz) |
| **Backbone** | Hidden Dimension ($d$) | `256` | Transformer representation width |
| | Layers ($M$) | `6` | Number of causal transformer layers |
| | Attention Heads | `4` | Self-attention heads |
| | FFN Expansion | `1024` | 4× hidden dimension |
| | Dropout | `0.0` | Pretraining dropout (no overfitting on raw corpus) |
| | Prediction Offsets | `1, 8, 32` | Multi-timescale prediction steps |
| | $\lambda_{\text{MSE}}$ | `0.5` | Weight for continuous dequantized MSE loss |
| **Aggregator** | Aggregation Layers | `2` | Spatial transformer layers across channels |
| | Pooling Mode | `meanmax` | Temporal pooling strategy over samples |
| | Context Blocks ($N$) | `4` | Number of 4-second blocks in temporal history |
| **Training** | Optimizer | Fused AdamW | $\beta_1=0.9, \beta_2=0.95, \text{weight decay}=0.05$ |
| | Learning Rate (Pretrain) | `3e-4` | Warmup 1000 steps $\to$ cosine decay to `1e-5` |
| | Learning Rate (Fine-tune)| `1e-4` | Base trunk rate after LP-FT unfreezing |
| | Head Learning Rate | `1e-3` | Classifier head rate |
| | Preictal Batch Fraction | `0.25` | Reserved positive slots per training batch |
| | Focal Loss $\gamma$ | `2.0` | Focusing parameter for binary head |
