# eeg_gpt

Autoregressive GPT pretraining and downstream seizure forecasting / classification for multi-channel scalp EEG, built for the Temple University Seizure Corpus (TUSZ).

The codebase implements a two-stage modeling approach:
1. **Self-Supervised Pretraining (Stage 1):** Next-token and multi-horizon autoregression across 20 bipolar EEG channels using discrete amplitude quantization, Rotary Position Embeddings (RoPE), and continuous amplitude regression loss. Pretraining runs over all available recording sessions (including unlabeled background, postictal, and inter-seizure intervals).
2. **Downstream Transfer (Stage 2):** Seizure forecasting (preictal vs. background detection) and multi-type seizure classification. Channel representations are extracted by the pretrained trunk, aggregated across channels via a spatial transformer, and integrated across temporal blocks with a causal temporal context module.

The evaluation protocol enforces leak-free patient-disjoint splits and event-grouped cross-validation, reporting both window-level metrics and clinical event-level statistics (sensitivity at fixed specificity, false alarms per hour) against a linear band-power baseline.

---

## Repository Layout

```
eeg_gpt/
├── data/
│   ├── tokenizer.py        # AmplitudeTokenizer: z-score, clipping, 512-level quantization
│   ├── datasets.py         # SessionStore (parquet LRU cache), PretrainSessions, LabeledWindows
│   ├── events.py           # Seizure event identification, onset tracking, postictal margins
│   ├── features.py         # Hand-crafted 24-dim spectral/temporal features per channel
│   └── label_utils.py      # TUSZ label mapping, classification, and grouping logic
├── models/
│   └── gpt.py              # EEGGPT trunk, WindowEncoder, TemporalContext, SeizureModel
├── training/
│   ├── pretrain.py         # Stage 1: Autoregressive next-token + MSE training on raw sessions
│   ├── finetune.py         # Stage 2: Seizure forecasting & type classification fine-tuning
│   ├── pipeline.py         # End-to-end pipeline: patient profiling, pretraining, and 5-fold CV
│   ├── patient_sweep.py    # Patient profiling, candidate ranking, and 5-fold cross-validation
│   ├── recipe.py           # Training recipe: LP-FT warmup, class balancing, LLRD, EMA
│   ├── metrics.py          # Window AUROC, threshold sweeps, sensitivity at fixed specificity
│   ├── event_metrics.py    # Seizure-level sensitivity, false alarms per hour, firing power
│   ├── device.py           # Hardware allocation, multi-worker setup, AMP scaler
│   └── probe.py            # Frozen-trunk linear probe evaluation
├── tools/
│   └── check_csv_labels.py # CSV verification: label distributions, postictal flags, patient counts
├── patients.py             # CLI utility for ranking and profiling candidate patients
├── hardware.py             # System affinity, cgroup memory limits, cache sizing
└── tests/
    ├── test_pipeline.py    # End-to-end pipeline test on synthetic TUSZ corpus (CPU)
    ├── test_arch.py        # Architecture, tensor shapes, forward/backward, probing
    ├── test_protocol.py    # Cross-validation splitting, temporal leakage prevention
    ├── test_recipe.py      # Optimization recipe: LP-FT, class balancing, EMA
    └── test_stage1.py      # Stage 1 tokenizer, causal masking, multi-offset objectives
```

---

## Installation & Environment

### Requirements
- Python >= 3.9
- PyTorch >= 2.1 (compiled with CUDA support for `torch.nn.functional.scaled_dot_product_attention`)
- Dependencies listed in `requirements.txt`

```bash
git clone <repository-url>
cd tusz-gpt

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Verification Test
Run the end-to-end test suite to verify the environment and model pipeline. This creates a synthetic TUSZ-like corpus in a temporary directory and validates tokenization, causal masking, pretraining, fine-tuning, and cross-validation on CPU:

```bash
python -m eeg_gpt.tests.test_pipeline
```

---

## Data Specifications

The pipeline expects preprocessed TUSZ recordings and a master window annotation file.

### 1. Preprocessed Parquet Files (`<session_key>_proc.parquet`)
- **Location:** Specified via `--checkpoint-dir /path/to/parquet_dir`
- **Format:** Apache Parquet, with one column per EEG channel and rows corresponding to sampled time steps.
- **Montage:** 20 bipolar channels (standard TCP 10-20 derived bipolar montage).
- **Sampling Rate:** 256 Hz (default).
- **Session Keys:** Formatted as `[split_]patientid_sNNN_...` (e.g., `trn_aaaaaajy_s001_t000_proc.parquet`).

### 2. Window Annotation File (`master_full.csv`)
- **Location:** Specified via `--master-csv /path/to/master_full.csv`
- **Required Columns:**
  - `session_key`: Filename stem of the recording.
  - `start_time`, `stop_time`: Start and end times in seconds for each window (default window length: 4.0 s, 1024 samples).
  - `channel`: Filtered to `TERM` (montage-wide summary row).
  - `is_valid`: Boolean flag indicating valid recording quality.
  - `label`: Window classification label:
    - Background: `bg`, `bckg`
    - Preictal: `p<type>` (e.g., `pfnsz`, `pgnsz`, `pcpsz`)
    - Buffer: `p<type>_sopbuffer` (excluded from background/preictal evaluation)
    - Ictal: `<type>` (e.g., `fnsz`, `gnsz`, `cpsz`)

### 3. Clinical Prediction Horizon Definitions
- **SPH (Seizure Prediction Horizon):** 120 seconds (2 minutes). The intervention buffer immediately preceding seizure onset. Windows falling inside the SPH are excluded from training and evaluation scoring.
- **SOP (Seizure Occurrence Period):** 420 seconds (7 minutes). The target forecasting window prior to the SPH.
- **Preictal Definition:** Windows occurring in the interval $[t_{\text{onset}} - (\text{SOP} + \text{SPH}),\ t_{\text{onset}} - \text{SPH}]$, i.e. 2 to 9 minutes before seizure onset.

### 4. Data Inspection Utilities
Before launching training, inspect label distributions and candidate patient counts:

```bash
# Verify master CSV labels, postictal flags, and usable patient distributions
python -m eeg_gpt.tools.check_csv_labels --master-csv /path/to/master_full.csv

# Profile and rank candidate patients for patient-specific modeling
python -m eeg_gpt.patients --master-csv /path/to/master_full.csv
```

---

## Training Workflows

### Stage 1: Self-Supervised Pretraining

Stage 1 trains the `EEGGPT` trunk autoregressively across all available sessions, learning general temporal EEG dynamics without supervision.

```bash
python -m eeg_gpt.training.pretrain \
    --checkpoint-dir /path/to/parquet_dir \
    --out runs/pretrain_base \
    --block-size 1024 \
    --steps 100000 \
    --batch-size 64 \
    --d 256 \
    --layers 6 \
    --heads 4 \
    --ffn 1024 \
    --pred-offsets 1,8,32 \
    --amp \
    --num-workers 8
```

**Key Parameters:**
- `--pred-offsets 1,8,32`: Multi-horizon prediction targets (4 ms, 31 ms, 125 ms at 256 Hz), forcing the trunk to capture both high-frequency local dynamics and lower-frequency rhythmic structures.
- `--lambda-mse 0.5`: Dual continuous MSE loss on dequantized expected amplitude levels, regularizing the categorical cross-entropy head.
- `--val-sessions 16`: Number of held-out patient sessions used for tracking validation perplexity and linear probe convergence.

**Evaluating Convergence:**
- Monitor validation `ce@1` (cross-entropy at offset 1). A healthy pretraining run moves from the unigram token entropy (~4.2 nats) down towards 2.0–2.3 nats.
- The `best.pt` checkpoint is selected based on held-out `ce@1`.

---

### Stage 2: Cross-Patient Fine-Tuning

Stage 2 attaches spatial channel aggregation (`WindowEncoder`), temporal context modeling (`TemporalContext`), and prediction heads (binary forecasting and multi-type classification) to the pretrained trunk.

```bash
# Fine-tune pretrained trunk
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --pretrained runs/pretrain_base/best.pt \
    --out runs/ft_pretrained \
    --context-blocks 4 \
    --preictal-frac 0.25 \
    --freeze-epochs 3 \
    --llrd 0.65 \
    --baseline \
    --amp

# From-scratch baseline control (same architecture, randomly initialized trunk)
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --out runs/ft_scratch \
    --context-blocks 4 \
    --preictal-frac 0.25 \
    --freeze-epochs 3 \
    --baseline \
    --amp
```

**Optimization & Regularization Recipe:**
- `--context-blocks 4`: Hierarchical temporal context spanning 4 consecutive 4-second blocks (16 seconds of history).
- `--preictal-frac 0.25`: Reserves 25% of each training batch for preictal samples to address extreme class imbalance (typically <3% positive windows).
- `--freeze-epochs 3` & `--llrd 0.65`: Linear-probe-then-fine-tune (LP-FT). Keeps the backbone frozen for 3 epochs while training heads and aggregator, then unfreezes with layer-wise learning rate decay ($0.65^l$).
- `--feat-branch full`: Fuses hand-crafted time/frequency features with transformer representations before final classification.
- `--baseline`: Fits a linear ridge baseline using hand-crafted spectral/energy features on the exact same training split and evaluates it alongside the neural model.

---

### Patient-Specific Adaptation (5-Fold Cross-Validation)

Because EEG signatures vary substantially across individuals, patient-specific adaptation tailors the model to a single patient's recordings under a 5-fold cross-validation scheme.

```bash
# Mode A: Adapt Stage 1 pretrained trunk to patient
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --patient aaaaaajy \
    --patient-base pretrained \
    --pretrained runs/pretrain_base/best.pt \
    --cv-folds 5 \
    --context-blocks 4 \
    --out runs/pt_pre_aaaaaajy \
    --amp

# Mode B: Adapt general Stage 2 model to patient
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --patient aaaaaajy \
    --patient-base general \
    --general-checkpoint runs/ft_pretrained/best.pt \
    --cv-folds 5 \
    --context-blocks 4 \
    --out runs/pt_gen_aaaaaajy \
    --amp
```

---

### Candidate Patient Sweep & Ranking

Identify and benchmark which patients in `master_full.csv` have sufficient data volume, event diversity, and recording sessions for rigorous 5-fold cross-validation:

```bash
# Profile and rank candidates without training (generates candidates_profile.csv)
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --sweep-patients --sweep-profile-only \
    --out runs/patient_candidates

# Benchmark top-5 candidate patients across 5-fold CV, comparing pretrained vs general initialization
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --sweep-patients --patient-base both \
    --pretrained runs/pretrain_base/best.pt \
    --general-checkpoint runs/ft_pretrained/best.pt \
    --sweep-top-k 5 --cv-folds 5 \
    --out runs/candidate_sweep --amp
```

---

### Automated End-to-End Pipeline

The pipeline module coordinates candidate ranking, patient-disjoint Stage 1 pretraining folds, and patient-specific Stage 2 cross-validation in a single reproducible command:

```bash
python -m eeg_gpt.training.pipeline \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --out runs/pipe_experiment \
    --top-k 3 \
    --cv-folds 5 \
    --pretrain-args "--steps 30000 --batch-size 64 --amp" \
    --finetune-args "--epochs 10 --context-blocks 4 --amp"
```

1. **Profiles** candidate patients and selects the top-K candidates (or accepts an explicit list via `--patients id1,id2`).
2. **Holds out** candidate patients from Stage 1 pretraining. Splits the remaining population into 5 disjoint folds and pretrains 5 corresponding trunk checkpoints.
3. **Fine-tunes** 5-fold CV models for each candidate patient, matching pretrain fold $k$ with fine-tuning fold $k$ (`--pretrain-select paired`).
4. **Writes comprehensive reports** to `runs/pipe_experiment/SUMMARY.md`.

---

### Standalone Checkpoint Evaluation

Evaluate an existing checkpoint without running training:

```bash
python -m eeg_gpt.training.finetune \
    --master-csv /path/to/master_full.csv \
    --checkpoint-dir /path/to/parquet_dir \
    --eval-only \
    --load runs/ft_pretrained/best.pt \
    --context-blocks 4 \
    --baseline
```

---

## Evaluation Methodology & Clinical Metrics

### Preventing Temporal Data Leakage
Standard random window-level train/validation splitting is invalid for continuous EEG. Consecutive 4-second windows share strong background autocorrelation, allowing models to achieve inflated AUROC (>0.90) simply by memorizing session-specific spectral baselines rather than learning preictal dynamics.

This framework enforces two leak-free partitioning strategies:
- **Session-disjoint splits:** Whole recording sessions are placed entirely into either train or evaluation splits.
- **Event-grouped folds:** When partitioning within a patient, all preictal windows preceding a seizure event and the contiguous background segments are grouped together. Contiguous background blocks are chunked with protective time margins to eliminate boundary leakage.

### Reported Metrics
Every evaluation logs:
- **Window AUROC:** Area under the ROC curve on individual 4-second evaluation windows.
- **Sensitivity at Fixed Specificity:** Clinically actionable operating points reporting true-positive sensitivity when false-positive rates are held at 80%, 90%, and 95% specificity.
- **Firing Power Smoothing:** Causal moving average of window scores over sliding durations (5, 15, and 60 blocks) to simulate real-time alarm filtering.
- **Event-Level Statistics:**
  - `sens_events`: Percentage of seizure events preceded by an alarm within the preictal horizon.
  - `fa_per_h`: False alarm rate per hour of non-preictal recording time.
  - Comparison against Poisson chance sensitivity at the identical false alarm rate.
- **Feature Baseline Comparison:** A linear model trained on 24 hand-crafted spectral/energy features per channel is evaluated on the exact same splits. A deep model must outperform this baseline to demonstrate meaningful representation learning.

---

## Hardware Sizing & Performance

Benchmarked on modern GPU workstations (NVIDIA A100 / RTX 3090, PyTorch 2.1, mixed precision `--amp`):

| Task | Configuration | Approx. Throughput | Memory Footprint |
|---|---|---|---|
| **Stage 1 Pretrain** | Batch 64, Block 1024, $d=256$, 6 layers | ~0.10 s / step (~640 samples/s) | ~6.5 GB VRAM |
| **Stage 2 Fine-tune** | Batch 16, Context 1 block (4 s) | ~0.08 s / step | ~3.8 GB VRAM |
| **Stage 2 Fine-tune** | Batch 16, Context 4 blocks (16 s) | ~0.24 s / step | ~5.2 GB VRAM |
| **Stage 2 Fine-tune** | Batch 16, Context 8 blocks (32 s) | ~0.52 s / step | ~7.8 GB VRAM |

**Resource Controls:**
- `--num-workers 8`: Sets background DataLoader worker processes.
- `--cache-capacity 8`: Number of `.parquet` sessions retained in memory per worker via `SessionStore`.
- `--max-gpu-usage 0.9`: Bounds PyTorch CUDA memory allocation to avoid out-of-memory errors on shared servers.
- `EEG_GPT_CACHE_GB`: Environment variable overriding total host RAM budget for recording caches.

---

## Technical Reference & Architecture

For mathematical formulations of tokenization, rotary positional embeddings, spatial/temporal transformer modules, loss equations, and feature extraction definitions, see [ARCHITECTURE.md](file:///mnt/c/Users/mmati/tusz-gpt/ARCHITECTURE.md).

---

## Citations & Related Work

This architecture draws from and builds upon several key developments in EEG representation learning:
- **Mahdi & Baghdadi (2026):** *Patient-Adaptive Transformer Networks for Seizure Prediction* (arXiv:2603.26821) — Discrete amplitude quantization, dual cross-entropy/MSE pretraining loss, and patient adaptation protocols.
- **BrainGPT / EEGPT (Yue et al., 2024):** *BrainGPT: Electrode-wise autoregressive foundation models* (arXiv:2410.19779) — Independent electrode sequence modeling and decoupled spatial aggregation.
- **Neuro-GPT (Cui et al., 2023):** Generative pretraining for continuous EEG representations.
- **LaBraM (Jiang et al., ICLR 2024):** Large brain models using patch-level discrete representations on unlabeled clinical corpora.
- **Snyder et al. (2008):** *Seizure prediction in epilepsy: the prospective challenge* — Clinical event-level metrics, false alarms per hour, and Poisson chance bounds.
