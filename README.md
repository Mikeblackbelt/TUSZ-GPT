# eeg_gpt — GPT pretraining + binary + multitype seizure heads (rebuild from scratch)

A new model on top of the existing preprocessing (TUSZ-Seizure-Predict: 20 bipolar channels,
0.5–40 Hz, 256 Hz, `<session_key>_proc.parquet` full recordings + `master_full.csv` with
`bg / p<type> / <type>_sopbuffer / <type>` rows, SPH 120 s / SOP 420 s). Nothing from the old
Conformer repo is reused except `label_utils.py` (your label semantics, verbatim).

```
eeg_gpt/
  data/tokenizer.py        amplitude tokenizer (Mahdi 2026 §III-D) + dequantization
  data/datasets.py         SessionStore (parquet LRU), PretrainSessions (all recordings, no labels),
                           LabeledWindows (master CSV rows -> per-channel tokens, context of N blocks)
  data/label_utils.py      your label logic (copied)
  models/gpt.py            EEGGPT trunk, WindowEncoder, TemporalContext, binary + multitype heads
  training/pretrain.py     Stage 1: next-token CE + dual MSE on every session (patient-disjoint val)
  training/finetune.py     Stage 2: binary (preictal | horizon) + type heads, full evaluation
  training/metrics.py      AUROC, sweep, per-session AUROC, firing-power smoothing, band-power baseline
  tests/test_pipeline.py   synthetic TUSZ-shaped corpus; runs every path end to end on CPU
```

## 1. Studies this is built on, and what each contributes

| study | what it does | what is taken |
|---|---|---|
| **Mahdi & Baghdadi 2026**, *Patient-Adaptive Transformer Networks*, arXiv 2603.26821 | z-score → clip ±5σ → 512 amplitude levels, one token per sample; GPT (d 128, 4 layers, 4 heads, learned positions, block 512) trained with next-token cross-entropy + λ=0.1 · MSE on the dequantized expected level; fine-tune with mean-pool → LN → GELU → dropout → FC, weighted CE, lr 3e-5, full fine-tuning; label = onset within 30 s; patient-specific | the tokenizer, the dual loss, the trunk size, the head, the fine-tune recipe, the horizon label (as an option), the patient-adaptive protocol (as an evaluation mode) |
| **BrainGPT / EEGPT** (Yue et al. 2024, arXiv 2410.19779) | electrode-wise modeling: each electrode is its own sequence with a learnable electrode token; autoregressive next-token pretraining; scaling laws in data and model size; spatial integration added at transfer time | one sequence per channel + channel embedding (instead of interleaving channels); channel aggregation as a separate module after the trunk |
| **Neuro-GPT** (Cui et al.) | EEG-Conformer encoder + GPT predicting the next chunk embedding, pretrained on TUH, fine-tuned on a small labeled set; the GPT is an auxiliary component, the encoder is what transfers | the trunk is the product; nothing is generated at inference |
| **LaBraM** (Jiang et al., ICLR 2024), **EEG2Rep**, **BENDR** | discrete/latent targets and masking at ~1 s patch scale on large unlabeled corpora | discrete CE targets (no MSE-to-the-mean collapse); optional multi-offset targets so the trunk must model beyond one-sample autocorrelation |
| **EEG-Conformer** (Song et al. 2023) + our rounds 9–10 | supervised classifier; our controlled runs: 39 ms ELU tokens fail on weak spectral signal, log band-power tokens and temporal integration help, per-4-s decisions need smoothing | mean-pooling over 1024 causal hidden states per channel is a learned, power-like summary; context of N blocks; firing-power smoothing and per-session AUROC in every evaluation; the linear band-power baseline as the floor |

## 2. What Mahdi 2026 gets right, what had to change

Kept: discrete amplitude tokens with a cross-entropy objective (a categorical target cannot be
"predicted to the mean" and cannot shrink — the two failure modes that killed the old
generative stage), the dual MSE for waveform fidelity, a small trunk, a plain pooled head, weighted
CE, low fine-tune learning rate with a separate higher head rate.

Changed, with the reason:

1. **Channels.** The paper flattens all channels into one stream, so a 512-token block spans
   ~15 time steps (~60 ms) — no rhythm fits in the context. Here every channel is its own
   1024-token sequence (4 s) with a channel-ID embedding; channel interaction is learned by a
   one-layer transformer over the 20 channel embeddings after the trunk.
2. **Corpus.** The paper pretrains and fine-tunes *per patient* on 0.07–0.42 h. Here Stage 1
   runs over **every** parquet recording, all of it (including the postictal, buffer and
   unlabeled stretches the master CSV never touches), with patient-disjoint held-out sessions
   for perplexity. This is the only place a generative stage can earn its keep.
3. **Evaluation.** The paper reports val accuracy 0.94–0.97 on 3 patients with random splits of
   50 %-overlapping windows and no baseline; adjacent windows share 15 s, so that split leaks.
   Here: TUSZ patient-disjoint splits, non-overlapping rows, the linear band-power baseline on
   the same windows in every run, per-session AUROC, and the patient-adaptive protocol done
   chronologically (earlier sessions → later sessions of the same patient).
4. **Context.** The paper feeds 7 500 positions of raw channels through the trunk. Here context
   is hierarchical: block embeddings (4 s) → causal transformer over N blocks → decision. 8
   blocks = 32 s, 15 = 60 s, at ~N× the single-block cost, bounded by `seq_chunk`.
5. **Multitype head** added (focal/generalized by default; 3-way or 13-way available), trained on
   ictal rows with a batch sampler that guarantees `--min-ictal-per-batch` ictal rows per batch.

## 3. Run plan

```bash
# 0. install (torch >= 2.1 for SDPA flash attention)
pip install -r requirements.txt
python -m eeg_gpt.tests.test_pipeline          # ~3 min on CPU, must print ALL TESTS PASSED

# 1. Stage 1: pretrain on every session (one pass over ~70 h of 20-ch EEG = 20k steps at batch 64)
python -m eeg_gpt.training.pretrain --checkpoint-dir ../checkpoints --out runs/pre_a \
    --block-size 1024 --d 128 --layers 4 --heads 4 --steps 100000 --batch-size 64 --amp --num-workers 8
#    variant worth one run: --pred-offsets 1,8,32   (multi-token prediction; runs/pre_b)

# 2. Stage 2: binary + type heads, single block, pretrained vs from-scratch (THE A/B)
python -m eeg_gpt.training.finetune --master-csv ../master_full.csv --checkpoint-dir ../checkpoints \
    --pretrained runs/pre_a/best.pt --out runs/ft_pre --context-blocks 1 --baseline --amp
python -m eeg_gpt.training.finetune --master-csv ../master_full.csv --checkpoint-dir ../checkpoints \
    --out runs/ft_scratch --context-blocks 1 --baseline --amp          # no --pretrained

# 3. context (the change that actually looks further): 8 blocks = 32 s, then 15
python -m eeg_gpt.training.finetune ... --pretrained runs/pre_a/best.pt --out runs/ft_ctx8 --context-blocks 8 --amp

# 4. Mahdi's own label and protocol, honestly: onset within 30 s; patient-adaptive on a patient with >= 2 sessions
python -m eeg_gpt.training.finetune ... --label horizon --horizon-s 30 --out runs/ft_horizon
python -m eeg_gpt.training.finetune ... --adapt-patient aaaaaaaa --adapt-train-frac 0.5 --out runs/adapt_aaaaaaaa

# 5. evaluate any checkpoint
python -m eeg_gpt.training.finetune ... --eval-only --load runs/ft_ctx8/best.pt --pretrained runs/pre_a/best.pt --context-blocks 8 --baseline
```

Every fine-tuning log prints, per epoch: binary AUROC, threshold-sweep balanced accuracy,
per-session AUROC (median, fraction of sessions > 0.6), AUROC after causal smoothing over
5/15/60 blocks, the band-power baseline (epoch 1), and the ictal-row type confusion matrix.

## 4. What decides the next step

- **Pretraining is working** if held-out `ce@1` keeps falling well below the unigram entropy of
  the tokens (the test corpus: 4.16 → 2.34 vs 3.25) and `mse` falls; `ppl@1` should end far
  below the 512-level uniform (512). If it plateaus early, increase data (more sessions, TUEG)
  before increasing the model.
- **Pretraining is worth keeping** only if `ft_pre` beats `ft_scratch` at the same epoch — that
  is the claim the whole design rests on, and it is a single A/B.
- **The first bar is the band-power baseline** printed in the same log (~0.57 on your data).
  Below it, look at train vs eval AUROC: train ≫ eval → shrink (`--d 64`, `--freeze-backbone`,
  dropout 0.3); train ≈ eval ≈ 0.55 → context and smoothing are the levers, not capacity.
- **Type head**: judge by the ictal-row matrix and `pred hist`; with `focal_generalized` and
  `--min-ictal-per-batch 4`, a collapsed histogram means the trunk's ictal features are not
  type-discriminative yet — fix occurrence first.
- **Patient-adaptive** numbers are a different task; report them separately from cross-patient.

## 5. Costs (A100-class, `--amp`)

| stage | per step | notes |
|---|---|---|
| pretrain, batch 64 × 1024 tokens, d 128 / 4 layers (0.9 M params) | ~0.1 s | 100k steps ≈ 3 h; IO-bound at first, keep `--num-workers 8`, `--cache-capacity 8` |
| finetune, batch 16, context 1 | ~0.1 s | 320 channel-sequences per step |
| finetune, batch 16, context 8 | ~0.6 s | 2 560 channel-sequences per step; `--freeze-backbone` makes it ~5× cheaper |

## 6. Honest expectations

Cross-patient per-4-s-window preictal AUROC on TUSZ-style data with SPH 2 min / SOP 7 min sits
around 0.6 in the literature that evaluates it honestly. The pieces here that can move that
number are (in order of evidence) temporal context and smoothing, pretraining on a large
unlabeled corpus, and patient adaptation. The design gives each of them a switch and an A/B.
Nothing generates a future at inference; the trunk is the product.
