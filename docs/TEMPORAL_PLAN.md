# Segment-level (temporal) real/fake detection: audit and plan

Started 2026-11-24. Goal: label each part of a recording REAL or FAKE from a model trained with
time-stamped labels, and keep the whole-clip verdict as a separate output.

## 1. Audit: why the current model only judges whole clips

| Part | What it does now | Consequence |
|---|---|---|
| Front end `features/ssl.py` | WavLM-Base+, learned weighted sum of 13 layers, 128-d projection, 50 frames/s | time-resolved |
| CNN `models/crnn.py` | 3 SE-residual blocks, each `MaxPool2d(2)` | time /8: 6.25 frames/s, one frame every 0.16 s |
| BiLSTM | 2 layers, 256-d output per frame `h (B, T', 256)` | still time-resolved |
| Attentive statistics pooling | weighted mean and std of `h` over **all** frames | **time is collapsed here** |
| Head | MLP -> one spoof logit | one number per input |
| Training `train.py` | 4-s random crops, one clip label (`BCEWithLogitsLoss`), ASVspoof 2019 LA train (fully real or fully fake utterances) | the model was never told *where* a fake starts or ends |
| Inference `inference.py` / `single.py` | file = mean logit of 4-s windows; timeline = the same clip score on sliding 4-s windows every 0.5 s | the current timeline is a whole-clip score moved along the file: 4-s resolution, smears short fakes |
| Confidence | calibrated P(fake) of the clip (Platt fit on ASVspoof dev) | no per-segment confidence |

The limitation is the attentive pooling + clip-only labels. Everything before the pooling already has a 0.16-s
time axis, which matches PartialSpoof's 0.16-s segment labels exactly.

## 2. Design

- **Model (`CRNN(frame_head=True)`)**: keep WavLM + CNN + BiLSTM + attentive pooling + clip head (E7a weights,
  warm start). Add a frame head (`Linear 256->64, ReLU, Dropout, Linear 64->1`) on the BiLSTM output, one logit
  per 0.16 s. The clip head stays the separate whole-file verdict.
- **Loss**: frame BCE (class-weighted, label-masked) + clip BCE (multi-task, as in Zhang et al. 2021).
- **Data**: PartialSpoof v1.2 (English, ASVspoof 2019 sources) + ASVspoof 2019 LA train (fully real / fully fake).
  - train / validation: PartialSpoof **dev**, split by speaker (validation speakers never seen in training) +
    ASVspoof train for fully real / fully fake clips;
  - test: PartialSpoof **eval** only (unseen speakers, unseen attacks A07-A19), never used for any choice.
  - The PartialSpoof train set was not downloaded (6 GB download cap); it uses the same attacks as dev.
- **Training crops**: 4 s, start aligned to the 0.16-s grid so labels stay exact; short clips are repeated
  together with their labels.
- **Inference (any length)**: 4-s windows, hop 2 s (aligned to 0.16 s), overlapping frame logits averaged;
  calibration fitted on validation frames, then each frame takes its more probable class (FAKE only if
  P(fake) > P(real), ties REAL; updated 2026-11-24: the first version's uncertain band and no-speech label were
  removed, so every speech frame is REAL or FAKE; silent frames are NO SPEECH, drawn grey); consecutive frames merged into regions with
  no smoothing beyond the 0.16-s frame (short fakes are kept).
- **Evaluation** on PartialSpoof eval, new model vs E7a's sliding-window timeline: clip accuracy, frame EER /
  precision / recall / F1 / FPR / FNR / confusion matrix, segment precision / recall / F1 (IoU >= 0.5), boundary
  error, recall of short fake regions (< 0.5 s, 0.5-1 s, 1-2 s, > 2 s), threshold sweep.

## 3. Phases

1. Audit and plan (this file).
2. Data: manifests for PartialSpoof dev/eval with 0.16-s labels and vad; speaker split.
3. Model + temporal training loop + tests.
4. Smoke run, then full training (warm start from E7a).
5. Calibration and thresholds on validation; evaluation on eval; honest comparison with E7a.
6. Temporal inference in `adfd.single` (frame track + regions), API field, UI uses the real frame predictions.
7. Mixed-clip tests, report, limitations.

## 4. Status (2026-11-24)

All phases done for a first version: model `runs/t1_wavlm_crnn_temporal_s42` (2 epochs), calibration fitted on the
validation speakers, evaluation on PartialSpoof eval and on 10-15 s mixes, app + API integration. Results and
limitations: `experiments/temporal_t1/REPORT.md`. Reproduce:

    python scripts/data/build_partialspoof.py
    python -m adfd.train_temporal --config configs/experiments/t1_wavlm_crnn_temporal.yaml --run-name t1_wavlm_crnn_temporal_s42
    python scripts/evaluation/eval_temporal.py --run runs/t1_wavlm_crnn_temporal_s42 --fit
    python scripts/evaluation/eval_temporal.py --run runs/t1_wavlm_crnn_temporal_s42 --split test --limit 6000
    python scripts/evaluation/eval_temporal.py --run runs/t1_wavlm_crnn_temporal_s42 --split test --long 300

Next: more seeds / epochs, tests on DEEP-VOICE and degraded audio, a full (71k) eval pass.
