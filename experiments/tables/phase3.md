# Phase 3 results

EER in %, mean ± std over seeds (n = seeds; n = 1 rows are the shortened Phase 3 prototype runs, see the plan). All trained models use the Phase 2 protocol (silence-trimmed audio, RawBoost + gain) except E1. Degraded = mean EER over 10 codec/noise/channel conditions of the ASVspoof eval (reported) or dev (selection) subset. Speed = seconds of audio scored per second (see `experiments/efficiency/efficiency.md`).

| Model | n | dev | eval | eval min t-DCF | In-the-Wild | DEEP-VOICE (recording) | degraded eval | degraded dev | parameters | GPU × real time | CPU × real time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Phase 2 reference* |  |  |  |  |  |  |  |  |  |  |  |
| E1 CNN-BiLSTM, MFCC, untrimmed, no augmentation (v1 paper model) | 3 | 1.36 ± 0.47 | 7.97 ± 0.04 | 0.210 ± 0.014 | 47.32 ± 4.60 | 52.38 ± 6.58 | 33.62 ± 3.91 | – | 786,625 | 23,278 | 815.7 |
| E4a CNN-BiLSTM, LFCC (Phase 2 selection) | 3 | 0.34 ± 0.20 | 14.38 ± 1.09 | 0.344 ± 0.024 | 47.39 ± 11.43 | 50.30 ± 12.95 | 24.51 ± 2.26 | 18.43 ± 2.09 | 786,625 | 8,579 | 288.4 |
| *P3.1 published baselines* |  |  |  |  |  |  |  |  |  |  |  |
| E5a LCNN, LFCC | 1 | 0.24 | 15.36 | 0.318 | 53.59 | 25.00 | 26.44 | 19.52 | 269,729 | 8,917 | 521.0 |
| E5b RawNet2, raw waveform | 1 | 5.25 | 19.51 | 0.493 | 44.90 | 50.89 | 25.88 | 14.25 | 17,620,385 | 1,870 | 61.8 |
| *P3.2 CNN-RNN upgrades* |  |  |  |  |  |  |  |  |  |  |  |
| E6b E6a + attentive stats pooling (CNN-BiLSTM-Att) | 1 | 1.09 | 17.73 | 0.336 | 71.78 | 25.00 | 37.18 | 30.49 | 1,422,206 | 4,925 | 290.8 |
| *P3.3 pretrained SSL front end* |  |  |  |  |  |  |  |  |  |  |  |
| E7a WavLM-Base+ (fine-tuned) + CNN-BiLSTM-Att | 1 | 0.78 | 6.22 | 0.125 | 10.67 | 12.50 | – | 2.82 | 96,199,035 | 337 | 16.4 |
| *P3.4 score fusion (dev-trained logistic regression)* |  |  |  |  |  |  |  |  |  |  |  |
| Fusion: e4a_rawboost + e7a_wavlm_crnn | 1 | 0.12 | 7.34 | 0.140 | 17.08 | 35.71 | – | – | sum of members | – | – |

Fusion details, including every pair: `experiments/fusion/fusion.md`.

**Model for the app: E7a WavLM-Base+ (fine-tuned) + CNN-BiLSTM-Att**, chosen on dev data only: lowest mean degraded-dev EER (2.82%; clean dev 0.78%). Ranking: e7a_wavlm_crnn 2.82, e5b_rawnet2 14.25, e4a_rawboost 18.43, e5a_lcnn 19.52, e6b_crnn_seres_asp 30.49.
