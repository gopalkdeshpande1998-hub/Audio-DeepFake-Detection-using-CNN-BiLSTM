# Reliability: e1_cnn_bilstm_mfcc_s42

Calibration (Platt, balanced classes) fitted on clean dev: p_fake = sigmoid(0.531·z + -1.623). Verdict: Real if p_fake ≤ 0.526, Fake if p_fake ≥ 0.526, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.23 | 0.27 | 10.9 | 0.0 | 89.1 | 0.87 | 0.86 | 0.87 |
| ASVspoof eval | 71,237 | 16.76 | 22.20 | 41.8 | 0.0 | 58.2 | 31.55 | 0.41 | 35.13 |
| In-the-Wild | 31,779 | 48.41 | 39.10 | 29.9 | 0.0 | 70.2 | 47.82 | 64.30 | 19.98 |
| DEEP-VOICE (windows) | 14,896 | 20.27 | 24.93 | 30.9 | 0.0 | 69.1 | 34.57 | 64.77 | 30.25 |

Reliability diagrams: `experiments/reliability/e1_cnn_bilstm_mfcc_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 10.38 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 1.0 | 1.0 |
| ASVspoof eval, fake | 200 | 2.0 | 7.5 | 9.5 |
| In-the-Wild, real | 200 | 1.5 | 11.0 | 12.0 |
| In-the-Wild, fake | 200 | 1.5 | 3.5 | 5.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 1.000 | 0.931 | 10.0 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 0.244 | 0.098 | 12.0 |
| white noise | yes | it sounds like noise rather than speech | 0.454 | 0.152 | 9.8 |
| pink noise | **no** | – | 0.320 | 0.117 | 10.2 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 0.069 | 0.047 | 10.3 |
| music-like chord | **no** | – | 1.000 | 0.999 | 9.2 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech | 0.010 | 0.017 | 9.2 |
| near silence | yes | it sounds like noise rather than speech | 0.199 | 0.086 | 8.3 |
