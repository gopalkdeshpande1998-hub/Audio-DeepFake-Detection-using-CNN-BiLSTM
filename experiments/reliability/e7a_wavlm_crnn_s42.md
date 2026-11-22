# Reliability: e7a_wavlm_crnn_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.706·z + -0.001). Verdict: Real if p_fake ≤ 0.415, Fake if p_fake ≥ 0.415, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.72 | 0.87 | 11.3 | 0.0 | 88.7 | 1.08 | 0.16 | 1.19 |
| ASVspoof eval | 71,237 | 13.96 | 12.22 | 26.0 | 0.0 | 74.0 | 15.74 | 0.11 | 17.54 |
| In-the-Wild | 31,779 | 7.33 | 5.02 | 64.2 | 0.0 | 35.8 | 10.99 | 7.63 | 16.66 |
| DEEP-VOICE (windows) | 14,896 | 56.88 | 53.10 | 74.0 | 0.0 | 26.0 | 62.74 | 4.99 | 70.99 |

Reliability diagrams: `experiments/reliability/e7a_wavlm_crnn_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 6.64 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 0.0 | 0.0 |
| ASVspoof eval, fake | 200 | 2.0 | 10.0 | 11.5 |
| In-the-Wild, real | 200 | 1.5 | 9.0 | 9.5 |
| In-the-Wild, fake | 200 | 1.5 | 25.5 | 27.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.003 | 0.018 | 3.3 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.005 | 0.025 | 4.1 |
| white noise | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.068 | 0.136 | 8.1 |
| pink noise | **no** | – | 0.013 | 0.045 | 5.5 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 0.006 | 0.026 | 4.2 |
| music-like chord | **no** | – | 0.019 | 0.058 | 6.3 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.942 | 0.878 | 8.6 |
| near silence | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.048 | 0.108 | 7.7 |
