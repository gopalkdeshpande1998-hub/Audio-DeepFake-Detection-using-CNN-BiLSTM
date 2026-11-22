# Reliability: e2a_lfcc_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.059·z + -0.562). Verdict: Real if p_fake ≤ 0.192, Fake if p_fake ≥ 0.772, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.06 | 33.09 | 1.4 | 98.0 | 0.6 | 0.00 | 0.00 | 0.00 |
| ASVspoof eval | 71,237 | 27.22 | 17.43 | 2.1 | 97.8 | 0.0 | 18.65 | 0.00 | 0.45 |
| In-the-Wild | 31,779 | 61.45 | 23.72 | 0.1 | 96.3 | 3.6 | 59.79 | 3.50 | 0.00 |
| DEEP-VOICE (windows) | 14,896 | 12.76 | 20.66 | 0.0 | 98.3 | 1.7 | 16.06 | 2.15 | 0.00 |

Reliability diagrams: `experiments/reliability/e2a_lfcc_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 11.72 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 4.0 | 4.0 |
| ASVspoof eval, fake | 200 | 2.0 | 4.0 | 6.0 |
| In-the-Wild, real | 200 | 1.5 | 14.0 | 15.0 |
| In-the-Wild, fake | 200 | 1.5 | 0.5 | 2.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 1.000 | 0.676 | 7.2 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 0.685 | 13.3 |
| white noise | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.005 | 0.294 | 45.7 |
| pink noise | yes | the model's internal features are far from anything seen in training | 0.000 | 0.172 | 56.5 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 1.000 | 0.600 | 9.3 |
| music-like chord | **no** | – | 1.000 | 0.753 | 5.9 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 1.000 | 0.602 | 39.9 |
| near silence | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 1.000 | 0.630 | 51.8 |
