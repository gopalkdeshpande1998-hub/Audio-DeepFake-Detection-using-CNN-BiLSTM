# Reliability: e6b_crnn_seres_asp_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.292·z + -1.416). Verdict: Real if p_fake ≤ 0.290, Fake if p_fake ≥ 0.707, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.43 | 23.14 | 11.8 | 50.3 | 37.9 | 3.18 | 0.00 | 1.76 |
| ASVspoof eval | 71,237 | 23.26 | 40.22 | 41.5 | 38.6 | 19.9 | 50.88 | 0.00 | 34.86 |
| In-the-Wild | 31,779 | 61.93 | 35.36 | 2.1 | 33.9 | 64.0 | 77.65 | 79.94 | 2.88 |
| DEEP-VOICE (windows) | 14,896 | 10.93 | 18.22 | 1.7 | 2.0 | 96.3 | 9.93 | 77.87 | 0.00 |

Reliability diagrams: `experiments/reliability/e6b_crnn_seres_asp_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 5.31 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 8.0 | 8.0 |
| ASVspoof eval, fake | 200 | 2.0 | 1.5 | 3.5 |
| In-the-Wild, real | 200 | 1.5 | 0.0 | 1.5 |
| In-the-Wild, fake | 200 | 1.5 | 1.0 | 2.5 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.998 | 0.586 | 1.8 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 1.000 | 0.706 | 0.6 |
| white noise | yes | it sounds like noise rather than speech | 0.994 | 0.514 | 2.8 |
| pink noise | **no** | – | 0.999 | 0.683 | 1.1 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 0.098 | 0.113 | 5.2 |
| music-like chord | **no** | – | 1.000 | 0.705 | 0.6 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech | 0.052 | 0.094 | 4.4 |
| near silence | yes | it sounds like noise rather than speech | 0.999 | 0.645 | 1.8 |
