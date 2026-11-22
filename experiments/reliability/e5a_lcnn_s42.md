# Reliability: e5a_lcnn_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.296·z + -1.162). Verdict: Real if p_fake ≤ 0.226, Fake if p_fake ≥ 0.774, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.22 | 11.18 | 10.1 | 17.6 | 72.3 | 0.02 | 0.00 | 0.02 |
| ASVspoof eval | 71,237 | 23.88 | 26.22 | 38.1 | 30.5 | 31.4 | 40.07 | 0.00 | 31.06 |
| In-the-Wild | 31,779 | 50.91 | 15.38 | 8.4 | 86.4 | 5.3 | 51.19 | 5.03 | 10.29 |
| DEEP-VOICE (windows) | 14,896 | 9.29 | 10.14 | 1.6 | 56.7 | 41.7 | 6.06 | 16.17 | 0.69 |

Reliability diagrams: `experiments/reliability/e5a_lcnn_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 96-d penultimate layer, threshold 7.28 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 5.5 | 5.5 |
| ASVspoof eval, fake | 200 | 2.0 | 7.0 | 8.5 |
| In-the-Wild, real | 200 | 1.5 | 17.0 | 18.0 |
| In-the-Wild, fake | 200 | 1.5 | 15.5 | 16.5 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 0.986 | 29.0 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 0.992 | 26.6 |
| white noise | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.141 | 0.155 | 9.5 |
| pink noise | yes | the model's internal features are far from anything seen in training | 0.242 | 0.182 | 11.0 |
| 50 Hz hum | yes | it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 0.917 | 0.389 | 24.7 |
| music-like chord | yes | the model's internal features are far from anything seen in training | 0.987 | 0.528 | 19.4 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.916 | 0.388 | 13.9 |
| near silence | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.193 | 0.170 | 11.2 |
