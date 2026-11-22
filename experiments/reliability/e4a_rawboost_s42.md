# Reliability: e4a_rawboost_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.181·z + -0.316). Verdict: Real if p_fake ≤ 0.109, Fake if p_fake ≥ 0.736, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.50 | 22.59 | 0.0 | 45.6 | 54.4 | 0.04 | 0.00 | 0.02 |
| ASVspoof eval | 71,237 | 22.41 | 14.67 | 0.0 | 76.9 | 23.1 | 0.03 | 0.00 | 0.01 |
| In-the-Wild | 31,779 | 57.69 | 31.74 | 1.0 | 47.8 | 51.1 | 49.98 | 41.49 | 0.02 |
| DEEP-VOICE (windows) | 14,896 | 12.79 | 14.71 | 2.4 | 33.8 | 63.7 | 12.82 | 52.04 | 2.26 |

Reliability diagrams: `experiments/reliability/e4a_rawboost_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 11.33 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 4.0 | 4.0 |
| ASVspoof eval, fake | 200 | 2.0 | 0.5 | 2.5 |
| In-the-Wild, real | 200 | 1.5 | 6.0 | 7.0 |
| In-the-Wild, fake | 200 | 1.5 | 6.5 | 7.5 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 0.931 | 13.2 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 0.910 | 19.0 |
| white noise | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.000 | 0.066 | 63.0 |
| pink noise | yes | the model's internal features are far from anything seen in training | 0.000 | 0.071 | 58.5 |
| 50 Hz hum | yes | it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 0.000 | 0.019 | 41.8 |
| music-like chord | yes | the model's internal features are far from anything seen in training | 1.000 | 0.752 | 13.2 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.000 | 0.040 | 40.9 |
| near silence | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.000 | 0.071 | 59.5 |
