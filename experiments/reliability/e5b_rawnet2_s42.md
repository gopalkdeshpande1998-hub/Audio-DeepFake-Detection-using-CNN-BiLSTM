# Reliability: e5b_rawnet2_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.411·z + -0.307). Verdict: Real if p_fake ≤ 0.133, Fake if p_fake ≥ 0.723, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 1.58 | 4.20 | 7.5 | 9.2 | 83.3 | 0.81 | 3.41 | 0.43 |
| ASVspoof eval | 71,237 | 23.09 | 16.58 | 19.7 | 32.7 | 47.6 | 18.65 | 2.04 | 13.76 |
| In-the-Wild | 31,779 | 33.50 | 20.18 | 12.7 | 58.5 | 28.8 | 48.85 | 27.38 | 8.29 |
| DEEP-VOICE (windows) | 14,896 | 41.30 | 34.50 | 16.5 | 62.1 | 21.4 | 49.50 | 28.41 | 17.37 |

Reliability diagrams: `experiments/reliability/e5b_rawnet2_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 1024-d penultimate layer, threshold 11.09 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 5.5 | 5.5 |
| ASVspoof eval, fake | 200 | 2.0 | 0.5 | 2.5 |
| In-the-Wild, real | 200 | 1.5 | 2.0 | 3.5 |
| In-the-Wild, fake | 200 | 1.5 | 0.5 | 2.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 0.982 | 0.793 | 20.1 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 0.001 | 0.034 | 11.3 |
| white noise | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.055 | 0.186 | 11.2 |
| pink noise | **no** | – | 0.969 | 0.751 | 10.1 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 0.999 | 0.930 | 9.1 |
| music-like chord | yes | the model's internal features are far from anything seen in training | 0.005 | 0.078 | 13.6 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech | 0.317 | 0.349 | 7.3 |
| near silence | yes | it sounds like noise rather than speech | 0.057 | 0.188 | 10.5 |
