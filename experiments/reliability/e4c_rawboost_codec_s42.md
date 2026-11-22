# Reliability: e4c_rawboost_codec_s42

Calibration (Platt, balanced classes) fitted on clean dev + 10 degraded dev conditions: p_fake = sigmoid(0.659·z + -0.751). Verdict: Real if p_fake ≤ 0.188, Fake if p_fake ≥ 0.597, else Uncertain (≤ 5% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 0.51 | 2.78 | 9.8 | 1.9 | 88.3 | 0.42 | 0.71 | 0.38 |
| ASVspoof eval | 71,237 | 15.16 | 18.02 | 26.7 | 16.5 | 56.8 | 20.40 | 0.60 | 18.93 |
| In-the-Wild | 31,779 | 60.53 | 57.51 | 1.4 | 5.3 | 93.2 | 63.91 | 95.20 | 1.91 |
| DEEP-VOICE (windows) | 14,896 | 10.23 | 8.94 | 0.1 | 5.6 | 94.3 | 10.83 | 81.63 | 0.02 |

Reliability diagrams: `experiments/reliability/e4c_rawboost_codec_s42.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 10.64 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 4.5 | 4.5 |
| ASVspoof eval, fake | 200 | 2.0 | 1.0 | 3.0 |
| In-the-Wild, real | 200 | 1.5 | 2.0 | 3.0 |
| In-the-Wild, fake | 200 | 1.5 | 1.5 | 3.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 1.000 | 24.7 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 1.000 | 17.5 |
| white noise | yes | it sounds like noise rather than speech | 0.836 | 0.580 | 10.4 |
| pink noise | yes | the model's internal features are far from anything seen in training | 0.959 | 0.790 | 19.6 |
| 50 Hz hum | yes | it lacks the rhythm of speech; the model's internal features are far from anything seen in training | 1.000 | 1.000 | 21.5 |
| music-like chord | yes | the model's internal features are far from anything seen in training | 0.947 | 0.760 | 10.9 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 1.000 | 0.992 | 17.1 |
| near silence | yes | it sounds like noise rather than speech | 0.833 | 0.576 | 10.3 |
