# Reliability: vg_v2

Calibration (Platt, balanced classes) fitted on native sigmoid of the checkpoint (not refitted): p_fake = sigmoid(1.000·z + 0.000). Verdict: Real if p_fake ≤ 0.500, Fake if p_fake ≥ 0.500, else Uncertain (≤ 0% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 52.52 | 52.52 | 61.4 | 0.0 | 38.6 | 56.69 | 27.16 | 60.06 |
| ASVspoof eval | 71,237 | 36.17 | 36.17 | 42.9 | 0.0 | 57.1 | 39.50 | 33.64 | 40.17 |
| In-the-Wild | 31,779 | 2.11 | 2.11 | 63.8 | 0.0 | 36.2 | 3.18 | 1.73 | 5.62 |
| DEEP-VOICE (windows) | 14,896 | 81.37 | 81.37 | 95.8 | 0.0 | 4.2 | 83.43 | 0.59 | 95.26 |

Reliability diagrams: `experiments/reliability/vg_v2.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 5.95 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 0.0 | 0.0 |
| ASVspoof eval, fake | 200 | 2.0 | 1.5 | 3.5 |
| In-the-Wild, real | 200 | 1.5 | 26.5 | 27.0 |
| In-the-Wild, fake | 200 | 1.5 | 7.5 | 9.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.000 | 0.000 | 4.0 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.998 | 0.998 | 3.0 |
| white noise | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.000 | 0.000 | 18.4 |
| pink noise | yes | the model's internal features are far from anything seen in training | 0.000 | 0.000 | 18.5 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 0.000 | 0.000 | 4.5 |
| music-like chord | **no** | – | 0.000 | 0.000 | 4.0 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 1.000 | 1.000 | 9.6 |
| near silence | yes | it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 0.000 | 0.000 | 18.4 |
