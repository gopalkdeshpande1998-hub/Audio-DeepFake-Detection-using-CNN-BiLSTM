# Reliability: vg_improved

Calibration (Platt, balanced classes) fitted on native sigmoid of the checkpoint (not refitted): p_fake = sigmoid(1.000·z + 0.000). Verdict: Real if p_fake ≤ 0.500, Fake if p_fake ≥ 0.500, else Uncertain (≤ 0% of each dev class on the wrong side).

| Set | n | ECE raw (%) | ECE calibrated (%) | Real % | Uncertain % | Fake % | error among decided % | real → Fake % | fake → Real % |
|---|---|---|---|---|---|---|---|---|---|
| ASVspoof dev | 24,844 | 39.93 | 39.93 | 48.8 | 0.0 | 51.2 | 49.71 | 54.55 | 49.15 |
| ASVspoof eval | 71,237 | 26.61 | 26.61 | 35.5 | 0.0 | 64.5 | 37.97 | 62.13 | 35.19 |
| In-the-Wild | 31,779 | 21.69 | 21.69 | 60.0 | 0.0 | 40.0 | 34.88 | 30.01 | 43.13 |
| DEEP-VOICE (windows) | 14,896 | 81.28 | 81.28 | 95.6 | 0.0 | 4.4 | 83.16 | 0.38 | 94.98 |

Reliability diagrams: `experiments/reliability/vg_improved.png`.

## Out-of-distribution check

Embedding check: Mahalanobis distance on the 128-d penultimate layer, threshold 6.85 (99th percentile of 1000 dev clips). Signal checks: ranges from bona fide dev speech (`experiments/reliability/signal_ranges.json`).

Flag rate on test speech, real and fake (false alarms: should be low; fake speech is still speech):

| Set | n | signal check % | embedding check % | either % |
|---|---|---|---|---|
| ASVspoof eval, real | 200 | 0.0 | 0.0 | 0.0 |
| ASVspoof eval, fake | 200 | 2.0 | 0.5 | 2.5 |
| In-the-Wild, real | 200 | 1.5 | 0.5 | 2.0 |
| In-the-Wild, fake | 200 | 1.5 | 0.5 | 2.0 |

Synthetic non-speech (should be flagged):

| Input | flagged | reasons | p_fake raw | p_fake calibrated | embedding distance |
|---|---|---|---|---|---|
| tone 440 Hz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.930 | 0.930 | 2.0 |
| tone 1 kHz | yes | it sounds like a steady tone or hum; it lacks the rhythm of speech | 0.930 | 0.930 | 6.0 |
| white noise | yes | it sounds like noise rather than speech | 0.755 | 0.755 | 2.5 |
| pink noise | **no** | – | 0.560 | 0.560 | 2.8 |
| 50 Hz hum | yes | it lacks the rhythm of speech | 0.156 | 0.156 | 3.5 |
| music-like chord | **no** | – | 0.765 | 0.765 | 2.1 |
| click train | yes | most of the audio is silence; it sounds like noise rather than speech; the model's internal features are far from anything seen in training | 1.000 | 1.000 | 19.4 |
| near silence | yes | it sounds like noise rather than speech | 0.746 | 0.746 | 2.5 |
