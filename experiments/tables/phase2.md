# Phase 2 results

EER in %. Mean ± std when several seeds exist. Every choice (best feature, best augmentation) is made on **ASVspoof dev EER only**; the test columns are reported, never used for selection.

## P2.1 Input-feature ablation (same CNN-BiLSTM, untrimmed audio)

| Input / setting | seeds | dev EER | eval EER | eval min t-DCF | In-the-Wild EER | DEEP-VOICE EER (window / recording) |
|---|---|---|---|---|---|---|
| MFCC-40 (E1, paper feature) | 3 | 1.36 ± 0.47 | 7.97 ± 0.04 | 0.210 ± 0.014 | 47.32 ± 4.60 | 48.84 ± 1.32 / 52.38 ± 6.58 |
| LFCC-60 | 3 | 0.04 ± 0.03 | 6.64 ± 1.52 | 0.155 ± 0.028 | 42.51 ± 6.63 | 46.43 ± 2.63 / 45.54 ± 7.09 |
| CQCC-60 | 3 | 2.29 ± 0.22 | 11.00 ± 0.32 | 0.234 ± 0.008 | 26.87 ± 2.82 | 43.66 ± 2.40 / 30.06 ± 6.58 |
| Log-mel-80 | 3 | 0.10 ± 0.05 | 6.48 ± 0.61 | 0.112 ± 0.005 | 64.79 ± 8.06 | 49.28 ± 1.43 / 50.00 ± 0.00 |
| Sinc filterbank (raw waveform) | 3 | 0.08 ± 0.00 | 6.94 ± 0.20 | 0.129 ± 0.020 | 55.35 ± 4.83 | 43.05 ± 1.49 / 38.10 ± 1.03 |

## P2.2 Silence shortcut

### Control: classifiers that see only silence and duration

| Input | asvspoof2019_la:dev | asvspoof2019_la:eval | in_the_wild:test |
|---|---|---|---|
| lead_s only (longer = more likely real) | 32.57 | 15.93 | 68.04 |
| trail_s only (longer = more likely fake) | 51.22 | 61.65 | 32.57 |
| total_s only (longer = more likely real) | 48.16 | 41.96 | 58.41 |
| speech_s only (longer = more likely fake) | 44.95 | 43.86 | 42.61 |
| all 4, logistic regression | 30.77 | 20.05 | 45.44 |
| all 4, gradient boosting | 23.16 | 23.94 | 44.37 |

### Models scored on silence-trimmed test audio

| Model | trained on | scored on | seeds | dev EER | eval EER | eval min t-DCF | In-the-Wild EER |
|---|---|---|---|---|---|---|---|
| E1 MFCC | untrimmed | untrimmed | 3 | 1.36 ± 0.47 | 7.97 ± 0.04 | 0.210 ± 0.014 | 47.32 ± 4.60 |
| E1 MFCC | untrimmed | trimmed | 3 | 16.76 ± 1.26 | 27.68 ± 0.65 | 0.790 ± 0.003 | 50.57 ± 4.76 |
| E2a LFCC | untrimmed | untrimmed | 3 | 0.04 ± 0.03 | 6.64 ± 1.52 | 0.155 ± 0.028 | 42.51 ± 6.63 |
| E2a LFCC | untrimmed | trimmed | 3 | 4.11 ± 2.28 | 22.05 ± 0.29 | 0.510 ± 0.033 | 46.08 ± 7.42 |
| E3 LFCC | trimmed | untrimmed | 3 | 0.35 ± 0.28 | 20.75 ± 2.22 | 0.356 ± 0.018 | 57.23 ± 5.26 |
| E3 LFCC | trimmed | trimmed | 3 | 0.13 ± 0.06 | 21.47 ± 0.95 | 0.355 ± 0.007 | 57.24 ± 3.85 |

## P2.3 Augmentation ablation (LFCC-60, silence-trimmed)

| Input / setting | seeds | dev EER | eval EER | eval min t-DCF | In-the-Wild EER | DEEP-VOICE EER (window / recording) |
|---|---|---|---|---|---|---|
| LFCC-60, trimmed, no augmentation (E3) | 3 | 0.13 ± 0.06 | 21.47 ± 0.95 | 0.355 ± 0.007 | 57.24 ± 3.85 | 38.14 ± 2.36 / 41.67 ± 7.22 |
| + RawBoost | 3 | 0.34 ± 0.20 | 14.38 ± 1.09 | 0.344 ± 0.024 | 47.39 ± 11.43 | 43.00 ± 6.56 / 50.30 ± 12.95 |
| + Codec | 3 | 0.64 ± 0.43 | 14.34 ± 2.10 | 0.353 ± 0.034 | 58.96 ± 10.13 | 33.87 ± 4.88 / 28.27 ± 7.27 |
| + RawBoost + codec | 3 | 1.90 ± 0.65 | 12.81 ± 0.56 | 0.310 ± 0.001 | 53.45 ± 9.44 | 37.76 ± 4.30 / 29.17 ± 7.22 |
| + RawBoost + codec + resample + speed | 3 | 10.95 ± 1.80 | 17.96 ± 0.29 | 0.459 ± 0.013 | 36.41 ± 3.77 | 47.02 ± 4.46 / 45.83 ± 7.22 |

## P2.3 Augmentation selection: EER (%) on a degraded ASVspoof **dev** subset

Subset: 2,000 real + 770 per attack (A01–A06) from dev. Selection metric = mean over the 10 conditions (averaged over seeds); this is the only table used to choose the augmentation.

| Model | seeds | clean | mp3_51k | mp3_19k | opus_84k | opus_10k | telephone | white_20db | white_10db | white_5db | pink_10db | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2a LFCC, untrimmed | 3 | 0.05 | 19.71 | 32.77 | 0.12 | 14.22 | 25.02 | 39.92 | 51.01 | 49.17 | 37.29 | **26.93** ± 0.93 |
| E3 LFCC, trimmed | 3 | 0.15 | 31.90 | 33.31 | 0.28 | 32.44 | 36.55 | 29.63 | 45.10 | 50.35 | 36.68 | **29.64** ± 5.39 |
| E4a + RawBoost | 3 | 0.31 | 20.02 | 23.97 | 0.33 | 21.50 | 24.90 | 12.17 | 25.73 | 34.11 | 21.22 | **18.43** ± 2.09 |
| E4b + codec | 3 | 0.58 | 13.49 | 17.97 | 0.67 | 18.35 | 16.08 | 47.83 | 53.05 | 58.47 | 43.21 | **26.97** ± 1.33 |
| E4c + RawBoost + codec | 3 | 1.72 | 13.75 | 17.49 | 1.83 | 18.11 | 15.44 | 23.90 | 35.39 | 38.10 | 32.93 | **19.86** ± 2.27 |
| E4d + RawBoost + codec + resample + speed | 3 | 10.63 | 12.14 | 15.43 | 10.66 | 16.37 | 14.20 | 26.71 | 30.89 | 36.91 | 29.53 | **20.35** ± 0.33 |

**Selected augmentation: E4a + RawBoost** (lowest mean degraded-dev EER, 18.43%, over 3 seeds).

## P2.4 Robustness: EER (%) on a degraded ASVspoof eval subset

Subset: 2,000 real + 770 per attack (A07–A19) from eval, identical audio for every model. Reported only.

| Model | seeds | clean | mp3_51k | mp3_19k | opus_84k | opus_10k | telephone | white_20db | white_10db | white_5db | pink_10db | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 MFCC, untrimmed | 3 | 8.20 | 11.43 | 12.05 | 10.46 | 18.72 | 69.12 | 39.60 | 50.00 | 59.80 | 56.85 | **33.62** ± 3.91 |
| E2a LFCC, untrimmed | 3 | 6.65 | 21.74 | 45.20 | 7.58 | 17.43 | 27.02 | 52.67 | 56.88 | 56.41 | 32.51 | **32.41** ± 0.90 |
| E3 LFCC, trimmed | 3 | 21.76 | 35.15 | 36.65 | 22.08 | 36.22 | 38.40 | 28.08 | 38.33 | 51.64 | 32.80 | **34.11** ± 5.11 |
| E4a + RawBoost | 3 | 14.26 | 26.83 | 30.93 | 14.33 | 27.40 | 28.47 | 18.79 | 26.96 | 35.03 | 22.10 | **24.51** ± 2.26 |
| E4b + codec | 3 | 14.33 | 22.43 | 27.18 | 14.80 | 25.52 | 24.01 | 46.12 | 57.88 | 63.07 | 42.30 | **33.76** ± 3.16 |
| E4c + RawBoost + codec | 3 | 12.48 | 19.61 | 23.88 | 12.53 | 22.96 | 19.95 | 24.28 | 36.73 | 39.11 | 31.77 | **24.33** ± 3.11 |
| E4d + RawBoost + codec + resample + speed | 3 | 17.67 | 18.45 | 22.27 | 17.75 | 22.13 | 21.27 | 30.22 | 31.80 | 37.68 | 28.93 | **24.82** ± 2.30 |

Conditions: `clean` = no degradation (reference); `mp3_51k` = MP3, about 51 kbps; `mp3_19k` = MP3, about 19 kbps; `opus_84k` = Opus, about 84 kbps; `opus_10k` = Opus, about 10 kbps; `telephone` = 300-3400 Hz, 8 kHz mu-law; `white_20db` = white noise, SNR 20 dB; `white_10db` = white noise, SNR 10 dB; `white_5db` = white noise, SNR 5 dB; `pink_10db` = pink noise, SNR 10 dB

Figures: `experiments/tables/robustness_eval.png`, `experiments/tables/robustness_dev.png` (mean over seeds, bars = ± 1 std).

## P2.2 Length shortcut (from `experiments/shortcuts/length_check.md`)

EER (%) by duration of the audio the model saw (mean over seeds; n = clips real / fake). ρ = Spearman correlation of score with duration inside each class (|ρ| < 0.1 = negligible).

### E1 MFCC, untrimmed (3 seeds, whole-file duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 6.04 (n=265/16,366) | 9.43 (n=5,360/33,401) | 8.39 (n=1,554/10,721) | 8.74 (n=176/3,394) | 7.97 | +0.00 | +0.13 |
| in_the_wild:test | 47.04 (n=5,134/1,950) | 44.67 (n=8,409/3,892) | 49.95 (n=3,686/2,429) | 52.58 (n=2,734/3,545) | 47.32 | -0.11 | +0.03 |

### E2a LFCC, untrimmed (3 seeds, whole-file duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 3.17 (n=265/16,366) | 7.91 (n=5,360/33,401) | 8.23 (n=1,554/10,721) | 7.91 (n=176/3,394) | 6.64 | -0.09 | +0.05 |
| in_the_wild:test | 43.81 (n=5,134/1,950) | 41.25 (n=8,409/3,892) | 44.98 (n=3,686/2,429) | 48.16 (n=2,734/3,545) | 42.50 | -0.14 | -0.00 |

### E3 LFCC, trimmed (3 seeds, speech-only duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 22.49 (n=3,710/25,063) | 21.32 (n=3,406/28,940) | 19.14 (n=221/7,428) | – (n=18/2,451) | 21.47 | -0.03 | -0.02 |
| in_the_wild:test | 57.22 (n=5,901/2,606) | 58.90 (n=8,185/3,706) | 56.91 (n=3,330/2,230) | 56.13 (n=2,547/3,274) | 57.24 | -0.03 | -0.04 |

### E4c LFCC, trimmed, RawBoost + codec (3 seeds, speech-only duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 15.12 (n=3,710/25,063) | 11.85 (n=3,406/28,940) | 9.32 (n=221/7,428) | – (n=18/2,451) | 12.81 | +0.17 | -0.10 |
| in_the_wild:test | 54.58 (n=5,901/2,606) | 58.06 (n=8,185/3,706) | 55.26 (n=3,330/2,230) | 52.23 (n=2,547/3,274) | 53.45 | -0.27 | -0.25 |

### E4d LFCC, trimmed, full augmentation (3 seeds, speech-only duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 20.84 (n=3,710/25,063) | 17.02 (n=3,406/28,940) | 14.39 (n=221/7,428) | – (n=18/2,451) | 17.96 | +0.12 | -0.13 |
| in_the_wild:test | 36.47 (n=5,901/2,606) | 39.66 (n=8,185/3,706) | 36.37 (n=3,330/2,230) | 38.15 (n=2,547/3,274) | 36.41 | -0.33 | -0.24 |

Bins with fewer than 20 clips of either class have no EER (–).

## P2.2 Loudness shortcut (from `experiments/shortcuts/loudness_check.md`)

### Control: classifiers that see only loudness

EER (%). Speech RMS level (silence excluded), peak level, crest factor (peak − RMS). Trained on ASVspoof 2019 LA train. 50% = no information.

| Input | asvspoof2019_la:dev | asvspoof2019_la:eval | in_the_wild:test |
|---|---|---|---|
| rms_db only (higher = more likely real) | 46.43 | 54.93 | 40.32 |
| peak_db only (higher = more likely real) | 66.25 | 66.46 | 47.00 |
| crest_db only (higher = more likely fake) | 46.43 | 54.92 | 39.27 |
| all 3, logistic regression | 46.46 | 54.93 | 52.87 |
| all 3, gradient boosting | 46.55 | 49.49 | 60.51 |

Median levels (dBFS; real / fake):

| Set | speech RMS | peak | crest |
|---|---|---|---|
| asvspoof2019_la:train | -17.0 / -17.2 | -0.0 / -0.0 | 17.0 / 17.2 |
| asvspoof2019_la:dev | -16.6 / -17.0 | -0.0 / -0.0 | 16.6 / 17.0 |
| asvspoof2019_la:eval | -17.4 / -16.8 | -0.0 / -0.0 | 17.3 / 16.8 |
| in_the_wild:test | -25.7 / -28.2 | -9.3 / -10.1 | 16.4 / 17.8 |

### Model sensitivity: EER (%) on normal vs RMS-normalised test audio

Every test crop scaled to the same RMS level (−26 dBFS, gain limited to ±20 dB). Mean ± std over seeds.

| Model | set | normal | RMS-normalised | change |
|---|---|---|---|---|
| E1 MFCC, untrimmed | asvspoof2019_la:dev | 1.36 ± 0.47 (n=3) | 1.44 ± 0.54 (n=3) | +0.08 |
| E1 MFCC, untrimmed | asvspoof2019_la:eval | 7.97 ± 0.04 (n=3) | 7.92 ± 0.09 (n=3) | -0.05 |
| E1 MFCC, untrimmed | in_the_wild:test | 47.32 ± 4.60 (n=3) | 46.59 ± 4.90 (n=3) | -0.74 |
| E2a LFCC, untrimmed | asvspoof2019_la:dev | 0.04 ± 0.03 (n=3) | 0.15 ± 0.08 (n=3) | +0.11 |
| E2a LFCC, untrimmed | asvspoof2019_la:eval | 6.64 ± 1.52 (n=3) | 7.43 ± 1.23 (n=3) | +0.78 |
| E2a LFCC, untrimmed | in_the_wild:test | 42.51 ± 6.63 (n=3) | 42.87 ± 5.76 (n=3) | +0.36 |
| E4c LFCC, trimmed, RawBoost + codec | asvspoof2019_la:dev | 1.90 ± 0.65 (n=3) | 2.78 ± 0.75 (n=3) | +0.89 |
| E4c LFCC, trimmed, RawBoost + codec | asvspoof2019_la:eval | 12.81 ± 0.56 (n=3) | 13.41 ± 0.40 (n=3) | +0.60 |
| E4c LFCC, trimmed, RawBoost + codec | in_the_wild:test | 53.45 ± 9.44 (n=3) | 52.10 ± 10.39 (n=3) | -1.35 |
| E4d LFCC, trimmed, full augmentation | asvspoof2019_la:dev | 10.95 ± 1.80 (n=3) | 12.04 ± 2.20 (n=3) | +1.09 |
| E4d LFCC, trimmed, full augmentation | asvspoof2019_la:eval | 17.96 ± 0.29 (n=3) | 18.39 ± 0.38 (n=3) | +0.43 |
| E4d LFCC, trimmed, full augmentation | in_the_wild:test | 36.41 ± 3.77 (n=3) | 36.24 ± 3.89 (n=3) | -0.17 |
