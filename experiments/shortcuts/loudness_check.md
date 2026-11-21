# Loudness shortcut check

## Control: classifiers that see only loudness

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

## Model sensitivity: EER (%) on normal vs RMS-normalised test audio

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
