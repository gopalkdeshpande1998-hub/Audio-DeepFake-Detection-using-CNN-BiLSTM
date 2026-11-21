# Length shortcut check

EER (%) by duration of the audio the model saw (mean over seeds; n = clips real / fake). ρ = Spearman correlation of score with duration inside each class (|ρ| < 0.1 = negligible).

## E1 MFCC, untrimmed (3 seeds, whole-file duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 6.04 (n=265/16,366) | 9.43 (n=5,360/33,401) | 8.39 (n=1,554/10,721) | 8.74 (n=176/3,394) | 7.97 | +0.00 | +0.13 |
| in_the_wild:test | 47.04 (n=5,134/1,950) | 44.67 (n=8,409/3,892) | 49.95 (n=3,686/2,429) | 52.58 (n=2,734/3,545) | 47.32 | -0.11 | +0.03 |

## E2a LFCC, untrimmed (3 seeds, whole-file duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 3.17 (n=265/16,366) | 7.91 (n=5,360/33,401) | 8.23 (n=1,554/10,721) | 7.91 (n=176/3,394) | 6.64 | -0.09 | +0.05 |
| in_the_wild:test | 43.81 (n=5,134/1,950) | 41.25 (n=8,409/3,892) | 44.98 (n=3,686/2,429) | 48.16 (n=2,734/3,545) | 42.50 | -0.14 | -0.00 |

## E3 LFCC, trimmed (3 seeds, speech-only duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 22.49 (n=3,710/25,063) | 21.32 (n=3,406/28,940) | 19.14 (n=221/7,428) | – (n=18/2,451) | 21.47 | -0.03 | -0.02 |
| in_the_wild:test | 57.22 (n=5,901/2,606) | 58.90 (n=8,185/3,706) | 56.91 (n=3,330/2,230) | 56.13 (n=2,547/3,274) | 57.24 | -0.03 | -0.04 |

## E4c LFCC, trimmed, RawBoost + codec (3 seeds, speech-only duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 15.12 (n=3,710/25,063) | 11.85 (n=3,406/28,940) | 9.32 (n=221/7,428) | – (n=18/2,451) | 12.81 | +0.17 | -0.10 |
| in_the_wild:test | 54.58 (n=5,901/2,606) | 58.06 (n=8,185/3,706) | 55.26 (n=3,330/2,230) | 52.23 (n=2,547/3,274) | 53.45 | -0.27 | -0.25 |

## E4d LFCC, trimmed, full augmentation (3 seeds, speech-only duration)

| Set | < 2 s | 2–4 s | 4–6 s | ≥ 6 s | all | ρ real | ρ fake |
|---|---|---|---|---|---|---|---|
| asvspoof2019_la:eval | 20.84 (n=3,710/25,063) | 17.02 (n=3,406/28,940) | 14.39 (n=221/7,428) | – (n=18/2,451) | 17.96 | +0.12 | -0.13 |
| in_the_wild:test | 36.47 (n=5,901/2,606) | 39.66 (n=8,185/3,706) | 36.37 (n=3,330/2,230) | 38.15 (n=2,547/3,274) | 36.41 | -0.33 | -0.24 |

Bins with fewer than 20 clips of either class have no EER (–).
