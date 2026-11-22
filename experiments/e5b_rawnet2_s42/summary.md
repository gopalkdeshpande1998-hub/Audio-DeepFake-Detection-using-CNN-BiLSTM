# Results: e5b_rawnet2_s42

Model: rawnet2 (17,620,385 parameters), best dev epoch 12.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 5.25 | 4.80–5.75 | 0.1508 | 0.9894 | 94.76 | 99.37 | 94.76 |
| asvspoof2019_la:eval | 71,237 | 19.51 | 18.98–20.01 | 0.4935 | 0.8935 | 62.34 | 99.26 | 58.44 |
| in_the_wild:test | 31,779 | 44.90 | 44.44–45.50 |  | 0.5567 | 56.45 | 41.60 | 42.37 |
| deep_voice:test | 14,896 | 53.54 | 40.77–67.04 |  | 0.4364 | 33.44 | 84.13 | 29.49 |
| deep_voice:test (per recording) | 64 | 50.89 |  |  | 0.4085 | 25.00 | 83.33 | 17.86 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31.93 | 6.17 | 1.85 | 35.10 | 23.71 | 26.33 | 5.59 | 13.54 | 23.10 | 10.42 | 14.53 | 24.81 | 3.58 |
