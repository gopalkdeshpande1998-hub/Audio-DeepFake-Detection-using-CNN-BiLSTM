# Results: e5a_lcnn_s42

Model: lcnn (269,729 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.24 | 0.15–0.41 | 0.0077 | 1.0000 | 99.76 | 99.97 | 99.76 |
| asvspoof2019_la:eval | 71,237 | 15.36 | 15.08–15.64 | 0.3179 | 0.9207 | 65.22 | 99.97 | 61.23 |
| in_the_wild:test | 31,779 | 53.59 | 53.03–54.12 |  | 0.4544 | 41.86 | 35.12 | 66.48 |
| deep_voice:test | 14,896 | 30.94 | 14.65–49.87 |  | 0.7515 | 88.26 | 90.48 | 96.76 |
| deep_voice:test (per recording) | 64 | 25.00 |  |  | 0.8348 | 90.62 | 90.32 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.18 | 0.04 | 0.06 | 36.75 | 4.78 | 47.05 | 20.19 | 2.48 | 5.00 | 0.24 | 7.55 | 18.19 | 0.08 |
