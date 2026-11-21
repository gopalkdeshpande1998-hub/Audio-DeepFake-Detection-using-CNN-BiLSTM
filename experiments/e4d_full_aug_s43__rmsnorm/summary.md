# Results: e4d_full_aug_s43__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 9.85 |  | 0.2477 | 0.9694 | 91.02 | 98.60 | 91.29 |
| asvspoof2019_la:eval | 71,237 | 17.99 |  | 0.486 | 0.9083 | 77.15 | 98.17 | 75.93 |
| in_the_wild:test | 31,779 | 31.74 |  |  | 0.7593 | 68.64 | 56.60 | 67.09 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 28.23 | 0.08 | 0.10 | 33.17 | 14.00 | 27.04 | 13.08 | 1.52 | 6.63 | 19.55 | 14.70 | 22.67 | 13.96 |
