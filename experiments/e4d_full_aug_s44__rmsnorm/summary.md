# Results: e4d_full_aug_s44__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 12.01 |  | 0.3088 | 0.9539 | 89.66 | 98.12 | 90.21 |
| asvspoof2019_la:eval | 71,237 | 18.44 |  | 0.4613 | 0.9093 | 77.26 | 98.15 | 76.07 |
| in_the_wild:test | 31,779 | 38.42 |  |  | 0.6657 | 49.12 | 40.97 | 83.59 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 28.49 | 0.18 | 0.27 | 30.52 | 13.12 | 26.72 | 8.47 | 0.83 | 7.04 | 20.12 | 21.13 | 17.50 | 22.83 |
