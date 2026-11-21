# Results: e2a_lfcc_s43__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.07 |  | 0.0016 | 1.0000 | 99.90 | 100.00 | 99.89 |
| asvspoof2019_la:eval | 71,237 | 6.19 |  | 0.1665 | 0.9721 | 63.92 | 99.98 | 59.78 |
| in_the_wild:test | 31,779 | 41.41 |  |  | 0.6186 | 37.03 | 36.62 | 94.93 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.12 | 0.02 | 0.02 | 1.65 | 1.49 | 1.44 | 1.79 | 0.47 | 2.89 | 0.28 | 31.62 | 15.93 | 0.33 |
