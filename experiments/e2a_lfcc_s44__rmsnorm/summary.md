# Results: e2a_lfcc_s44__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.24 |  | 0.0063 | 0.9999 | 99.72 | 99.97 | 99.72 |
| asvspoof2019_la:eval | 71,237 | 8.65 |  | 0.1934 | 0.9646 | 66.61 | 99.97 | 62.79 |
| in_the_wild:test | 31,779 | 49.21 |  |  | 0.5066 | 37.18 | 36.62 | 94.39 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.16 | 0.12 | 0.06 | 2.71 | 2.71 | 8.57 | 3.63 | 1.02 | 3.07 | 0.90 | 30.04 | 25.21 | 0.30 |
