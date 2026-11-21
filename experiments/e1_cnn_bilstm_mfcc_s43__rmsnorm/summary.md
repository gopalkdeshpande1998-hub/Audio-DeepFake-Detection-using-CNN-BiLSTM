# Results: e1_cnn_bilstm_mfcc_s43__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.92 |  | 0.0621 | 0.9979 | 97.56 | 99.81 | 97.47 |
| asvspoof2019_la:eval | 71,237 | 8.00 |  | 0.2203 | 0.9726 | 68.46 | 99.85 | 64.92 |
| in_the_wild:test | 31,779 | 51.06 |  |  | 0.5088 | 48.81 | 37.71 | 57.82 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.14 | 6.73 | 1.55 | 4.15 | 2.99 | 4.13 | 3.62 | 5.68 | 3.58 | 2.69 | 28.02 | 13.62 | 7.10 |
