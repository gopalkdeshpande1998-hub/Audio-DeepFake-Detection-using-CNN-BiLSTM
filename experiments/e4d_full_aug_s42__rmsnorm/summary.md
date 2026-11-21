# Results: e4d_full_aug_s42__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 14.25 |  | 0.3675 | 0.9411 | 89.06 | 97.49 | 90.13 |
| asvspoof2019_la:eval | 71,237 | 18.75 |  | 0.4852 | 0.9009 | 80.99 | 97.47 | 80.91 |
| in_the_wild:test | 31,779 | 38.54 |  |  | 0.6554 | 51.71 | 42.19 | 80.70 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 31.71 | 0.42 | 0.59 | 35.23 | 9.04 | 28.63 | 13.92 | 2.79 | 7.34 | 25.32 | 17.93 | 8.47 | 20.70 |
