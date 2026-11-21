# Results: e4c_rawboost_codec_s42__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.92 |  | 0.0614 | 0.9978 | 98.68 | 99.59 | 98.94 |
| asvspoof2019_la:eval | 71,237 | 12.96 |  | 0.3223 | 0.9475 | 75.24 | 99.63 | 72.66 |
| in_the_wild:test | 31,779 | 51.04 |  |  | 0.4906 | 37.44 | 36.64 | 93.55 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.21 | 0.02 | 0.14 | 28.13 | 10.87 | 29.29 | 9.89 | 0.63 | 6.39 | 3.19 | 20.21 | 11.35 | 1.47 |
