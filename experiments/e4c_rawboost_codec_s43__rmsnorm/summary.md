# Results: e4c_rawboost_codec_s43__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 3.18 |  | 0.0897 | 0.9957 | 96.51 | 99.67 | 96.43 |
| asvspoof2019_la:eval | 71,237 | 13.70 |  | 0.347 | 0.9411 | 73.49 | 99.57 | 70.74 |
| in_the_wild:test | 31,779 | 42.29 |  |  | 0.6060 | 48.81 | 40.59 | 81.27 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7.21 | 0.08 | 0.23 | 30.43 | 7.14 | 31.18 | 14.08 | 1.59 | 5.66 | 6.86 | 15.55 | 11.91 | 2.89 |
