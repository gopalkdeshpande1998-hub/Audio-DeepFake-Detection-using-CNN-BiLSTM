# Results: e4c_rawboost_codec_s44__rmsnorm

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 3.26 |  | 0.0975 | 0.9958 | 96.11 | 99.66 | 95.99 |
| asvspoof2019_la:eval | 71,237 | 13.57 |  | 0.3309 | 0.9400 | 75.84 | 99.56 | 73.39 |
| in_the_wild:test | 31,779 | 62.98 |  |  | 0.3424 | 34.27 | 33.85 | 80.45 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.85 | 0.10 | 0.35 | 28.98 | 5.45 | 36.12 | 18.49 | 1.32 | 6.76 | 5.04 | 17.46 | 4.62 | 3.02 |
