# Results: e4c_rawboost_codec_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 1.22 | 1.06–1.44 | 0.0289 | 0.9987 | 98.78 | 99.86 | 98.78 |
| asvspoof2019_la:eval | 71,237 | 12.30 | 11.95–12.60 | 0.31 | 0.9516 | 69.56 | 99.85 | 66.15 |
| in_the_wild:test | 31,779 | 52.19 | 51.62–52.76 |  | 0.4705 | 36.55 | 36.10 | 91.78 |
| deep_voice:test | 14,896 | 41.94 | 29.20–54.19 |  | 0.6275 | 88.00 | 88.96 | 98.51 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.6786 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.73 | 0.06 | 0.23 | 27.03 | 11.15 | 27.29 | 9.24 | 0.90 | 9.20 | 1.56 | 17.96 | 9.04 | 0.57 |
