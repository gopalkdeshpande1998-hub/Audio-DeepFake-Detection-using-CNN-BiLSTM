# Results: e4c_rawboost_codec_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 2.51 | 2.16–2.83 | 0.0732 | 0.9975 | 97.48 | 99.71 | 97.48 |
| asvspoof2019_la:eval | 71,237 | 12.71 | 12.33–13.02 | 0.3091 | 0.9458 | 76.97 | 99.60 | 74.62 |
| in_the_wild:test | 31,779 | 63.46 | 62.89–63.94 |  | 0.3294 | 33.74 | 33.56 | 79.80 |
| deep_voice:test | 14,896 | 37.97 | 21.80–56.35 |  | 0.6691 | 87.22 | 88.16 | 98.64 |
| deep_voice:test (per recording) | 64 | 25.00 |  |  | 0.7522 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.15 | 0.08 | 0.51 | 28.20 | 4.98 | 35.06 | 16.18 | 1.51 | 9.12 | 3.26 | 14.51 | 4.34 | 1.99 |
