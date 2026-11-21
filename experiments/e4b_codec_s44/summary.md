# Results: e4b_codec_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.39 | 0.28–0.63 | 0.0131 | 0.9998 | 99.61 | 99.95 | 99.61 |
| asvspoof2019_la:eval | 71,237 | 14.03 | 13.72–14.44 | 0.3464 | 0.9373 | 66.06 | 99.87 | 62.24 |
| in_the_wild:test | 31,779 | 61.39 | 60.82–61.89 |  | 0.3591 | 31.13 | 28.56 | 56.78 |
| deep_voice:test | 14,896 | 36.30 | 19.11–50.99 |  | 0.6873 | 84.36 | 88.45 | 94.46 |
| deep_voice:test (per recording) | 64 | 25.00 |  |  | 0.7634 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.69 | 0.08 | 0.12 | 24.61 | 11.60 | 34.55 | 18.14 | 0.68 | 4.15 | 1.06 | 14.37 | 22.14 | 0.28 |
