# Results: e4b_codec_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 46.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.39 | 0.28–0.56 | 0.013 | 0.9997 | 99.60 | 99.95 | 99.60 |
| asvspoof2019_la:eval | 71,237 | 12.42 | 12.09–12.73 | 0.3227 | 0.9487 | 62.96 | 99.94 | 58.72 |
| in_the_wild:test | 31,779 | 67.66 | 67.05–68.27 |  | 0.3053 | 30.46 | 29.37 | 61.97 |
| deep_voice:test | 14,896 | 37.06 | 25.27–49.79 |  | 0.6693 | 84.67 | 88.08 | 95.39 |
| deep_voice:test (per recording) | 64 | 36.61 |  |  | 0.7121 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.57 | 0.08 | 0.10 | 21.53 | 8.77 | 30.32 | 13.21 | 0.53 | 11.30 | 0.68 | 12.01 | 19.07 | 0.10 |
