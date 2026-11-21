# Results: e2d_sinc_s43

Model: cnn_bilstm (786,765 parameters), best dev epoch 10.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.08 | 0.04–0.12 | 0.0013 | 1.0000 | 99.92 | 99.99 | 99.91 |
| asvspoof2019_la:eval | 71,237 | 6.94 | 6.73–7.12 | 0.1258 | 0.9618 | 74.48 | 99.97 | 71.57 |
| in_the_wild:test | 31,779 | 51.51 | 50.84–52.15 |  | 0.4706 | 36.28 | 35.12 | 84.26 |
| deep_voice:test | 14,896 | 43.72 | 24.21–63.42 |  | 0.5943 | 88.69 | 89.44 | 98.73 |
| deep_voice:test (per recording) | 64 | 39.29 |  |  | 0.5781 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.18 | 0.61 | 0.22 | 1.29 | 1.08 | 0.67 | 0.72 | 1.75 | 3.18 | 0.19 | 46.99 | 0.88 | 0.10 |
