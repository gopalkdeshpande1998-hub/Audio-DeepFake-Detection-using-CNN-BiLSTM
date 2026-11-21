# Results: e2d_sinc_s42

Model: cnn_bilstm (786,765 parameters), best dev epoch 12.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.08 | 0.04–0.12 | 0.0019 | 1.0000 | 99.92 | 99.99 | 99.91 |
| asvspoof2019_la:eval | 71,237 | 7.15 | 6.95–7.36 | 0.1511 | 0.9687 | 73.03 | 99.98 | 69.94 |
| in_the_wild:test | 31,779 | 53.77 | 53.13–54.32 |  | 0.4573 | 38.76 | 36.55 | 87.97 |
| deep_voice:test | 14,896 | 41.34 | 18.61–62.96 |  | 0.6277 | 86.94 | 88.46 | 97.83 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.6116 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.12 | 0.27 | 0.20 | 1.52 | 2.14 | 0.55 | 0.65 | 2.79 | 5.27 | 0.16 | 40.13 | 5.86 | 0.27 |
