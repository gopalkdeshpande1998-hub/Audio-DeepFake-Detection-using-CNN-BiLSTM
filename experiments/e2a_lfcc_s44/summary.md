# Results: e2a_lfcc_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.07 | 0.04–0.15 | 0.0016 | 1.0000 | 99.94 | 100.00 | 99.94 |
| asvspoof2019_la:eval | 71,237 | 8.39 | 8.13–8.61 | 0.1872 | 0.9670 | 64.64 | 99.98 | 60.58 |
| in_the_wild:test | 31,779 | 50.15 | 49.58–50.72 |  | 0.5047 | 37.54 | 36.78 | 94.59 |
| deep_voice:test | 14,896 | 44.03 | 26.47–60.94 |  | 0.5970 | 87.51 | 87.70 | 99.71 |
| deep_voice:test (per recording) | 64 | 48.21 |  |  | 0.5670 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.08 | 0.07 | 2.42 | 1.85 | 12.48 | 3.90 | 0.79 | 3.01 | 0.18 | 28.67 | 21.45 | 0.08 |
