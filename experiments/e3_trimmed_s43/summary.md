# Results: e3_trimmed_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 33.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.12 | 0.08–0.23 | 0.0039 | 0.9999 | 99.88 | 99.99 | 99.87 |
| asvspoof2019_la:eval | 71,237 | 20.61 | 20.29–21.00 | 0.3478 | 0.8729 | 66.50 | 99.98 | 62.66 |
| in_the_wild:test | 31,779 | 59.27 | 58.61–59.82 |  | 0.3820 | 35.58 | 36.08 | 94.91 |
| deep_voice:test | 14,896 | 40.86 | 23.69–60.65 |  | 0.6157 | 87.46 | 87.53 | 99.90 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5491 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.06 | 0.06 | 0.02 | 37.36 | 0.71 | 59.81 | 36.49 | 0.45 | 8.44 | 0.22 | 6.94 | 36.69 | 0.00 |
