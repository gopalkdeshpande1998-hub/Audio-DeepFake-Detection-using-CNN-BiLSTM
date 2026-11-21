# Results: e3_trimmed_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.08 | 0.04–0.20 | 0.0027 | 1.0000 | 99.91 | 99.99 | 99.91 |
| asvspoof2019_la:eval | 71,237 | 21.30 | 20.97–21.66 | 0.3605 | 0.8690 | 65.76 | 99.97 | 61.84 |
| in_the_wild:test | 31,779 | 52.80 | 52.22–53.37 |  | 0.4613 | 36.45 | 36.15 | 92.54 |
| deep_voice:test | 14,896 | 36.68 | 21.68–53.22 |  | 0.6778 | 86.82 | 87.78 | 98.67 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.7143 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.02 | 0.00 | 37.70 | 2.58 | 53.99 | 39.15 | 0.16 | 3.38 | 0.24 | 8.18 | 42.73 | 0.06 |
