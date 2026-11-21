# Results: e2c_logmel_s44

Model: cnn_bilstm (786,625 parameters), best dev epoch 2.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.16 | 0.04–0.31 | 0.0041 | 1.0000 | 99.84 | 99.98 | 99.84 |
| asvspoof2019_la:eval | 71,237 | 5.80 | 5.62–5.97 | 0.1171 | 0.9762 | 82.16 | 99.97 | 80.12 |
| in_the_wild:test | 31,779 | 59.55 | 58.99–60.19 |  | 0.3576 | 33.22 | 34.40 | 87.77 |
| deep_voice:test | 14,896 | 50.91 | 34.14–66.89 |  | 0.5120 | 88.20 | 88.15 | 99.96 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.4219 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.06 | 2.20 | 0.10 | 0.84 | 0.95 | 0.41 | 0.49 | 1.22 | 2.19 | 0.14 | 34.59 | 2.89 | 0.49 |
