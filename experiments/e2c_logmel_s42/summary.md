# Results: e2c_logmel_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 10.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.07 | 0.03–0.12 | 0.0016 | 1.0000 | 99.94 | 100.00 | 99.94 |
| asvspoof2019_la:eval | 71,237 | 6.65 | 6.45–6.82 | 0.1111 | 0.9671 | 82.77 | 99.98 | 80.80 |
| in_the_wild:test | 31,779 | 60.73 | 60.07–61.34 |  | 0.3512 | 36.12 | 36.28 | 94.94 |
| deep_voice:test | 14,896 | 48.23 | 29.45–67.22 |  | 0.5456 | 87.53 | 87.57 | 99.94 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5290 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.04 | 1.59 | 0.00 | 0.69 | 0.63 | 0.15 | 0.15 | 0.77 | 1.83 | 0.10 | 44.01 | 3.58 | 0.16 |
