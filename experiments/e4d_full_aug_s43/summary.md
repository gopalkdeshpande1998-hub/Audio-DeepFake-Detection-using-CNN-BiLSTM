# Results: e4d_full_aug_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 8.90 | 8.31–9.33 | 0.2325 | 0.9744 | 91.10 | 98.89 | 91.11 |
| asvspoof2019_la:eval | 71,237 | 17.66 | 17.29–18.10 | 0.4719 | 0.9115 | 74.80 | 98.63 | 72.91 |
| in_the_wild:test | 31,779 | 32.09 | 31.53–32.70 |  | 0.7612 | 66.95 | 54.25 | 70.84 |
| deep_voice:test | 14,896 | 51.45 | 44.41–59.43 |  | 0.4733 | 53.91 | 86.50 | 56.08 |
| deep_voice:test (per recording) | 64 | 50.00 |  |  | 0.5312 | 60.94 | 87.80 | 64.29 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30.75 | 0.14 | 0.22 | 32.03 | 15.04 | 24.99 | 9.14 | 2.16 | 8.16 | 17.16 | 14.30 | 24.44 | 12.54 |
