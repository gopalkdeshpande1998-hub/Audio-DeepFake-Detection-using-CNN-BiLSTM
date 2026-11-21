# Results: e3_trimmed_s42

Model: cnn_bilstm (786,625 parameters), best dev epoch 10.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.20 | 0.15–0.35 | 0.0057 | 0.9999 | 99.80 | 99.98 | 99.80 |
| asvspoof2019_la:eval | 71,237 | 22.50 | 22.09–22.86 | 0.358 | 0.8540 | 67.89 | 99.95 | 64.22 |
| in_the_wild:test | 31,779 | 59.66 | 59.08–60.16 |  | 0.3687 | 36.17 | 35.91 | 91.28 |
| deep_voice:test | 14,896 | 36.89 | 19.02–52.31 |  | 0.7105 | 87.97 | 89.43 | 97.81 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.7723 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.28 | 0.02 | 0.00 | 33.12 | 3.28 | 56.33 | 33.09 | 0.30 | 4.46 | 0.55 | 10.36 | 57.55 | 0.07 |
