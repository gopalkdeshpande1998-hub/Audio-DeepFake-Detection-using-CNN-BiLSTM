# Results: e2b_cqcc_s43

Model: cnn_bilstm (786,625 parameters), best dev epoch 8.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 2.04 | 1.81–2.51 | 0.0695 | 0.9972 | 97.95 | 99.76 | 97.95 |
| asvspoof2019_la:eval | 71,237 | 11.37 | 11.11–11.64 | 0.2244 | 0.9484 | 82.66 | 99.77 | 80.85 |
| in_the_wild:test | 31,779 | 26.88 | 26.43–27.38 |  | 0.8117 | 69.67 | 55.88 | 87.54 |
| deep_voice:test | 14,896 | 41.23 | 27.70–52.52 |  | 0.6465 | 56.51 | 91.05 | 55.78 |
| deep_voice:test (per recording) | 64 | 25.00 |  |  | 0.7723 | 45.31 | 92.00 | 41.07 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.53 | 1.63 | 0.28 | 1.39 | 0.81 | 1.26 | 2.24 | 1.53 | 0.55 | 1.87 | 31.48 | 38.59 | 12.11 |
