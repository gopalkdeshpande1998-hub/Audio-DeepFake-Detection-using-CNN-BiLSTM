# Results: e2d_sinc_s44

Model: cnn_bilstm (786,765 parameters), best dev epoch 3.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.08 | 0.04–0.20 | 0.0024 | 1.0000 | 99.92 | 99.99 | 99.91 |
| asvspoof2019_la:eval | 71,237 | 6.75 | 6.56–6.94 | 0.1111 | 0.9668 | 81.06 | 99.95 | 78.92 |
| in_the_wild:test | 31,779 | 60.77 | 60.27–61.33 |  | 0.3770 | 37.65 | 36.43 | 90.83 |
| deep_voice:test | 14,896 | 44.09 | 24.63–64.64 |  | 0.5867 | 87.91 | 88.24 | 99.44 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.5536 | 87.50 | 87.50 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.83 | 0.26 | 0.91 | 0.98 | 0.61 | 0.38 | 1.12 | 1.47 | 0.24 | 44.10 | 2.38 | 0.22 |
