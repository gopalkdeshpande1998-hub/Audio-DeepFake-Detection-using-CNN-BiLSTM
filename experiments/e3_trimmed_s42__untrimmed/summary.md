# Results: e3_trimmed_s42__untrimmed

Model: cnn_bilstm (786,625 parameters), best dev epoch 10.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.43 | 0.32–0.54 | 0.009 | 0.9999 | 98.94 | 100.00 | 98.82 |
| asvspoof2019_la:eval | 71,237 | 18.19 | 17.88–18.54 | 0.377 | 0.8595 | 64.94 | 99.98 | 60.91 |
| in_the_wild:test | 31,779 | 60.97 | 60.40–61.49 |  | 0.3587 | 36.32 | 35.93 | 91.00 |
| deep_voice:test | 14,896 | 36.89 | 19.02–52.31 |  | 0.7105 | 87.97 | 89.43 | 97.81 |
| deep_voice:test (per recording) | 64 | 37.50 |  |  | 0.7723 | 89.06 | 88.89 | 100.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.26 | 0.02 | 0.04 | 27.78 | 2.62 | 46.60 | 27.37 | 0.28 | 4.78 | 0.55 | 17.30 | 69.17 | 0.16 |
