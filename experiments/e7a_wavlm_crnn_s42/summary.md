# Results: e7a_wavlm_crnn_s42

Model: crnn (91,998,587 parameters), best dev epoch 2.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.78 | 0.65–0.88 | 0.0137 | 0.9994 | 99.22 | 99.91 | 99.22 |
| asvspoof2019_la:eval | 71,237 | 6.22 | 6.05–6.42 | 0.1254 | 0.9793 | 88.35 | 99.94 | 87.06 |
| in_the_wild:test | 31,779 | 10.67 | 10.32–11.01 |  | 0.9586 | 88.77 | 80.52 | 92.05 |
| deep_voice:test | 14,896 | 23.68 | 13.66–37.63 |  | 0.8300 | 54.07 | 96.94 | 49.06 |
| deep_voice:test (per recording) | 64 | 12.50 |  |  | 0.9219 | 54.69 | 96.55 | 50.00 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.42 | 0.15 | 0.06 | 31.11 | 6.05 | 7.29 | 0.20 | 0.47 | 1.79 | 1.67 | 0.16 | 1.77 | 0.71 |
