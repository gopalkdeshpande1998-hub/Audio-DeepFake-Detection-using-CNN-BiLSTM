# Results: e7b_wavlm_crnn_itw_s42

Model: crnn (91,998,587 parameters), best dev epoch 2.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la:dev | 24,844 | 0.36 | 0.28–0.51 | 0.0096 | 0.9998 | 99.63 | 99.96 | 99.63 |
| asvspoof2019_la:eval | 71,237 | 7.29 | 7.07–7.49 | 0.1557 | 0.9754 | 83.26 | 99.97 | 81.35 |
| in_the_wild_xd:test | 14,294 | 2.65 | 2.31–2.92 |  | 0.9969 | 98.22 | 98.82 | 96.76 |
| deep_voice:test | 14,896 | 16.81 | 9.11–25.00 |  | 0.9235 | 34.25 | 99.91 | 24.88 |
| deep_voice:test (per recording) | 64 | 9.82 |  |  | 0.9866 | 29.69 | 100.00 | 19.64 |
| speech_dataset:test | 1,866 | 28.40 | 25.94–30.36 |  | 0.7908 | 52.04 | 83.93 | 5.04 |

## asvspoof2019_la:eval: EER per attack (%)

| A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3.82 | 0.07 | 0.04 | 32.29 | 11.70 | 6.24 | 0.91 | 0.37 | 3.58 | 0.73 | 0.18 | 3.25 | 0.38 |
