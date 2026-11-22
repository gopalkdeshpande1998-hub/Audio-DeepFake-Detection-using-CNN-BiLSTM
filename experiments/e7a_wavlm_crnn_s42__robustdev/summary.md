# Results: e7a_wavlm_crnn_s42__robustdev

Model: crnn (91,998,587 parameters), best dev epoch 2.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.85 |  | 0.0148 | 0.9995 | 99.12 | 99.59 | 99.16 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 0.80 |  | 0.0166 | 0.9995 | 99.09 | 99.44 | 99.26 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 1.21 |  | 0.0337 | 0.9987 | 97.90 | 97.44 | 99.61 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.81 |  | 0.0158 | 0.9995 | 99.21 | 99.70 | 99.18 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 3.55 |  | 0.1175 | 0.9942 | 86.83 | 84.15 | 99.96 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 1.60 |  | 0.0411 | 0.9983 | 98.04 | 97.91 | 99.31 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 1.94 |  | 0.0446 | 0.9953 | 97.79 | 99.64 | 97.19 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 4.35 |  | 0.1088 | 0.9898 | 95.68 | 98.24 | 95.52 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 8.85 |  | 0.2308 | 0.9728 | 90.29 | 89.54 | 97.47 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 4.26 |  | 0.0979 | 0.9877 | 94.35 | 99.58 | 92.29 |
