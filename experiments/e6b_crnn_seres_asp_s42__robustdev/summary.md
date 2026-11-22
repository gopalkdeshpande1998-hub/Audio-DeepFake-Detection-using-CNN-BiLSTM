# Results: e6b_crnn_seres_asp_s42__robustdev

Model: crnn (1,422,206 parameters), best dev epoch 13.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.95 |  | 0.0294 | 0.9994 | 99.06 | 99.61 | 99.05 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 43.30 |  | 0.9757 | 0.6838 | 69.83 | 69.83 | 99.98 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 29.45 |  | 0.8455 | 0.7545 | 69.64 | 69.79 | 99.61 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.91 |  | 0.0282 | 0.9995 | 98.73 | 99.72 | 98.46 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 37.24 |  | 0.9611 | 0.6675 | 69.27 | 69.71 | 98.98 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 39.25 |  | 0.9728 | 0.6711 | 69.79 | 69.79 | 99.98 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 28.10 |  | 0.8117 | 0.7798 | 74.31 | 83.30 | 79.03 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 39.80 |  | 0.9327 | 0.6480 | 70.02 | 70.76 | 97.19 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 50.70 |  | 0.9931 | 0.4911 | 60.82 | 68.34 | 81.71 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 35.26 |  | 0.9214 | 0.7049 | 71.06 | 70.83 | 99.52 |
