# Results: e4d_full_aug_s43__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 8.19 |  | 0.2176 | 0.9772 | 91.81 | 95.99 | 92.12 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 9.20 |  | 0.2456 | 0.9717 | 91.30 | 93.45 | 94.13 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 12.55 |  | 0.3413 | 0.9478 | 88.10 | 90.02 | 93.29 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 8.00 |  | 0.223 | 0.9770 | 91.90 | 95.95 | 92.29 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 13.99 |  | 0.366 | 0.9400 | 83.63 | 81.97 | 98.12 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 11.95 |  | 0.2953 | 0.9563 | 88.70 | 91.37 | 92.55 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 31.29 |  | 0.8104 | 0.7609 | 59.67 | 88.12 | 48.79 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 36.84 |  | 0.9084 | 0.6862 | 52.61 | 83.60 | 39.94 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 41.39 |  | 0.8871 | 0.6146 | 58.75 | 69.80 | 72.06 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 33.90 |  | 0.8899 | 0.7326 | 61.60 | 83.98 | 55.58 |
