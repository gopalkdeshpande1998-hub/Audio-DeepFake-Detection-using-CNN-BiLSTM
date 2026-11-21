# Results: e4d_full_aug_s44__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 11.31 |  | 0.2692 | 0.9593 | 88.67 | 94.59 | 88.85 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 12.24 |  | 0.2991 | 0.9505 | 87.75 | 93.53 | 88.57 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 16.60 |  | 0.3817 | 0.9177 | 83.93 | 88.52 | 88.44 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 11.46 |  | 0.2749 | 0.9584 | 88.52 | 94.57 | 88.64 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 17.01 |  | 0.4259 | 0.9124 | 82.39 | 82.47 | 94.94 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 15.14 |  | 0.3489 | 0.9330 | 85.09 | 90.98 | 87.29 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 25.85 |  | 0.4862 | 0.8072 | 69.17 | 70.30 | 96.65 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 28.55 |  | 0.6409 | 0.7788 | 69.65 | 69.76 | 99.76 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 35.64 |  | 0.8419 | 0.6971 | 69.71 | 69.78 | 99.85 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 27.81 |  | 0.6321 | 0.8053 | 69.80 | 69.83 | 99.89 |
