# Results: e4d_full_aug_s42__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 12.40 |  | 0.3309 | 0.9533 | 87.69 | 94.19 | 87.77 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 14.99 |  | 0.3685 | 0.9353 | 85.86 | 90.54 | 89.05 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 17.15 |  | 0.4497 | 0.9124 | 82.19 | 92.41 | 81.15 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 12.51 |  | 0.3337 | 0.9522 | 87.52 | 93.97 | 87.75 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 18.11 |  | 0.4486 | 0.9044 | 83.10 | 86.37 | 89.98 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 15.50 |  | 0.3763 | 0.9276 | 82.58 | 94.57 | 79.61 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 23.00 |  | 0.5751 | 0.8360 | 71.18 | 94.14 | 62.60 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 27.26 |  | 0.6582 | 0.7906 | 56.66 | 97.10 | 39.07 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 33.70 |  | 0.7593 | 0.7242 | 68.52 | 77.13 | 78.03 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 26.89 |  | 0.5974 | 0.7873 | 70.36 | 90.84 | 63.98 |
