# Results: e5a_lcnn_s42__robustdev

Model: lcnn (269,729 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.26 |  | 0.007 | 1.0000 | 99.74 | 99.89 | 99.74 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 22.45 |  | 0.4269 | 0.8679 | 77.24 | 79.10 | 91.58 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 25.41 |  | 0.4553 | 0.8208 | 69.24 | 69.89 | 98.27 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.30 |  | 0.008 | 1.0000 | 99.68 | 99.87 | 99.68 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 25.21 |  | 0.4732 | 0.8341 | 71.84 | 72.75 | 95.39 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 23.85 |  | 0.4117 | 0.8501 | 70.98 | 71.02 | 98.68 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 12.31 |  | 0.3219 | 0.9468 | 76.78 | 98.83 | 67.53 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 25.10 |  | 0.6712 | 0.8277 | 69.44 | 91.42 | 62.03 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 40.50 |  | 0.9303 | 0.6338 | 43.96 | 88.43 | 22.66 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 19.85 |  | 0.5392 | 0.8836 | 74.80 | 94.27 | 68.03 |
