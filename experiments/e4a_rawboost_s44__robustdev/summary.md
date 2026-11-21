# Results: e4a_rawboost_s44__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 40.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.15 |  | 0.0045 | 1.0000 | 99.82 | 99.93 | 99.81 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 19.00 |  | 0.3661 | 0.8963 | 77.08 | 98.50 | 68.20 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 22.90 |  | 0.4408 | 0.8527 | 77.27 | 84.13 | 83.10 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.15 |  | 0.0039 | 1.0000 | 99.70 | 99.98 | 99.59 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 19.15 |  | 0.4089 | 0.9021 | 74.76 | 98.33 | 64.94 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 23.95 |  | 0.4547 | 0.8414 | 76.09 | 88.38 | 75.69 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 9.00 |  | 0.2434 | 0.9706 | 57.58 | 99.89 | 39.26 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 22.00 |  | 0.5628 | 0.8546 | 55.06 | 98.30 | 36.23 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 31.30 |  | 0.7741 | 0.7628 | 47.58 | 94.78 | 26.34 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 15.86 |  | 0.388 | 0.9141 | 76.72 | 97.74 | 68.23 |
