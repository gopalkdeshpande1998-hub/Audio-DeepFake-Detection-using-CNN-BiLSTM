# Results: e4a_rawboost_s43__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 40.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.35 |  | 0.0102 | 0.9994 | 99.65 | 99.85 | 99.65 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 17.96 |  | 0.3422 | 0.9005 | 72.27 | 99.86 | 60.35 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 25.10 |  | 0.493 | 0.8226 | 73.23 | 78.71 | 84.50 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.35 |  | 0.0111 | 0.9994 | 99.52 | 99.87 | 99.44 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 22.66 |  | 0.4443 | 0.8584 | 74.85 | 95.84 | 66.86 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 25.85 |  | 0.4481 | 0.8289 | 74.56 | 91.78 | 69.81 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 11.55 |  | 0.3014 | 0.9563 | 63.26 | 99.73 | 47.49 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 25.95 |  | 0.6692 | 0.8220 | 40.29 | 99.26 | 14.55 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 33.65 |  | 0.803 | 0.7185 | 36.44 | 98.36 | 9.07 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 20.60 |  | 0.5175 | 0.8674 | 62.39 | 98.59 | 46.77 |
