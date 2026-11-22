# Results: e5b_rawnet2_s42__robustdev

Model: rawnet2 (17,620,385 parameters), best dev epoch 12.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 5.10 |  | 0.1418 | 0.9901 | 94.89 | 97.81 | 94.81 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 5.25 |  | 0.1501 | 0.9897 | 95.00 | 97.48 | 95.30 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 5.80 |  | 0.1693 | 0.9860 | 94.41 | 96.20 | 95.78 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 5.15 |  | 0.1396 | 0.9901 | 94.82 | 97.75 | 94.76 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 7.25 |  | 0.1994 | 0.9791 | 92.49 | 92.37 | 97.27 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 10.30 |  | 0.2961 | 0.9647 | 88.90 | 96.09 | 87.66 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 17.50 |  | 0.4543 | 0.8844 | 71.53 | 97.47 | 60.78 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 29.80 |  | 0.6884 | 0.7615 | 54.44 | 97.12 | 35.78 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 28.05 |  | 0.6919 | 0.7875 | 46.95 | 98.68 | 24.31 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 28.31 |  | 0.6884 | 0.7728 | 71.37 | 85.83 | 70.65 |
