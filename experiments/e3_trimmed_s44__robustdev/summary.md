# Results: e3_trimmed_s44__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.14 |  | 0.0037 | 1.0000 | 99.88 | 99.96 | 99.87 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 22.55 |  | 0.6715 | 0.8267 | 69.83 | 69.82 | 100.00 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 26.25 |  | 0.5756 | 0.8071 | 69.77 | 69.78 | 99.98 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 0.20 |  | 0.0045 | 0.9999 | 99.79 | 99.96 | 99.74 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 31.75 |  | 0.7099 | 0.7450 | 69.73 | 69.78 | 99.87 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 33.00 |  | 0.662 | 0.7482 | 69.79 | 69.79 | 100.00 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 23.35 |  | 0.5626 | 0.8237 | 35.42 | 100.00 | 7.47 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 39.05 |  | 0.8759 | 0.6500 | 30.47 | 100.00 | 0.37 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 37.40 |  | 0.9015 | 0.6613 | 30.62 | 100.00 | 0.58 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 28.40 |  | 0.8039 | 0.7893 | 31.39 | 100.00 | 1.69 |
