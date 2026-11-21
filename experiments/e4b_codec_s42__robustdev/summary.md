# Results: e4b_codec_s42__robustdev

Model: cnn_bilstm (786,625 parameters), best dev epoch 4.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robustdev_clean:test | 6,620 | 0.95 |  | 0.0231 | 0.9994 | 99.00 | 99.61 | 98.96 |
| asvspoof2019_la_robustdev_mp3_51k:test | 6,620 | 21.70 |  | 0.3748 | 0.8754 | 75.82 | 98.27 | 66.52 |
| asvspoof2019_la_robustdev_mp3_19k:test | 6,620 | 24.55 |  | 0.5453 | 0.8319 | 75.12 | 90.89 | 71.52 |
| asvspoof2019_la_robustdev_opus_84k:test | 6,620 | 1.05 |  | 0.023 | 0.9992 | 98.78 | 99.74 | 98.51 |
| asvspoof2019_la_robustdev_opus_10k:test | 6,620 | 25.39 |  | 0.457 | 0.8439 | 73.90 | 94.68 | 66.32 |
| asvspoof2019_la_robustdev_telephone:test | 6,620 | 25.65 |  | 0.4344 | 0.8324 | 74.08 | 83.43 | 78.44 |
| asvspoof2019_la_robustdev_white_20db:test | 6,620 | 53.20 |  | 0.8877 | 0.4986 | 32.73 | 100.00 | 3.61 |
| asvspoof2019_la_robustdev_white_10db:test | 6,620 | 42.61 |  | 0.9545 | 0.6109 | 31.66 | 87.50 | 2.42 |
| asvspoof2019_la_robustdev_white_5db:test | 6,620 | 57.00 |  | 0.9991 | 0.4301 | 33.10 | 62.17 | 10.56 |
| asvspoof2019_la_robustdev_pink_10db:test | 6,620 | 31.04 |  | 0.6606 | 0.7718 | 30.76 | 100.00 | 0.78 |
