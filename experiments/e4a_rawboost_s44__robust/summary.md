# Results: e4a_rawboost_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 40.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 13.65 | 12.95–14.42 | 0.3344 | 0.9424 | 62.09 | 99.98 | 54.53 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 24.10 | 23.05–25.09 | 0.5718 | 0.8581 | 49.15 | 99.44 | 39.21 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 31.65 | 30.69–32.62 | 0.7297 | 0.7671 | 71.30 | 90.61 | 73.15 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 13.85 | 13.26–14.53 | 0.3271 | 0.9428 | 61.22 | 100.00 | 53.47 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 26.76 | 25.66–27.74 | 0.6688 | 0.8195 | 45.99 | 98.57 | 35.71 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 27.61 | 26.68–28.42 | 0.6652 | 0.8013 | 69.33 | 94.34 | 67.24 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 13.25 | 12.51–13.96 | 0.3472 | 0.9462 | 38.24 | 99.96 | 25.91 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 20.55 | 19.58–21.42 | 0.5467 | 0.8697 | 43.28 | 99.63 | 32.07 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 31.09 | 30.07–32.19 | 0.7872 | 0.7756 | 38.23 | 97.75 | 26.49 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 17.75 | 16.89–18.77 | 0.4686 | 0.8940 | 62.16 | 99.05 | 55.13 |
