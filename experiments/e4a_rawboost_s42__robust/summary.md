# Results: e4a_rawboost_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 21.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 15.40 | 14.78–16.11 | 0.3596 | 0.9279 | 63.27 | 99.95 | 55.96 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 30.59 | 29.49–31.64 | 0.7324 | 0.7783 | 58.32 | 94.87 | 52.85 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 29.95 | 28.68–31.13 | 0.7123 | 0.7874 | 83.63 | 85.53 | 96.72 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 15.35 | 14.73–16.13 | 0.3634 | 0.9273 | 61.91 | 99.96 | 54.32 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 26.29 | 25.29–27.06 | 0.6969 | 0.8125 | 77.63 | 91.75 | 80.39 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 27.91 | 26.94–29.00 | 0.6994 | 0.8029 | 84.15 | 85.92 | 96.84 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 24.70 | 23.90–25.57 | 0.5757 | 0.8410 | 44.00 | 99.79 | 32.89 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 29.70 | 28.25–30.94 | 0.8227 | 0.7811 | 28.59 | 98.06 | 14.62 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 38.95 | 37.97–39.85 | 0.8398 | 0.6614 | 20.61 | 98.17 | 4.84 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 25.40 | 24.36–26.33 | 0.6746 | 0.8193 | 42.14 | 99.07 | 30.87 |
