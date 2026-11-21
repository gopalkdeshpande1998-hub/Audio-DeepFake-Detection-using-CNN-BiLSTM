# Results: e2a_lfcc_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 28.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 6.05 | 5.61–6.43 | 0.1359 | 0.9749 | 76.14 | 99.97 | 71.40 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 16.15 | 15.48–16.84 | 0.3998 | 0.9095 | 84.21 | 84.22 | 99.75 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 25.96 | 25.18–26.97 | 0.7066 | 0.7803 | 83.32 | 83.40 | 99.87 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 6.99 | 6.50–7.39 | 0.1611 | 0.9707 | 74.22 | 99.97 | 69.09 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 16.85 | 16.19–17.55 | 0.2833 | 0.9004 | 83.80 | 90.10 | 90.51 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 22.50 | 21.79–23.24 | 0.3991 | 0.8379 | 83.35 | 83.35 | 100.00 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 56.35 | 55.27–57.27 | 0.9998 | 0.4133 | 39.56 | 80.64 | 36.16 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 54.85 | 53.48–56.08 | 1.0 | 0.3701 | 60.78 | 80.61 | 69.72 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 54.45 | 52.85–55.71 | 1.0 | 0.4133 | 77.08 | 82.74 | 91.61 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 53.60 | 52.71–54.63 | 0.8356 | 0.4534 | 17.92 | 100.00 | 1.52 |
