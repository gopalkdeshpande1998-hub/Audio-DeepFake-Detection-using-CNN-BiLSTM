# Results: e3_trimmed_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 10.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 22.99 | 22.06–23.75 | 0.3546 | 0.8519 | 70.36 | 99.91 | 64.50 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 30.30 | 29.09–31.23 | 0.7192 | 0.7712 | 83.21 | 83.37 | 99.74 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 39.90 | 38.92–40.94 | 0.8082 | 0.6513 | 83.34 | 83.35 | 99.99 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 23.00 | 22.19–23.79 | 0.3554 | 0.8465 | 70.18 | 99.91 | 64.29 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 39.70 | 38.66–40.73 | 0.848 | 0.6535 | 81.72 | 83.54 | 97.22 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 39.35 | 38.01–40.58 | 0.9535 | 0.6530 | 83.35 | 83.35 | 100.00 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 17.25 | 16.50–18.01 | 0.3522 | 0.8805 | 18.46 | 100.00 | 2.17 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 32.10 | 31.33–32.94 | 0.8967 | 0.6676 | 16.67 | 100.00 | 0.02 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 53.65 | 52.47–54.93 | 0.9981 | 0.3701 | 16.65 | 0.00 | 0.00 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 24.14 | 23.28–24.95 | 0.5458 | 0.8272 | 16.66 | 100.00 | 0.01 |
