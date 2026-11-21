# Results: e1_cnn_bilstm_mfcc_s42__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 27.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 8.20 | 7.76–8.78 | 0.1932 | 0.9742 | 70.39 | 99.88 | 64.56 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 11.25 | 10.62–11.95 | 0.3386 | 0.9525 | 89.30 | 97.21 | 89.74 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 11.30 | 10.62–11.93 | 0.2904 | 0.9506 | 61.41 | 99.70 | 53.86 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 9.90 | 9.22–10.59 | 0.2682 | 0.9602 | 65.95 | 99.80 | 59.27 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 13.60 | 12.76–14.46 | 0.396 | 0.9343 | 80.12 | 97.97 | 77.76 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 69.15 | 67.97–70.41 | 0.9996 | 0.3078 | 78.38 | 83.39 | 92.48 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 32.35 | 31.20–33.54 | 0.8282 | 0.7315 | 51.97 | 94.33 | 45.07 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 48.85 | 47.67–49.85 | 0.9348 | 0.5222 | 25.32 | 96.60 | 10.78 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 57.50 | 56.34–58.55 | 0.9947 | 0.3937 | 17.15 | 96.88 | 0.62 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 54.41 | 53.54–55.36 | 0.9704 | 0.4249 | 19.89 | 97.32 | 4.00 |
