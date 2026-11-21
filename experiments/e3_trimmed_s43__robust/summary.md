# Results: e3_trimmed_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 33.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 20.80 | 20.04–21.49 | 0.3366 | 0.8713 | 69.20 | 100.00 | 63.05 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 42.15 | 41.10–43.22 | 0.9868 | 0.6127 | 83.32 | 83.34 | 99.97 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 38.45 | 37.48–39.36 | 0.9027 | 0.6503 | 83.35 | 83.35 | 100.00 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 21.06 | 20.38–21.84 | 0.3387 | 0.8680 | 68.88 | 100.00 | 62.67 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 35.44 | 34.33–36.58 | 0.8186 | 0.7063 | 83.30 | 83.37 | 99.89 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 36.91 | 35.87–37.98 | 0.8658 | 0.6900 | 83.35 | 83.35 | 100.00 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 42.80 | 41.72–43.88 | 0.9471 | 0.6017 | 24.90 | 95.66 | 10.36 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 44.74 | 43.48–45.74 | 0.9998 | 0.5441 | 18.33 | 71.22 | 3.39 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 66.05 | 64.87–67.48 | 1.0 | 0.3021 | 18.96 | 57.86 | 10.19 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 50.49 | 49.34–51.47 | 0.9721 | 0.5009 | 19.43 | 98.26 | 3.39 |
