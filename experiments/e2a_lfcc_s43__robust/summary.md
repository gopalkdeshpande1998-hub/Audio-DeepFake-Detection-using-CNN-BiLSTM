# Results: e2a_lfcc_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 29.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 5.64 | 5.25–6.07 | 0.1355 | 0.9746 | 64.79 | 99.95 | 57.78 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 28.81 | 27.34–30.02 | 0.9426 | 0.7720 | 83.44 | 83.46 | 99.94 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 50.90 | 49.50–52.34 | 0.9995 | 0.5056 | 83.26 | 83.42 | 99.73 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 6.55 | 6.19–7.05 | 0.1516 | 0.9708 | 63.20 | 99.98 | 55.85 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 17.64 | 16.92–18.27 | 0.2943 | 0.8986 | 82.57 | 94.98 | 83.51 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 21.00 | 20.23–21.81 | 0.434 | 0.8477 | 82.41 | 83.27 | 98.73 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 55.21 | 54.13–56.27 | 1.0 | 0.4756 | 33.31 | 84.02 | 24.69 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 66.06 | 65.02–67.24 | 1.0 | 0.2612 | 32.11 | 71.08 | 31.28 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 57.50 | 55.99–59.00 | 0.9996 | 0.4044 | 23.45 | 67.94 | 15.43 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 19.81 | 19.02–20.55 | 0.3621 | 0.8638 | 16.86 | 100.00 | 0.25 |
