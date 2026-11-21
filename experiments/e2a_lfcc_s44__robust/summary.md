# Results: e2a_lfcc_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 19.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 8.26 | 7.72–8.90 | 0.1904 | 0.9682 | 66.88 | 99.95 | 60.29 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 20.25 | 19.15–20.94 | 0.4911 | 0.8867 | 84.18 | 84.20 | 99.73 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 58.75 | 57.77–59.96 | 0.9995 | 0.3931 | 69.28 | 82.63 | 79.95 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 9.20 | 8.65–9.74 | 0.1987 | 0.9635 | 64.77 | 99.97 | 57.75 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 17.80 | 17.12–18.61 | 0.3645 | 0.8915 | 54.79 | 99.85 | 45.82 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 37.55 | 36.60–38.63 | 0.8169 | 0.6482 | 81.94 | 83.16 | 98.22 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 46.46 | 45.46–47.34 | 0.8352 | 0.5763 | 49.03 | 88.22 | 44.83 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 49.74 | 48.73–50.64 | 0.9512 | 0.5143 | 45.08 | 86.10 | 40.67 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 57.30 | 56.13–58.49 | 0.9998 | 0.4006 | 18.53 | 75.57 | 3.34 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 24.11 | 22.68–25.14 | 0.4828 | 0.8667 | 16.79 | 100.00 | 0.17 |
