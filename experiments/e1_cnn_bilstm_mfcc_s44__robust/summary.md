# Results: e1_cnn_bilstm_mfcc_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 17.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 8.11 | 7.53–8.74 | 0.2216 | 0.9725 | 69.13 | 99.75 | 63.12 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 10.85 | 10.13–11.66 | 0.3325 | 0.9524 | 84.83 | 98.32 | 83.22 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 11.35 | 10.80–12.03 | 0.264 | 0.9492 | 56.61 | 99.71 | 48.08 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 9.65 | 8.76–10.26 | 0.3086 | 0.9570 | 52.56 | 99.65 | 43.24 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 18.00 | 17.02–19.07 | 0.5838 | 0.8932 | 61.36 | 97.37 | 55.12 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 61.64 | 60.32–62.67 | 0.9998 | 0.3542 | 68.93 | 81.30 | 81.46 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 37.16 | 35.82–38.34 | 0.8487 | 0.7014 | 51.88 | 91.98 | 46.30 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 45.59 | 44.55–46.66 | 0.8582 | 0.5612 | 48.49 | 89.31 | 43.40 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 53.00 | 51.85–53.93 | 0.9683 | 0.4519 | 27.96 | 91.55 | 14.95 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 55.59 | 54.63–56.55 | 0.952 | 0.4162 | 74.35 | 81.84 | 88.97 |
