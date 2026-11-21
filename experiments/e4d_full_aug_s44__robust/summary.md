# Results: e4d_full_aug_s44__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 17.41 | 16.58–18.30 | 0.4456 | 0.9168 | 75.76 | 97.52 | 72.77 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 17.95 | 17.00–18.89 | 0.4377 | 0.9122 | 76.55 | 97.13 | 74.06 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 21.35 | 20.34–22.08 | 0.5393 | 0.8808 | 78.54 | 95.03 | 78.35 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 17.35 | 16.61–18.14 | 0.4386 | 0.9169 | 75.77 | 97.57 | 72.74 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 20.55 | 19.79–21.46 | 0.5294 | 0.8812 | 84.51 | 92.27 | 88.86 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 21.15 | 20.05–21.98 | 0.4999 | 0.8859 | 75.74 | 95.98 | 73.99 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 33.40 | 32.49–34.18 | 0.5993 | 0.7245 | 78.48 | 83.30 | 92.78 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 32.59 | 31.76–33.58 | 0.7067 | 0.7219 | 82.81 | 83.27 | 99.32 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 41.80 | 40.60–42.81 | 0.9106 | 0.6147 | 83.16 | 83.34 | 99.73 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 31.90 | 30.94–32.79 | 0.6776 | 0.7558 | 83.22 | 83.37 | 99.78 |
