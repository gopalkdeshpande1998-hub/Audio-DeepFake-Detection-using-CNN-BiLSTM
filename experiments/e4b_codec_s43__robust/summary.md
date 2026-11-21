# Results: e4b_codec_s43__robust

Model: cnn_bilstm (786,625 parameters), best dev epoch 46.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 12.51 | 11.70–13.09 | 0.3116 | 0.9502 | 65.40 | 99.88 | 58.55 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 18.70 | 17.89–19.40 | 0.4641 | 0.9042 | 55.66 | 99.53 | 47.02 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 23.00 | 22.00–23.75 | 0.5637 | 0.8636 | 45.66 | 99.57 | 34.96 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 12.65 | 11.95–13.32 | 0.3204 | 0.9481 | 64.35 | 99.97 | 57.24 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 23.11 | 22.02–24.15 | 0.6 | 0.8589 | 51.63 | 98.70 | 42.53 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 21.84 | 20.84–22.80 | 0.5087 | 0.8811 | 54.15 | 99.41 | 45.25 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 57.85 | 56.66–59.05 | 0.9978 | 0.4005 | 17.26 | 86.87 | 0.86 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 76.04 | 75.23–76.93 | 0.9999 | 0.2049 | 18.08 | 62.65 | 4.26 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 67.55 | 66.44–68.70 | 0.9997 | 0.2830 | 16.94 | 56.70 | 1.48 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 49.55 | 48.10–50.66 | 0.9751 | 0.5223 | 16.88 | 100.00 | 0.27 |
