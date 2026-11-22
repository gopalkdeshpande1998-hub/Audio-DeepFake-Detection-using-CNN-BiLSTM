# Results: e5a_lcnn_s42__robust

Model: lcnn (269,729 parameters), best dev epoch 22.
Score threshold for accuracy/precision/recall: dev EER threshold (spoof = positive class).

| Test set | n | EER % | EER 95% CI | min t-DCF | AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|---|---|---|---|
| asvspoof2019_la_robust_clean:test | 12,010 | 15.36 | 14.70–16.02 | 0.3118 | 0.9218 | 67.62 | 99.97 | 61.17 |
| asvspoof2019_la_robust_mp3_51k:test | 12,010 | 27.20 | 26.23–28.12 | 0.6295 | 0.8158 | 77.44 | 91.27 | 80.64 |
| asvspoof2019_la_robust_mp3_19k:test | 12,010 | 28.95 | 27.99–29.94 | 0.662 | 0.7835 | 82.98 | 83.72 | 98.79 |
| asvspoof2019_la_robust_opus_84k:test | 12,010 | 15.56 | 14.85–16.26 | 0.3093 | 0.9196 | 66.89 | 99.98 | 60.29 |
| asvspoof2019_la_robust_opus_10k:test | 12,010 | 29.10 | 28.08–30.17 | 0.6964 | 0.7913 | 83.00 | 86.46 | 94.39 |
| asvspoof2019_la_robust_telephone:test | 12,010 | 26.46 | 25.56–27.43 | 0.6612 | 0.8171 | 84.12 | 85.04 | 98.23 |
| asvspoof2019_la_robust_white_20db:test | 12,010 | 22.55 | 21.74–23.43 | 0.5171 | 0.8579 | 52.13 | 99.67 | 42.71 |
| asvspoof2019_la_robust_white_10db:test | 12,010 | 27.35 | 26.53–28.37 | 0.6784 | 0.7958 | 62.00 | 96.12 | 56.69 |
| asvspoof2019_la_robust_white_5db:test | 12,010 | 49.01 | 48.16–49.99 | 0.9547 | 0.5131 | 30.19 | 91.86 | 17.82 |
| asvspoof2019_la_robust_pink_10db:test | 12,010 | 22.90 | 21.90–23.89 | 0.5727 | 0.8453 | 66.05 | 97.43 | 60.88 |
