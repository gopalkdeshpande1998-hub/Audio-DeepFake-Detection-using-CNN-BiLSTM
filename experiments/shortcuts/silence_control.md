# Silence / duration shortcut control

EER (%) of classifiers that see only silence and duration, never the audio content.
Trained on ASVspoof 2019 LA train. 50% = no information.

| Input | asvspoof2019_la:dev | asvspoof2019_la:eval | in_the_wild:test |
|---|---|---|---|
| lead_s only (longer = more likely real) | 32.57 | 15.93 | 68.04 |
| trail_s only (longer = more likely fake) | 51.22 | 61.65 | 32.57 |
| total_s only (longer = more likely real) | 48.16 | 41.96 | 58.41 |
| speech_s only (longer = more likely fake) | 44.95 | 43.86 | 42.61 |
| all 4, logistic regression | 30.77 | 20.05 | 45.44 |
| all 4, gradient boosting | 23.16 | 23.94 | 44.37 |
