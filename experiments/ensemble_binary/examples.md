# Merged detector: end-to-end examples (CPU)

Produced by `scripts/check_ensemble_examples.py`. A handful of clips, so the accuracy here is anecdotal; the dataset-level numbers are in `ensemble.md`.

Structural checks failed in 0 of 22 cases. Verdict correct in 73% of cases.

| case | truth | dur_s | verdict | correct | fake_% | E7a / VGi / VGv2 % | timeline min–max % | steps | flagged segments | time_s | checks | splice: fake part flagged % | splice: real parts flagged % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ASVspoof eval real LA_E_2515281 | REAL | 5.0 | REAL | True | 27.4 | 1 / 49 / 49 | 6–48 | 11 | 0 | 5.7 | ok |  |  |
| ASVspoof eval real LA_E_2293965 | REAL | 7.5 | FAKE | False | 53.7 | 1 / 95 / 100 | 23–52 | 15 | 1 | 3.0 | ok |  |  |
| ASVspoof eval real LA_E_7023214 | REAL | 4.5 | REAL | True | 17.1 | 1 / 35 / 0 | 17–49 | 10 | 0 | 2.0 | ok |  |  |
| In-the-Wild test real (Donald Trump) | REAL | 5.9 | REAL | True | 9.2 | 1 / 0 / 0 | 11–12 | 12 | 0 | 2.9 | ok |  |  |
| In-the-Wild test real (Donald Trump) | REAL | 5.4 | REAL | True | 9.2 | 1 / 0 / 0 | 12–15 | 11 | 0 | 2.6 | ok |  |  |
| In-the-Wild test real (Orson Welles) | REAL | 5.7 | REAL | True | 9.7 | 2 / 0 / 0 | 10–38 | 12 | 0 | 2.8 | ok |  |  |
| ASVspoof eval fake A15 LA_E_7976034 | FAKE | 7.2 | FAKE | True | 68.6 | 61 / 79 / 89 | 61–89 | 15 | 1 | 3.9 | ok |  |  |
| ASVspoof eval fake A17 LA_E_1640969 | FAKE | 4.9 | REAL | False | 47.0 | 99 / 1 / 1 | 76–93 | 10 | 1 | 2.0 | ok |  |  |
| ASVspoof eval fake A19 LA_E_1749948 | FAKE | 5.5 | REAL | False | 47.0 | 99 / 0 / 0 | 18–71 | 11 | 1 | 2.8 | ok |  |  |
| In-the-Wild test fake (William F. Buckley) | FAKE | 5.5 | FAKE | True | 69.9 | 99 / 0 / 100 | 76–87 | 12 | 1 | 2.9 | ok |  |  |
| In-the-Wild test fake (Alec Guinness) | FAKE | 7.9 | FAKE | True | 90.7 | 99 / 94 / 100 | 85–91 | 16 | 1 | 4.2 | ok |  |  |
| In-the-Wild test fake (Alec Guinness) | FAKE | 7.6 | FAKE | True | 82.5 | 99 / 35 / 100 | 64–85 | 16 | 1 | 4.3 | ok |  |  |
| DEEP-VOICE test taylor-original (60 s excerpt) | REAL | 60.0 | REAL | True | 9.2 | 1 / 0 / 2 | 7–19 | 120 | 0 | 39.2 | ok |  |  |
| DEEP-VOICE test taylor-to-trump (60 s excerpt) | FAKE | 60.0 | REAL | False | 12.4 | 6 / 1 / 0 | 8–49 | 120 | 0 | 39.1 | ok |  |  |
| DEEP-VOICE test trump-original (60 s excerpt) | REAL | 60.0 | REAL | True | 9.6 | 2 / 0 / 0 | 7–67 | 120 | 1 | 39.8 | ok |  |  |
| DEEP-VOICE test trump-to-taylor (60 s excerpt) | FAKE | 60.0 | REAL | False | 18.0 | 24 / 0 / 0 | 9–61 | 120 | 3 | 39.6 | ok |  |  |
| DEEP-VOICE trump-original (180 s excerpt, long) | REAL | 180.0 | REAL | True | 9.5 | 2 / 0 / 0 | 5–67 | 121 | 6 | 69.1 | ok |  |  |
| short 1.5 s (ASVspoof real) | REAL | 1.5 | REAL | True | 49.0 | 45 / 86 / 1 | 45–45 | 1 | 0 | 0.8 | ok |  |  |
| short 3.0 s (ASVspoof real) | REAL | 3.0 | REAL | True | 25.5 | 1 / 85 / 1 | 19–40 | 6 | 0 | 1.1 | ok |  |  |
| short 3.0 s (ASVspoof fake) | FAKE | 3.0 | FAKE | True | 92.1 | 99 / 98 / 100 | 92–94 | 6 | 1 | 1.2 | ok |  |  |
| spliced ASVspoof: real | fake | real (fake 5.3–11.9 s) | FAKE | 16.6 | FAKE | True | 54.0 | 82 / 1 / 100 | 16–94 | 34 | 1 | 10.1 | ok | 100.0 | 25.0 |
| spliced In-the-Wild: real | fake | real (fake 8.9–13.9 s) | FAKE | 18.5 | REAL | False | 36.1 | 74 / 0 / 24 | 9–88 | 38 | 1 | 11.8 | ok | 92.0 | 2.0 |
