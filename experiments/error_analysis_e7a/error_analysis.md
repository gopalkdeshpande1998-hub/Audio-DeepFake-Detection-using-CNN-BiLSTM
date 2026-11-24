# Error analysis of WavLM + CNN-BiLSTM-Att (E7a), the app's only model (P4.6)

Produced by `scripts/error_analysis.py` from the stored per-utterance scores; decision: FAKE if E7a's calibrated P(fake) >= 0.5.

## ASVspoof eval

n = 71,237 · accuracy 83.3% · real → FAKE 0.1% · fake → REAL 18.6% · EER 6.22%

**Accuracy by confidence of the model probability**

| confidence | n | accuracy % |
|---|---|---|
| very low (50±10%) | 1,499 | 47.3 |
| low (60–70% / 30–40%) | 1,676 | 48.3 |
| medium (70–80% / 20–30%) | 1,996 | 47.7 |
| high (80–90% / 10–20%) | 2,989 | 52.8 |
| very high (≥90% / ≤10%) | 63,077 | 87.7 |

**Per attack** (fake → REAL at 0.5; EER against all bona fide eval)

| attack | n | fake → REAL % | EER % |
|---|---|---|---|
| A07 | 4,914 | 22.0 | 3.42 |
| A08 | 4,914 | 0.3 | 0.15 |
| A09 | 4,914 | 0.0 | 0.06 |
| A10 | 4,914 | 93.0 | 31.11 |
| A11 | 4,914 | 52.0 | 6.05 |
| A12 | 4,914 | 34.2 | 7.29 |
| A13 | 4,914 | 0.6 | 0.20 |
| A14 | 4,914 | 1.9 | 0.47 |
| A15 | 4,914 | 14.2 | 1.79 |
| A16 | 4,914 | 5.1 | 1.67 |
| A17 | 4,914 | 0.3 | 0.16 |
| A18 | 4,914 | 14.9 | 1.77 |
| A19 | 4,914 | 2.5 | 0.71 |

**By clip duration**

| duration | n | accuracy % | real → FAKE % | fake → REAL % |
|---|---|---|---|---|
| < 2 s | 16,631 | 66.0 | 0.8 | 34.6 |
| 2–3 s | 23,117 | 83.2 | 0.1 | 19.1 |
| 3–4 s | 15,644 | 90.8 | 0.1 | 11.1 |
| 4–6 s | 12,275 | 94.0 | 0.1 | 6.8 |
| ≥ 6 s | 3,570 | 96.0 | 0.0 | 4.2 |

## In-the-Wild

n = 31,779 · accuracy 88.9% · real → FAKE 6.7% · fake → REAL 18.6% · EER 10.67%

**Accuracy by confidence of the model probability**

| confidence | n | accuracy % |
|---|---|---|
| very low (50±10%) | 909 | 52.1 |
| low (60–70% / 30–40%) | 1,067 | 53.3 |
| medium (70–80% / 20–30%) | 1,263 | 58.6 |
| high (80–90% / 10–20%) | 2,055 | 67.9 |
| very high (≥90% / ≤10%) | 26,485 | 94.6 |

**By clip duration**

| duration | n | accuracy % | real → FAKE % | fake → REAL % |
|---|---|---|---|---|
| < 2 s | 7,084 | 83.3 | 9.0 | 37.2 |
| 2–3 s | 7,127 | 87.7 | 3.6 | 31.5 |
| 3–4 s | 5,174 | 91.7 | 5.2 | 14.6 |
| 4–6 s | 6,115 | 92.1 | 7.0 | 9.3 |
| ≥ 6 s | 6,279 | 91.0 | 9.7 | 8.4 |

**Hardest In-the-Wild speakers** (lowest accuracy, ≥ 30 clips)

| speaker | n | accuracy % |
|---|---|---|
| Adam Driver | 217 | 41.9 |
| Malcolm X | 176 | 47.2 |
| Frank Sinatra | 55 | 65.5 |
| Nelson Mandela | 381 | 67.2 |
| FDR | 471 | 68.2 |
| Calvin Coolidge | 73 | 69.9 |
| Martin Luther King | 799 | 72.1 |
| George W. Bush | 230 | 73.0 |
| Queen Elizabeth II | 464 | 74.1 |
| Alan Watts | 378 | 74.6 |
