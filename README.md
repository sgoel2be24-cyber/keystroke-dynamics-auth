# Keystroke Dynamics: Behavioral Biometric Authentication

Semester ML project (team of 3). Can we recognise people by **how** they type a password?

**Dataset:** [CMU Keystroke Dynamics Benchmark](https://www.cs.cmu.edu/~keystroke/) (Killourhy & Maxion, DSN-2009).
51 users typed `.tie5Roanl` 400 times each over 8 sessions. Each sample has 31 timing features: hold times (H),
down-down times (DD) and up-down times (UD).

## Results

| Task | Owner | Best model | Result |
|---|---|---|---|
| 1. Identification (who typed this? 51 classes) | A | 1D-CNN | **89.5%** accuracy (train sessions 1-6, test 7-8) |
| 2. Authentication, classical (is this really you?) | B | kNN, scaled Manhattan | **7.8%** EER (paper's best: 9.6%) |
| 2. Authentication, deep | C | Dense autoencoder | 10.0% EER (LSTM autoencoder: 11.5%) |
| 3. Session drift | C | Retraining on recent sessions | EER 9.3% → 6.5% (static model: 9.3% → 13.1%) |

- B reproduces 7 of the paper's 10 detectors to within 0.1 EER points, so the evaluation matches the published protocol.
- C's scaled-Manhattan check (9.53%) matches B's (9.5%), so classical and deep numbers are directly comparable.
- Typing drifts: by session 8 users type 27% faster than in session 1.

All numbers are in [`results.csv`](results.csv) (`model, task, metric, value`). EER values are stored as fractions (0.078 = 7.8%).

## Split (frozen in [`split.txt`](split.txt))

- **Task 1 (identification):** train on sessions 1-6 (300 reps per user), test on sessions 7-8 (100 reps per user).
  Keys `TASK1_TRAIN_SESSIONS` / `TASK1_TEST_SESSIONS`. Task 1 is not compared with the paper, so it uses more (and more
  recent) training data.
- **Tasks 2 and 3 (authentication, drift):** the paper's protocol: train on sessions 1-4 (200 reps), test on 5-8 (200 reps).
- **Task 2 impostors:** first 5 reps of session 1 from each of the other 50 users (250 per user), as in the paper.
- Splitting by session in time order (not at random) avoids leakage between near-identical samples from the same day.

### Task 1 split comparison

[`experiments/task1_split_comparison/`](experiments/task1_split_comparison/) holds A's notebook run on two other splits.

| Model | Train 1-4, test 5-8 (paper split) | **Train 1-6, test 7-8 (main)** | Train 1-5 + 8, test 6-7 |
|---|---|---|---|
| 1D-CNN | 85.7% | **89.5%** | 94.7% |
| MLP | 84.2% | 88.8% | 93.9% |
| XGBoost | 80.5% | 86.8% | 93.2% |
| Random Forest | 81.8% | 87.0% | 92.9% |
| BiLSTM | 74.9% | 84.5% | 90.9% |
| SVM (RBF) | 76.2% | 81.4% | 87.9% |
| Logistic Regression | 69.1% | 75.8% | 81.6% |

- The closer the training sessions are in time to the test sessions, the higher the accuracy. This agrees with the drift
  results (Task 3).
- On the same test sessions (7-8), the paper-split 1D-CNN scores 82.8%, so two more recent training sessions add about 7 points.
- The 6-7 split is the best case but not realistic: it trains on session 8, which comes *after* the test sessions.
  The main split only predicts later typing from earlier typing, as a real system must.

## Repository layout

```
metrics.py                  shared: EER, results logging, split loader (frozen)
split.txt                   shared: train/test rule (frozen)
ae_models.py                Person C: autoencoders + UserDetector used by the demo
task1_model.py              Person A: 1D-CNN + Identifier used by the demo
scripts/                    train_demo_identifier.py (recreates models/task1_demo_cnn.pt)
data/                       CMU dataset (CSV)
notebooks/
  task1_identification.ipynb      Person A: EDA, LR, SVM, RF, XGBoost, MLP, 1D-CNN, BiLSTM
  task2_classical.ipynb           Person B: 12 detectors, EER vs the paper
  task2_deep_task3_drift.ipynb    Person C: dense/LSTM autoencoders, session drift
results.csv                 all results from the three notebooks
results_task2_classical_per_user.csv   Person B: per-user EER
models/task1_demo_cnn.pt               Person A: 1D-CNN for the demo
models/task2_best_classical.joblib     Person B: kNN detector + per-user thresholds
artifacts/                  Person C: 51 saved LSTM-AE detectors + per-user EER tables
report/figures/             Person C: figures for the report
app/                        Streamlit demo: app.py + one tab per person (tab_a, tab_b, tab_c) + common.py
docs/                       project plan and Person B / C handoff notes
experiments/                Task 1 split comparison (see above)
```

## Running

```bash
pip install -r requirements.txt
```

- **Notebooks:** open from `notebooks/`; each reads `../data`, `../metrics.py`, `../split.txt` and writes its own rows
  to `../results.csv`. Re-running a notebook replaces only that notebook's rows.
- **Demo:** from the repo root, `streamlit run app/app.py`. Nothing is retrained when the app starts; it loads the
  saved models.

## Demo

| Tab | What you can do | Model |
|---|---|---|
| A: Identification | Pick any test sample (or a random one): predicted user, confidence, top-5 guesses, typing-rhythm timeline. Model comparison and accuracy per test session. | 1D-CNN, `models/task1_demo_cnn.pt` (89.5% on sessions 7-8) |
| B: Classical auth | Claim an identity, then type as the real user or an impostor: accept / reject, with an acceptance-limit slider and live false-accept / false-reject rates. All 12 detectors vs the paper. | kNN, scaled Manhattan, `models/task2_best_classical.joblib` |
| C: Deep auth + drift | Same login test with the LSTM autoencoder, plus which keystrokes looked unusual. Deep vs classical comparison, and session-drift charts. | LSTM autoencoder per user, `artifacts/lstm_ae/` |

Code: `app/app.py` (shell), `app/tab_a.py`, `app/tab_b.py`, `app/tab_c.py`, shared helpers in `app/common.py`.

Person A's notebook saves a Random Forest (`models/task1_best_classical.joblib`), but it is about 780 MB, over
GitHub's 100 MB file limit, so it is not committed. The demo uses A's best model instead, the 1D-CNN (1.5 MB),
retrained with the notebook's exact split, architecture and recipe by `python scripts/train_demo_identifier.py`
(about 4 min on CPU). It scores 89.5% on sessions 7-8, the same as the notebook.
