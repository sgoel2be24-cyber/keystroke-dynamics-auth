# Keystroke Dynamics: Behavioral Biometric Authentication

Semester ML project (team of 3). Can we recognise people by **how** they type a password?

**Dataset:** [CMU Keystroke Dynamics Benchmark](https://www.cs.cmu.edu/~keystroke/) (Killourhy & Maxion, DSN-2009).
51 users typed `.tie5Roanl` 400 times each over 8 sessions. Each sample has 31 timing features: hold times (H),
down-down times (DD) and up-down times (UD).

## Results

| Task | Owner | Best model | Result |
|---|---|---|---|
| 1. Identification (who typed this? 51 classes) | A | 1D-CNN | **85.7%** accuracy |
| 2. Authentication, classical (is this really you?) | B | kNN, scaled Manhattan | **7.8%** EER (paper's best: 9.6%) |
| 2. Authentication, deep | C | Dense autoencoder | 10.0% EER (LSTM autoencoder: 11.5%) |
| 3. Session drift | C | Retraining on recent sessions | EER 9.3% → 6.5% (static model: 9.3% → 13.1%) |

- B reproduces 7 of the paper's 10 detectors to within 0.1 EER points, so the evaluation matches the published protocol.
- C's scaled-Manhattan check (9.53%) matches B's (9.5%), so classical and deep numbers are directly comparable.
- Typing drifts: by session 8 users type 27% faster than in session 1.

All numbers are in [`results.csv`](results.csv) (`model, task, metric, value`). EER values are stored as fractions (0.078 = 7.8%).

## Split (frozen in [`split.txt`](split.txt))

- **Train:** sessions 1-4 (200 reps per user). **Test:** sessions 5-8 (200 reps per user).
- **Task 2 impostors:** first 5 reps of session 1 from each of the other 50 users (250 per user), as in the paper.
- Splitting by session in time order (not at random) avoids leakage between near-identical samples from the same day.

### Extra experiment: more training sessions (Task 1 only)

[`experiments/task1_more_training_sessions/`](experiments/task1_more_training_sessions/) reruns A's notebook unchanged
with train = sessions 1-6 and test = sessions 7-8. This is an extra finding, not the main result.

| Model | Train 1-4, test 5-8 (main) | Train 1-4, scored on 7-8 only | Train 1-6, test 7-8 |
|---|---|---|---|
| 1D-CNN | 85.7% | 82.8% | **89.8%** |
| MLP | 84.2% | 79.6% | 88.6% |
| Random Forest | 81.8% | 76.6% | 87.0% |
| XGBoost | 80.5% | 75.4% | 86.8% |
| BiLSTM | 74.9% | 70.8% | 84.5% |

On the same test sessions, two more (and more recent) training sessions add 7-14 points. This agrees with the drift
results: recent typing matters. The "scored on 7-8 only" column comes from a rerun of the main split on a different
machine, where the 1D-CNN scored 86.6% overall instead of 85.7% (small differences across library versions are normal).

## Repository layout

```
metrics.py                  shared: EER, results logging, split loader (frozen)
split.txt                   shared: train/test rule (frozen)
ae_models.py                Person C: autoencoders + UserDetector used by the demo
data/                       CMU dataset (CSV)
notebooks/
  task1_identification.ipynb      Person A: EDA, LR, SVM, RF, XGBoost, MLP, 1D-CNN, BiLSTM
  task2_classical.ipynb           Person B: 12 detectors, EER vs the paper
  task2_deep_task3_drift.ipynb    Person C: dense/LSTM autoencoders, session drift
results.csv                 all results from the three notebooks
results_task2_classical_per_user.csv   Person B: per-user EER
models/task2_best_classical.joblib     Person B: kNN detector + per-user thresholds
artifacts/                  Person C: 51 saved LSTM-AE detectors + per-user EER tables
report/figures/             Person C: figures for the report
app/                        Streamlit demo (app.py shell + tab_c.py)
docs/                       project plan and Person B / C handoff notes
experiments/                extra Task 1 experiment (see above)
```

## Running

```bash
pip install -r requirements.txt
```

- **Notebooks:** open from `notebooks/`; each reads `../data`, `../metrics.py`, `../split.txt` and writes its own rows
  to `../results.csv`. Re-running a notebook replaces only that notebook's rows.
- **Demo:** from the repo root, `streamlit run app/app.py`. Person C's tab is live. Person A's and B's tabs are
  placeholders. C's tab also runs alone with `streamlit run app/tab_c.py`.
- Person A's saved model (`models/task1_best_classical.joblib`, Random Forest) is not committed: it is about 780 MB,
  over GitHub's 100 MB file limit. Run the Task 1 notebook to recreate it.
