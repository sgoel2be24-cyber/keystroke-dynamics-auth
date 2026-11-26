# Person B Handoff: Task 2 (Classical Authentication)

**Owner:** Person B · **Notebook:** `notebooks/task2_classical.ipynb` · **Task names in results.csv:** `task2_classical`, `task2_classical_practiced_impostors`, `task2_classical_by_session`

---

## 1. Contract change to agree on: `split.txt`

`split.txt` was never shared, so B recreated it from the values A's notebook printed (train = sessions 1-4, test = sessions 5-8, seed 42), plus the Task 2 numbers. It adds one key:

```
IMPOSTOR_SESSION=1
```

The paper's impostors are each other user's **first 5 reps, from session 1**. The sample loop in A's handoff (section 4) takes them from **session 5** instead. That makes every EER about 5 points worse and stops the numbers matching the published table (Manhattan (scaled): 9.5% with session 1 impostors, 14.6% with session 5). Both B and C must use the same impostor set or the classical-vs-deep comparison breaks.

**C:** build your impostor set like this:
```python
others = df[(df.subject != u) & (df.sessionIndex == CFG["IMPOSTOR_SESSION"])].sort_values(["subject", "rep"])
imp = others.groupby("subject").head(CFG["IMPOSTOR_PER_USER"])      # 50 users x 5 = 250 impostors
```

## 2. Results (mean EER over 51 users, %)

| Detector | EER | Std | Published (2009) |
|---|---|---|---|
| kNN, scaled Manhattan, k=1 *(new)* | **7.8** | 5.2 | — |
| Isolation Forest *(new)* | 8.4 ± 0.2 over 4 seeds (8.1 at seed 42) | 6.3 | — |
| One-class SVM (γ tuned on validation) | 9.1 | 6.3 | 10.2 |
| Manhattan (scaled) | 9.5 | 7.0 | 9.6 |
| Nearest neighbor (Mahalanobis) | 10.0 | 6.3 | 10.0 |
| Outlier count (z-score) | 10.1 | 7.7 | 10.2 |
| Mahalanobis | 11.0 | 6.4 | 11.0 |
| Manhattan (filtered) | 12.8 | 8.0 | 13.6 |
| Manhattan | 15.3 | 9.2 | 15.3 |
| k-means | 15.6 | 7.3 | 37.2 |
| Euclidean | 17.1 | 9.4 | 17.1 |
| Euclidean (normed) | 21.5 | 11.8 | 21.5 |

7 of the paper's 10 detectors reproduce the published EER within 0.1 points. kNN beats the paper's best for 39 of 51 users (Wilcoxon p = 0.0002). Hyperparameters were chosen only on a validation protocol inside sessions 1-4.

## 3. Findings the others can use

- **Practised impostors** (session 5 instead of session 1): every detector gets 4.4-7.8 points worse. kNN goes 7.8% → 12.3%.
- **Drift (for C):** EER rises from test session 5 to 8 for the good detectors, by about 2 points (kNN +1.0, Isolation Forest +2.4). Per-session rows are in `task2_classical_by_session`.
- **Recency beats quantity:** training on the last 25-50 reps (session 4) beats training on all 200 (outlier count 10.1% → 6.4%; Manhattan (scaled) 9.5% → 7.8%). This was confirmed on validation without test data. It is a future-work recommendation (adaptive templates), not a change to the protocol.
- **Features:** the 21 independent features (H + UD) do as well as all 31. Hold-only or DD-only features are far worse.
- **Hardest users (for A):** 47, 32, 2 and 31 have about 23-25% EER. 5 of A's 8 hardest-to-identify users are also among B's 8 hardest to authenticate (2, 16, 31, 32, 47).

## 4. Files

- `results.csv`: 96 rows for B. Re-running the notebook replaces only rows whose task starts with `task2_classical`.
- `results_task2_classical_per_user.csv`: per-user EER for all 12 detectors.
- `models/task2_best_classical.joblib`: kNN (scaled Manhattan), each user's sessions 1-4 enrolment reps, and per-user thresholds from the validation protocol. It holds only numpy arrays and dicts, so it loads regardless of the scikit-learn version.

## 5. B's remaining work

- [ ] Agree on `split.txt` with A and C (section 1)
- [ ] `app/tab_authentication.py`: claimed user + sample → accept/reject (snippet in the notebook's last cell)
- [ ] Report section B
- [ ] Review A's and C's sections
