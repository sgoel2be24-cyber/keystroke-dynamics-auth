# Person C Handoff: Task 2 (Deep Authentication) + Task 3 (Session Drift)

**Owner:** Person C · **Notebook:** `notebooks/task2_deep_task3_drift.ipynb` · **Task names in results.csv:** `task2_deep`, `task3_drift`

---

## 1. Contract: no change to `split.txt`

C uses `split.txt` exactly as B recreated it (train = sessions 1-4, test = sessions 5-8, `IMPOSTOR_SESSION=1`, `IMPOSTOR_PER_USER=5`, seed 42). The notebook asserts the shapes per user (200 train, 200 genuine test, 250 impostors), so it fails loudly if the contract changes.

C's impostor set is the same as B's: the first 5 reps of session 1 for each of the other 50 users.

**Cross-check against B:** C's scaled-Manhattan reference gives **9.53%** mean EER, B's gives 9.5% and the paper 9.6%. The two harnesses agree, so classical and deep numbers are directly comparable.

## 2. Results (mean EER over 51 users, %)

| Detector | EER | Std | Median |
|---|---|---|---|
| Dense autoencoder | 10.0 | 6.3 | 9.4 |
| LSTM autoencoder | 11.5 | 6.5 | 10.6 |
| *Scaled Manhattan (harness check only)* | 9.5 | 7.0 | 7.6 |

For reference, B's best classical detectors are kNN at 7.8%, Isolation Forest at 8.4% and one-class SVM at 9.1%. The deep models do not reach them.

- **Dense AE vs scaled Manhattan:** statistically tied (paired Wilcoxon p = 0.25; dense AE better for 21 of 51 users).
- **LSTM-AE vs scaled Manhattan:** significantly worse (p = 2.8e-05; better for only 13 of 51 users).
- **Operating point at the demo threshold** (95th percentile of validation error from the training sessions): LSTM-AE has FAR 24.9% and FRR 8.5%; dense AE has FAR 20.3% and FRR 8.2%. This threshold is lenient. It lets many impostors through, and the demo exposes a strictness slider for exactly this reason.

Training recipe: one small model per user on genuine data only, scored by reconstruction error. Dense AE is 33→32→8→32→33. LSTM-AE has hidden size 16 and an 8-d bottleneck. Epochs were chosen by early stopping on session 4 (stage 1), then the model was retrained on all 200 reps for that many epochs (stage 2). Hyperparameters came from a 7-user pilot on training-session validation data only; every config was within noise (11-12% EER). Test sessions were not used for tuning.

## 3. Findings the others can use

- **Typing drifts a lot:** by session 8 people type **27% faster** (0.727x their session-1 typing time), hold-time features have moved about 1.0 std from their own session 1, and **71%** (1117 of 1581) of (user, feature) pairs differ significantly between sessions 1 and 8 (KS test, p < 0.01).
- **Drift hurts static detectors** (trained on sessions 1-2, tested on sessions 3-8):

  | Detector | Session 3 | Session 8 | Slope (pts/session) |
  |---|---|---|---|
  | Scaled Manhattan, static | 9.3% | 13.1% | +0.79 |
  | LSTM-AE, static | 12.5% | 13.6% | +0.19 |
  | Scaled Manhattan, sliding retrain | 9.3% | 6.5% | -0.52 |
  | LSTM-AE, sliding retrain | 12.5% | 9.4% | -0.64 |

  *Sliding* means retraining on the two most recent sessions before each test. This agrees with B's finding that recency beats quantity, and with B's per-session rise of about 2 points from test session 5 to 8 for the good detectors.
- **Caveat for the sliding result:** impostor reps are always from session 1 (frozen protocol), so a model retrained on a user's recent, faster typing is automatically further from them. Read *sliding* as "drift is recoverable", not as an absolute accuracy gain. EER at each drift point uses only 50 genuine samples, so read trends, not single points.
- **For the demo:** at the default threshold some users are blocked often. User 2's real typing is rejected about 30% of the time on sessions 5-8, because their typing drifted. This is a good talking point, and the strictness slider shows the trade-off.
- **Hardest users (for A and B):** by LSTM-AE EER: 40, 32, 47, 2, 31, 3, 34, 41 (18-26%). By Manhattan: 47, 32, 31, 2, 35, 16, 40, 34. Users 47, 32, 2 and 31 appear in B's hardest list as well, so these users are hard for every detector family.

## 4. Files

- `ae_models.py` (repo root): `DenseAE`, `LSTMAE`, the training loop and `UserDetector` (loads a saved model; `.score()`, `.error()`, `.is_genuine()`). Imported by the notebook and the demo.
- `notebooks/task2_deep_task3_drift.ipynb`: about 6 min for Task 2 plus about 8 min for drift on CPU. `KD_FAST=1` runs 4 users in about 1 min. Re-running replaces only rows whose task is `task2_deep` or `task3_drift`.
- `results.csv`: 41 rows for C (13 `task2_deep`, 28 `task3_drift`). The `ref_manhattan_scaled` rows are a sanity check; **B's rows are the official classical numbers**.
- `artifacts/lstm_ae/user_<id>.pt`: 51 saved LSTM-AE detectors (about 1 MB total). Each contains the scaler, network weights and threshold, and loads with `UserDetector.load(path)`. Torch tensors and plain Python types only.
- `artifacts/task2_deep_per_user_eer.csv` and `artifacts/task3_drift_per_user_eer.csv`: per-user results.
- `report/figures/`: `task2_deep_results.png`, `task3_feature_drift.png`, `task3_drift_curves.png`.
- `app/tab_c.py`: the Streamlit tab, with three sub-tabs: *Verify a typing sample* (claimed user, genuine or impostor sample, accept/alert verdict, typing-rhythm timeline, per-keystroke error, strictness slider with live FAR/FRR), *Model results (51 users)* and *Session drift*. Run standalone with `streamlit run app/tab_c.py`.
- **Reproducibility:** a re-run on a second machine matched the first to within 0.03 points (dense AE 10.01 vs 10.04). That is normal variation across torch versions.

**Plugging into the shared shell:**
```python
import tab_c
with tab3:          # whichever tab container app.py gives C
    tab_c.render()
```
`st.set_page_config` is called only when `tab_c.py` runs standalone, so the shell keeps control of the page. Needs `torch` and `streamlit` installed, and the `artifacts/` folder committed.

## 5. C's remaining work

- [x] Notebook run and verified (second machine reproduces the numbers)
- [x] `app/tab_c.py` built and checked in the browser
- [ ] Plug `tab_c.render()` into the shared `app.py`
- [ ] Report section C (methods, Task 2 results table vs published and B's numbers, drift analysis, limitations)
- [ ] Review A's and B's sections
- [ ] Optional: live typing mode (browser `keydown`/`keyup` timestamps scored against the claimed user)
- [ ] Optional stretch: Siamese network for few-shot verification

**Viva points to prepare:** why the paper's standard NN gets 82.8% EER while a regularised autoencoder trained on genuine data only does not; why the LSTM adds nothing over a dense AE on 11 keystrokes and 200 samples; how EER differs from accuracy; what the impostor-from-session-1 caveat means for the drift result.
