# Keystroke Dynamics — Behavioral Biometric Authentication

**Project type:** ML/DL semester project (team of 3)
**Timeline:** 12 weeks
**Dataset:** CMU Keystroke Dynamics Benchmark (Killourhy & Maxion, DSN-2009)

---

## 1. Core Idea

Every person types with a unique rhythm — how long they hold each key down, how fast they move between specific key pairs, how consistently they pause. This is **keystroke dynamics**, a behavioral biometric (like a fingerprint, but based on behavior instead of biology).

The project builds an ML system that answers two questions:

1. **Identification** — Given a typing sample, which of 51 known users typed it?
2. **Authentication / Impostor detection** — Is this person really who they claim to be, or an impostor?

This maps directly to real-world use (banks, typing-based continuous MFA), which makes for a strong real-world-relevance section in the report/viva.

---

## 2. Dataset

**Source:** CMU Keystroke Dynamics Benchmark — `cs.cmu.edu/~keystroke` (also mirrored on Kaggle/GitHub).

**Structure:**
- 51 subjects, each typed the same 10-character password `.tie5Roanl` 400 times
  - 8 sessions × 50 reps per session, sessions at least 1 day apart (captures day-to-day variation)
- ~20,400 total samples, each row = 31 timing features

**Feature types:**

| Feature type | Meaning |
|---|---|
| `H.key` | Hold time — press to release of a key |
| `DD.key1.key2` | Down-Down — press of key1 to press of key2 |
| `UD.key1.key2` | Up-Down — release of key1 to press of key2 (can be negative = key overlap) |

**Why this dataset is a good pick:**
- Clean, no missing values, fully tabular, small (~2MB) — runs entirely on Colab/local CPU
- Comes with a published baseline table of 14 anomaly detectors, so results can be directly benchmarked against published research numbers

---

## 3. Problem Formulations

Structure the project as three tasks of increasing difficulty (natural report arc: warm-up → core → contribution).

### Task 1 — Multi-class Identification (start here)
- **Input:** 31 timing features → **Output:** which of 51 users (51-class classification)
- **Models to try, in order:** Logistic Regression → Random Forest / XGBoost → MLP → LSTM / 1D-CNN on the keystroke sequence
- **Expected result:** XGBoost ~95%+ accuracy — this is the confidence-building task

### Task 2 — One-vs-Rest Authentication (the paper's core setup)
- For each user: train a detector on their first 200 samples; test on their last 200 (genuine) + 5 samples from each of the other 50 users (impostors)
- **Metric:** Equal Error Rate (EER) from ROC curves — the point where false-accept rate = false-reject rate
- **Published benchmark EERs to compare against:**
  - Manhattan (scaled): 9.6%
  - Mahalanobis NN: 10.0%
  - One-class SVM: 10.3%
  - Standard Neural Net: 82.8% (badly tuned NNs fail here — good discussion material)
- **To implement:** classical detectors + a DL approach (e.g., LSTM autoencoder trained on genuine typing — high reconstruction error flags an impostor)

### Task 3 — Novelty / Extension (pick one — this is the project's "contribution")
Options:
1. **Siamese network** for few-shot verification ("authenticate from just 5 typing samples")
2. **Session drift analysis** — does accuracy decay across the 8 sessions? (tests whether typing biometrics are stable over time)
3. **Free-text generalization** — apply the trained approach to a second keystroke dataset and study transfer

---

## 4. Technical Pipeline

```
Raw CSV → Feature Engineering → Baselines → DL Models → Evaluation → Streamlit Demo
```

### Feature engineering ideas
- Derived features: total typing time, mean/std hold time, typing speed (chars/sec), digraph latency ratios
- Sequence view: reshape 31 features into a sequence of 11 keystrokes × 3 timing values → feeds LSTM
- Normalization: per-user vs. global (worth an ablation study)

### Model stack

| Stage | Models |
|---|---|
| Baselines | Logistic Regression, SVM, k-NN |
| Strong classical | Random Forest, XGBoost |
| Anomaly detectors | Mahalanobis distance, One-class SVM, Isolation Forest |
| Deep learning | MLP, 1D-CNN, BiLSTM, LSTM-autoencoder |
| Stretch | Siamese network (contrastive/triplet loss) |

### Evaluation
- **Task 1:** accuracy / F1
- **Task 2:** ROC curves + EER — replicate the paper's evaluation protocol (a reference R script exists and can be ported to Python; the port itself is a nice deliverable)
- **Task 3:** cross-session analysis (or equivalent, depending on chosen extension)

---

## 5. Team Work Split (3 people, 12 weeks)

| Member | Role | Responsibilities |
|---|---|---|
| Member 1 | Data + Baselines | Data pipeline, EDA notebook, feature engineering, classical ML models, paper's protocol replication |
| Member 2 | Deep Learning | MLP/CNN/LSTM implementation, training loops, hyperparameter tuning, Siamese network |
| Member 3 | Eval + Demo | Metrics/ROC/EER framework, error analysis, ablations, Streamlit live demo, report integration |

Note: everyone should still review each other's work — vivas often ask any member any question, regardless of their assigned role.

---

## 6. Demo

A Streamlit app with two modes:

1. **Playback mode:** select a sample from the dataset → visualize the typing rhythm (hold/flight times as a timeline chart) → model predicts the user + confidence
2. **Live mode (stretch):** a textbox using JavaScript `keydown`/`keyup` timestamps captures live typing of `.tie5Roanl` → model scores whether it matches a claimed identity → live "impostor alert"

Even a half-working live mode is memorable — demos tend to stick with evaluators more than accuracy tables do.

---

## 7. Why This Project Scores Well

- Unique — behavioral biometrics is uncommon territory for undergrad projects
- Authentic — real research dataset with published baselines to benchmark against
- Tabular-first with a genuine deep learning component
- Zero GPU risk — free CPU (e.g., Colab) is sufficient; no training-time meltdowns before deadline
- Rich report material — EDA on typing rhythms is visually engaging, the EER protocol adds rigor, session-drift adds a research angle
- Clear narrative — "passwords can be stolen; your typing rhythm can't" — a security story that's accessible to non-technical evaluators too

---

## 8. Honest Risks / Limitations

- Single fixed password in the dataset — note this as a limitation, propose free-text typing as future work
- 51-class identification (Task 1) is almost "too easy" — Task 2 (authentication protocol) must be the core of the project; Task 1 is just the warm-up
- Sequence data is short (11 keystrokes) — LSTMs will need careful regularization; worth discussing in the report
- The EER protocol is unfamiliar if the team has only worked with accuracy before — budget week 5–6 specifically for nailing this

---

## 9. Execution Plan / Next Steps

1. Set up a GitHub repo with the following structure:
   ```
   data/
   notebooks/
     01_eda.ipynb
     02_baselines.ipynb
     03_deep_models.ipynb
     04_authentication_protocol.ipynb
   app/            # Streamlit demo
   report/
   ```
2. Download and verify the dataset (`DSL-StrongPasswordData.csv`, 51 subjects × 400 samples × 31 features), run initial EDA
3. Build in order: baselines → deep learning models → authentication/EER protocol → demo, following the 12-week timeline above
