"""Person C's Streamlit tab: deep keystroke authentication demo + session-drift analysis.

Standalone:  streamlit run app/tab_c.py
Hosted:      import tab_c; tab_c.render()      (e.g. inside `with tab_c_container:` in the shared app.py)

Needs, produced by notebooks/task2_deep_task3_drift.ipynb:
  artifacts/lstm_ae/user_<id>.pt, artifacts/task2_deep_per_user_eer.csv, artifacts/task3_drift_per_user_eer.csv,
  report/figures/*.png.  Nothing is retrained here: models are loaded and scored on demand.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch


def _find_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):                      # works from repo root or from app/
        if os.path.exists(os.path.join(d, "metrics.py")):
            return d
        d = os.path.dirname(d)
    raise FileNotFoundError("Could not find metrics.py above tab_c.py; keep tab_c.py inside the repo.")


ROOT = _find_root()
sys.path.insert(0, ROOT)
from metrics import load_split          # noqa: E402  (shared contract file)
import ae_models as A                   # noqa: E402

CFG = load_split(os.path.join(ROOT, "split.txt"))
DATA = os.environ.get("KD_DATA", os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv"))
ART = os.environ.get("KD_ARTIFACTS", os.path.join(ROOT, "artifacts"))
FIG = os.environ.get("KD_FIGURES", os.path.join(ROOT, "report", "figures"))
KEYS = [".", "t", "i", "e", "5", "Shift+r", "o", "a", "n", "l", "Return"]   # password .tie5Roanl


# ----------------------------------------------------------------------------- data / models
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA).sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]
    return df, feat_cols


@st.cache_resource(show_spinner=False)
def get_detector(u):
    return A.UserDetector.load(os.path.join(ART, "lstm_ae", f"user_{u}.pt"))


def genuine_test(df, fc, u):
    """User u's genuine test reps (sessions 5-8) with their session/rep labels."""
    g = df[(df.subject == u) & (df.sessionIndex.isin(CFG["TEST_SESSIONS"]))]
    return g[fc].values, g[["sessionIndex", "rep"]].values


def impostor_pool(df, fc, u):
    """The frozen protocol's impostor set: first IMPOSTOR_PER_USER reps of IMPOSTOR_SESSION for every other user."""
    parts = []
    for v in sorted(df.subject.unique()):
        if v == u:
            continue
        g = df[(df.subject == v) & (df.sessionIndex == CFG["IMPOSTOR_SESSION"])].head(CFG["IMPOSTOR_PER_USER"])
        parts.append(g)
    return pd.concat(parts)


@st.cache_data(show_spinner=False)
def user_errors(u):
    """Reconstruction errors for the user's 200 genuine test reps and 250 impostor reps (for live FAR/FRR)."""
    df, fc = load_data()
    det = get_detector(u)
    Xg, _ = genuine_test(df, fc, u)
    Xi = impostor_pool(df, fc, u)[fc].values
    return det.error(Xg), det.error(Xi), det.threshold


def per_key_error(det, x):
    """Squared reconstruction error summed per keystroke (11 values): where did this sample look unlike the user?"""
    S = A.to_seq((np.atleast_2d(x) - det.mean) / det.scale)
    with torch.no_grad():
        out = det.model(torch.from_numpy(S)).numpy()
    return (((out - S) ** 2) * A.MASK).sum(axis=2)[0]


# ----------------------------------------------------------------------------- plots
def press_release(x):
    """Reconstruct key press/release times (s) from H and DD features."""
    press = np.r_[0.0, np.cumsum(x[A.DD_IDX])]
    return press, press + x[A.H_IDX]


def plot_rhythm(x, typical, label):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    p, r = press_release(typical)
    for k in range(11):
        ax.barh(k, r[k] - p[k], left=p[k], fill=False, edgecolor="k", ls="--", lw=1,
                label="user's typical rhythm" if k == 0 else None)
    p, r = press_release(x)
    for k in range(11):
        ax.barh(k, r[k] - p[k], left=p[k], color="tab:orange", alpha=.75, height=.55, label=label if k == 0 else None)
    ax.set_yticks(range(11)); ax.set_yticklabels(KEYS); ax.invert_yaxis()
    ax.set_xlabel("time since first key press (s)"); ax.set_title("Typing rhythm: when each key is held down")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_key_error(err):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.bar(KEYS, err, color="tab:red", alpha=.8)
    ax.set_ylabel("error"); ax.set_title("Which keystrokes looked least like this user")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- UI sections
def _image(path, caption=None):
    """st.image that works on old and new Streamlit (use_container_width was replaced by width='stretch')."""
    try:
        st.image(path, caption=caption, width="stretch")
    except TypeError:
        st.image(path, caption=caption, use_container_width=True)


def _missing(what):
    st.warning(f"{what} not found. Run notebooks/task2_deep_task3_drift.ipynb first, then reload.")


def verify_section():
    if not os.path.isdir(os.path.join(ART, "lstm_ae")):
        return _missing("Saved models (artifacts/lstm_ae)")
    df, fc = load_data()
    saved = sorted(int(f[5:-3]) for f in os.listdir(os.path.join(ART, "lstm_ae")) if f.startswith("user_") and f.endswith(".pt"))
    if not saved:
        return _missing("Saved models")

    st.markdown("Pick who the person **claims** to be, then feed in a typing sample. The claimed user's own autoencoder "
                "tries to reconstruct it; a large reconstruction error means *this is not how that person types*.")
    c1, c2 = st.columns(2)
    claimed = c1.selectbox("Claimed identity (subject)", saved, key="tabc_claimed")
    source = c2.radio("Sample typed by", ["The real user (genuine)", "Someone else (impostor)"], horizontal=True, key="tabc_source")

    det = get_detector(claimed)
    Xg, meta = genuine_test(df, fc, claimed)
    train_typical = df[(df.subject == claimed) & df.sessionIndex.isin(CFG["TRAIN_SESSIONS"])][fc].values.mean(0)

    if source.startswith("The real"):
        i = st.slider("Genuine sample (sessions 5-8)", 1, len(Xg), 1, key="tabc_gi") - 1
        x, desc = Xg[i], f"genuine, session {meta[i][0]}, rep {meta[i][1]}"
    else:
        pool = impostor_pool(df, fc, claimed)
        who = st.selectbox("Impostor (subject)", sorted(pool.subject.unique()), key="tabc_imp")
        sub = pool[pool.subject == who]
        j = st.slider("Impostor sample", 1, len(sub), 1, key="tabc_ii") - 1
        x, desc = sub[fc].values[j], f"impostor, subject {who}, session {sub.sessionIndex.iloc[j]}, rep {sub.rep.iloc[j]}"

    g_err, i_err, base_thr = user_errors(claimed)
    mult = st.slider("Strictness: threshold multiplier (lower = stricter)", 0.3, 3.0, 1.0, 0.05, key="tabc_mult",
                     help="1.0 is the threshold learned from the user's own training data. Lower rejects more impostors but also more genuine attempts.")
    thr = base_thr * mult
    err = float(det.error(x)[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Reconstruction error", f"{err:.2f}")
    m2.metric("Threshold", f"{thr:.2f}")
    m3.metric("Error / threshold", f"{err / thr:.2f}")
    if err <= thr:
        st.success(f"ACCEPTED: this sample is consistent with subject {claimed}'s typing.  ({desc})")
    else:
        st.error(f"IMPOSTOR ALERT: this does not look like subject {claimed}'s typing.  ({desc})")

    left, right = st.columns([3, 2])
    left.pyplot(plot_rhythm(x, train_typical, "this sample"))
    right.pyplot(plot_key_error(per_key_error(det, x)))

    st.markdown("#### What this threshold costs, for this user")
    far, frr = float(np.mean(i_err <= thr)), float(np.mean(g_err > thr))
    a, b = st.columns(2)
    a.metric("False accept rate (impostors let in)", f"{far:.1%}")
    b.metric("False reject rate (real user blocked)", f"{frr:.1%}")
    ms = np.round(np.arange(0.3, 3.01, 0.1), 2)
    curve = pd.DataFrame({"False accept": [np.mean(i_err <= base_thr * m) for m in ms],
                          "False reject": [np.mean(g_err > base_thr * m) for m in ms]}, index=pd.Index(ms, name="threshold multiplier"))
    st.line_chart(curve)
    st.caption("Measured on this user's 200 genuine test reps (sessions 5-8) and the 250 frozen impostor reps. "
               "The point where the two lines cross is this user's EER.")


def results_section():
    path = os.path.join(ART, "task2_deep_per_user_eer.csv")
    if not os.path.exists(path):
        return _missing("Task 2 results")
    d = pd.read_csv(path)
    s = (d.groupby("model").eer.agg(["mean", "median", "std"]) * 100).round(2)
    s.columns = ["mean EER %", "median EER %", "std EER %"]
    st.markdown("**Equal error rate across all 51 users** (each user has their own detector; lower is better). "
                "The paper's published scaled-Manhattan result is 9.6%.")
    st.dataframe(s.loc[[m for m in ("ref_manhattan_scaled", "dense_ae", "lstm_ae") if m in s.index]])
    st.markdown("The LSTM autoencoder does **not** beat the simple scaled-Manhattan detector: with only 11 keystrokes and 200 "
                "training samples per user, there is little sequential structure for the LSTM to exploit.")
    img = os.path.join(FIG, "task2_deep_results.png")
    if os.path.exists(img):
        _image(img)


def drift_section():
    path = os.path.join(ART, "task3_drift_per_user_eer.csv")
    if not os.path.exists(path):
        return _missing("Drift results")
    d = pd.read_csv(path)
    st.markdown("**Does typing change over time?** Each detector is trained on sessions 1-2 only and tested on every later session. "
                "*Static* = never updated. *Sliding* = retrained on the two most recent sessions before each test.")
    t = (d.groupby(["model", "mode", "session"]).eer.mean().unstack("session") * 100).round(1)
    st.dataframe(t.rename_axis(columns="EER % by session"))
    line = d.groupby(["model", "mode", "session"]).eer.mean().unstack(["model", "mode"])
    line.columns = [f"{m} / {mode}" for m, mode in line.columns]
    st.line_chart(line)
    for name, cap in (("task3_feature_drift.png", "People type faster and hold keys differently as sessions pass (relative to their own session 1)."),
                      ("task3_drift_curves.png", "Static detectors degrade with time; retraining on recent sessions recovers it. "
                                                 "Caveat: impostor reps are always from session 1, so the sliding curve can look better than it truly is.")):
        p = os.path.join(FIG, name)
        if os.path.exists(p):
            _image(p, cap)


def render():
    """Entry point used by the shared app.py shell."""
    st.header("Deep authentication and session drift")
    t1, t2, t3 = st.tabs(["Verify a typing sample", "Model results (51 users)", "Session drift"])
    with t1:
        verify_section()
    with t2:
        results_section()
    with t3:
        drift_section()


if __name__ == "__main__":
    st.set_page_config(page_title="Keystroke authentication (Person C)", layout="wide")
    render()
