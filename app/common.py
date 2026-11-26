"""Shared helpers for the three demo tabs: data, split, results, sample picker and charts."""
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metrics import eer, load_split  # noqa: E402,F401  (re-exported for the tabs)

CFG = load_split(os.path.join(ROOT, "split.txt"))
DATA = os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv")

# Categorical slots 1-3 of the reference palette (validated for up to 3 series).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GENUINE, IMPOSTOR = BLUE, ORANGE
LABEL_INK = "#888888"          # value labels: readable on light and dark themes

# Demo starts on a typical user: s051 is at the median EER for both kNN (6.6%) and the LSTM autoencoder (10.4%).
DEFAULT_USER = 51

_KEY_NAMES = {"period": ".", "five": "5", "Shift.r": "Shift+R", "Return": "Return"}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA).sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]
    return df, feat_cols


def key_labels(feat_cols):
    """The 11 keys of .tie5Roanl in typing order, from the hold-time column names."""
    return [_KEY_NAMES.get(c[2:], c[2:]) for c in feat_cols if c.startswith("H.")]


@st.cache_data
def load_results():
    return pd.read_csv(os.path.join(ROOT, "results.csv"))


def user_label(u):
    return f"s{int(u):03d}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:                          # Streamlit < 1.46
        st.altair_chart(chart, use_container_width=True)


def show_table(df):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def sample_row(df, user, session, rep):
    r = df[(df.subject == user) & (df.sessionIndex == session) & (df.rep == rep)]
    return r.iloc[0]


def sample_picker(prefix, df, users, sessions, default_user=None, default_session=None, label="Typing sample from"):
    """User / session / rep selectors plus a 'random sample' button. Returns the chosen row."""
    ku, ks, kr = f"{prefix}_user", f"{prefix}_session", f"{prefix}_rep"
    rng_key = f"{prefix}_rng"
    if rng_key not in st.session_state:
        st.session_state[rng_key] = np.random.default_rng(CFG["SEED"])
    users = list(users)

    def _random():
        rng = st.session_state[rng_key]
        st.session_state[ku] = users[rng.integers(len(users))]
        st.session_state[ks] = int(rng.choice(sessions))
        st.session_state[kr] = int(rng.integers(1, 51))

    if ku not in st.session_state or st.session_state[ku] not in users:
        st.session_state[ku] = default_user if default_user in users else users[0]
    if ks not in st.session_state or st.session_state[ks] not in sessions:
        st.session_state[ks] = default_session if default_session in sessions else sessions[0]
    if kr not in st.session_state:
        st.session_state[kr] = 1

    c1, c2, c3 = st.columns([2, 1, 1])
    user = c1.selectbox(label, users, key=ku, format_func=user_label)
    session = c2.selectbox("Session", sessions, key=ks)
    rep = c3.number_input("Repetition", 1, 50, key=kr)
    st.button("🎲 Random sample", key=f"{prefix}_random", on_click=_random)
    return sample_row(df, user, session, int(rep))


def rhythm_chart(row, feat_cols, title="Typing rhythm"):
    """Timeline of one sample: one bar per key, from key-down to key-up (milliseconds)."""
    keys = key_labels(feat_cols)
    H = [row[c] for c in feat_cols if c.startswith("H.")]
    DD = [row[c] for c in feat_cols if c.startswith("DD.")]
    press = np.concatenate([[0.0], np.cumsum(DD)])
    d = pd.DataFrame({"key": keys, "order": range(11), "down": 1000 * press, "up": 1000 * (press + np.array(H)),
                      "hold": 1000 * np.array(H)})
    return (alt.Chart(d, title=title).mark_bar(color=BLUE, cornerRadius=4, height=14)
            .encode(x=alt.X("down:Q", title="time since first key press (ms)"), x2="up:Q",
                    y=alt.Y("key:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelOverlap=False)),
                    tooltip=[alt.Tooltip("key:N", title="Key"), alt.Tooltip("down:Q", title="Pressed at (ms)", format=".0f"),
                             alt.Tooltip("hold:Q", title="Held for (ms)", format=".0f")])
            .properties(height=300))


def score_hist(gen, imp, cut, sample, x_title, cut_label, bins=40):
    """Genuine vs impostor distributions of a score, with the decision cut-off and the current sample."""
    lo, hi = np.percentile(np.r_[gen, imp], [0.5, 99.5])
    lo, hi = min(lo, cut, sample), max(hi, cut, sample)
    edges = np.linspace(lo, hi, bins + 1)
    rows = []
    for name, v in (("Genuine (real user)", gen), ("Impostors", imp)):
        h, _ = np.histogram(np.clip(v, lo, hi), edges)
        rows += [dict(group=name, x0=edges[i], x1=edges[i + 1], share=h[i] / len(v)) for i in range(bins)]
    d = pd.DataFrame(rows)
    color = alt.Color("group:N", title=None, scale=alt.Scale(domain=["Genuine (real user)", "Impostors"], range=[GENUINE, IMPOSTOR]),
                      legend=alt.Legend(orient="top"))
    bars = (alt.Chart(d).mark_rect(opacity=0.7).encode(
        x=alt.X("x0:Q", title=x_title, scale=alt.Scale(zero=False, nice=False)), x2="x1:Q",
        y=alt.Y("share:Q", title="share of attempts", axis=alt.Axis(format="%")),
        y2=alt.datum(0),
        color=color, tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("share:Q", title="Share", format=".1%")]))
    marks = pd.DataFrame({"x": [cut, sample], "what": [cut_label, "This sample"]})
    rule = alt.Chart(marks.iloc[[0]]).mark_rule(strokeDash=[6, 4], strokeWidth=2, color="gray").encode(
        x="x:Q", tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("x:Q", title="Value", format=".3f")])
    point = alt.Chart(marks.iloc[[1]]).mark_rule(strokeWidth=3, color="black").encode(
        x="x:Q", tooltip=[alt.Tooltip("what:N", title=""), alt.Tooltip("x:Q", title="Value", format=".3f")])
    text = alt.Chart(marks).mark_text(align="left", dx=4, dy=-4, baseline="bottom", fontSize=11, color=LABEL_INK).encode(
        x="x:Q", y=alt.value(12), text="what:N")
    return (bars + rule + point + text).properties(height=260)


def protocol_sets(df, feat_cols, user):
    """The paper's per-user protocol (frozen in split.txt): enrol, genuine test, impostors."""
    me = df[df.subject == user]
    enrol = me[me.sessionIndex.isin(CFG["TRAIN_SESSIONS"])][feat_cols].values
    genuine = me[me.sessionIndex.isin(CFG["TEST_SESSIONS"])][feat_cols].values
    others = df[(df.subject != user) & (df.sessionIndex == CFG["IMPOSTOR_SESSION"])].sort_values(["subject", "rep"])
    impostors = others.groupby("subject").head(CFG["IMPOSTOR_PER_USER"])[feat_cols].values
    return enrol, genuine, impostors


def claim_and_sample(prefix, df, users):
    """Claimed identity + who actually types (genuine or impostor). Returns (claimed, row, is_genuine)."""
    users = list(users)
    if f"{prefix}_claimed" not in st.session_state and DEFAULT_USER in users:
        st.session_state[f"{prefix}_claimed"] = DEFAULT_USER
    c1, c2 = st.columns([1, 1])
    claimed = c1.selectbox("Claimed identity (who logs in)", users, key=f"{prefix}_claimed", format_func=user_label)
    who = c2.radio("Who actually types?", ["The real user", "An impostor"], key=f"{prefix}_who", horizontal=True)
    if who == "The real user":
        row = sample_picker(f"{prefix}_gen", df, [claimed], CFG["TEST_SESSIONS"], default_user=claimed,
                            label="Real user (test sessions only)")
    else:
        others = [u for u in users if u != claimed]
        row = sample_picker(f"{prefix}_imp", df, others, list(range(1, 9)), default_session=CFG["IMPOSTOR_SESSION"],
                            label="Impostor")
    return claimed, row, who == "The real user"
