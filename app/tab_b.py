"""Person B's tab: Task 2, classical authentication. Is this really the claimed user, or an impostor?

Detector: B's best detector, kNN with scaled-Manhattan distance (k = 1), loaded from models/task2_best_classical.joblib:
each user's enrolment reps (sessions 1-4) and a per-user threshold chosen on B's validation protocol.
"""
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from common import (BLUE, ORANGE, ROOT, claim_and_sample, eer, load_data, load_results, pct, protocol_sets,
                    rhythm_chart, score_hist, show_chart, show_table, user_label)

MODEL = os.path.join(ROOT, "models", "task2_best_classical.joblib")

# Killourhy & Maxion (2009), Table 2: mean EER over 51 users.
PUBLISHED = {"Manhattan (scaled)": 0.096, "Nearest neighbor (Mahalanobis)": 0.100, "Outlier count (z-score)": 0.102,
             "One-class SVM": 0.102, "Mahalanobis": 0.110, "Manhattan (filtered)": 0.136, "Manhattan": 0.153,
             "Euclidean": 0.171, "Euclidean (normed)": 0.215, "k-means": 0.372}


@st.cache_resource
def _model():
    return joblib.load(MODEL)


def _distance_fn(enrol):
    """kNN (k = 1), scaled Manhattan: distance to the closest enrolment rep, each feature divided by its
    mean absolute deviation. Same as make_knn_scaled in B's notebook (score = -distance)."""
    a = np.abs(enrol - enrol.mean(0)).mean(0)
    T = enrol / a
    return lambda Z: np.abs((np.atleast_2d(Z) / a)[:, None, :] - T[None, :, :]).sum(2).min(1)


@st.cache_data
def _user_distances(user):
    df, feat_cols = load_data()
    art = _model()
    _, genuine, impostors = protocol_sets(df, feat_cols, user)
    dist = _distance_fn(np.asarray(art["enrol"][user]))
    return dist(genuine), dist(impostors)


def _verify_section(df, feat_cols, art):
    st.subheader("Try a login")
    users = sorted(int(u) for u in art["enrol"])
    claimed, row, is_genuine = claim_and_sample("b", df, users)
    x = row[art["feat_cols"]].values.astype(float)
    dist = _distance_fn(np.asarray(art["enrol"][claimed]))
    d = float(dist(x)[0])
    default_cut = -float(art["thresholds"][claimed])        # B stores score thresholds; score = -distance

    gen, imp = _user_distances(claimed)
    lo, hi = float(np.percentile(gen, 1)), float(np.percentile(imp, 99))
    cut = st.slider("Acceptance limit (accept if distance ≤ limit). Move left to be stricter.",
                    min_value=round(min(lo, default_cut), 1), max_value=round(max(hi, default_cut), 1),
                    value=round(default_cut, 1), step=0.1, format="%.1f", key=f"b_cut_{claimed}")
    accepted = d <= cut

    if accepted and is_genuine:
        st.success(f"✅ Accepted: the real {user_label(claimed)} is let in.")
    elif not accepted and not is_genuine:
        st.success(f"🛡️ Impostor caught: {user_label(int(row.subject))} pretending to be {user_label(claimed)} is rejected.")
    elif accepted:
        st.error(f"⚠️ False accept: impostor {user_label(int(row.subject))} got in as {user_label(claimed)}.")
    else:
        st.warning(f"⛔ False reject: the real {user_label(claimed)} was locked out.")

    far, frr = float((imp <= cut).mean()), float((gen > cut).mean())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Distance of this sample", f"{d:.1f}")
    m2.metric("Acceptance limit", f"{cut:.1f}")
    m3.metric("False accepts at this limit", pct(far), help="Share of this user's 250 impostor attempts let in")
    m4.metric("False rejects at this limit", pct(frr), help="Share of this user's 200 real test attempts blocked")
    st.caption(f"{user_label(claimed)}'s EER with this detector: {pct(eer(-gen, -imp))}. "
               "The default limit comes from B's validation run (sessions 1-3), never from the test data.")

    c1, c2 = st.columns(2)
    with c1:
        show_chart(score_hist(np.log10(gen), np.log10(imp), float(np.log10(cut)), float(np.log10(d)),
                              "log10 distance to the closest enrolment sample (lower = more like the user)", "Limit").properties(title=f"{user_label(claimed)}: real attempts vs impostors"))
    with c2:
        show_chart(rhythm_chart(row, feat_cols, title=f"Typing rhythm: {user_label(int(row.subject))}, "
                                                       f"session {int(row.sessionIndex)}, rep {int(row.rep)}"))


def _results_section():
    st.subheader("All 12 detectors vs the published paper")
    res = load_results()
    t = res[(res.task == "task2_classical") & (res.metric == "eer_mean")][["model", "value"]].rename(columns={"value": "ours"})
    t["paper"] = t.model.map(PUBLISHED)
    long = t.melt(id_vars="model", value_vars=["ours", "paper"], var_name="source", value_name="eer").dropna()
    long["source"] = long.source.map({"ours": "Our result", "paper": "Published (2009)"})
    order = t.sort_values("ours").model.tolist()
    bars = alt.Chart(long, title="Mean EER over 51 users (lower is better)").mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("eer:Q", title="equal error rate", axis=alt.Axis(format="%")),
        y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("source:N", sort=["Our result", "Published (2009)"]),
        color=alt.Color("source:N", title=None, scale=alt.Scale(domain=["Our result", "Published (2009)"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("model:N", title="Detector"), alt.Tooltip("source:N", title=""),
                 alt.Tooltip("eer:Q", title="EER", format=".1%")])
    show_chart(bars.properties(height=520))
    st.caption("Isolation Forest and kNN are new (not in the paper). 7 of the paper's 10 detectors match the published "
               "EER within 0.1 points, which shows the evaluation protocol is implemented correctly.")
    with st.expander("Table view"):
        tv = t.sort_values("ours").assign(ours=lambda d: d.ours.map(pct),
                                           paper=lambda d: d.paper.map(lambda v: "new" if pd.isna(v) else pct(v)))
        show_table(tv.rename(columns={"model": "Detector", "ours": "Our EER", "paper": "Published EER"}))


def render():
    st.header("Task 2 · Is this really you? (classical detectors)")
    st.markdown("**Authentication:** each user gets their own detector, trained only on their own typing "
                "(sessions 1-4), because a real system never has impostor samples in advance. "
                "This tab uses B's best detector: **kNN with scaled-Manhattan distance** (7.8% EER).")
    if not os.path.exists(MODEL):
        st.warning("Model not found: run notebooks/task2_classical.ipynb to create models/task2_best_classical.joblib.")
        return
    df, feat_cols = load_data()
    _verify_section(df, feat_cols, _model())
    st.divider()
    _results_section()
