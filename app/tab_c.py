"""Person C's tab: Task 2 (deep authentication) and Task 3 (session drift).

Detector: C's per-user LSTM autoencoder (artifacts/lstm_ae/user_<id>.pt), trained only on that user's sessions 1-4.
A sample the autoencoder cannot reconstruct well is flagged as an impostor.
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from common import (BLUE, LABEL_INK, ORANGE, ROOT, claim_and_sample, eer, key_labels, load_data, load_results, pct,
                    protocol_sets, rhythm_chart, score_hist, show_chart, show_table, user_label)
import ae_models as A

ART = os.path.join(ROOT, "artifacts")


@st.cache_resource
def _detector(user):
    return A.UserDetector.load(os.path.join(ART, "lstm_ae", f"user_{user}.pt"))


def _saved_users():
    d = os.path.join(ART, "lstm_ae")
    return sorted(int(f[5:-3]) for f in os.listdir(d) if f.startswith("user_") and f.endswith(".pt"))


@st.cache_data
def _user_errors(user):
    df, feat_cols = load_data()
    _, genuine, impostors = protocol_sets(df, feat_cols, user)
    det = _detector(user)
    return det.error(genuine), det.error(impostors)


@torch.no_grad()
def _per_key_error(det, x):
    S = A.to_seq((np.atleast_2d(x) - det.mean) / det.scale)
    rec = det.model(torch.from_numpy(S)).numpy()
    err = ((rec - S) ** 2) * A.MASK
    return (err.sum(2) / A.MASK.sum(1))[0]


def _verify_section(df, feat_cols):
    st.subheader("Try a login")
    claimed, row, is_genuine = claim_and_sample("c", df, _saved_users())
    det = _detector(claimed)
    x = row[feat_cols].values.astype(np.float32)
    err = float(det.error(x)[0])
    gen, imp = _user_errors(claimed)

    factor = st.select_slider("Strictness: acceptance limit as a multiple of the trained threshold "
                              "(lower = stricter)", options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], value=1.0,
                              key=f"c_factor_{claimed}")
    cut = det.threshold * factor
    accepted = err <= cut

    if accepted and is_genuine:
        st.success(f"✅ Accepted: the real {user_label(claimed)} is let in.")
    elif not accepted and not is_genuine:
        st.success(f"🛡️ Impostor caught: {user_label(int(row.subject))} pretending to be {user_label(claimed)} is rejected.")
    elif accepted:
        st.error(f"⚠️ False accept: impostor {user_label(int(row.subject))} got in as {user_label(claimed)}.")
    else:
        st.warning(f"⛔ False reject: the real {user_label(claimed)} was locked out.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reconstruction error", f"{err:.2f}")
    m2.metric("Acceptance limit", f"{cut:.2f}")
    m3.metric("False accepts at this limit", pct(float((imp <= cut).mean())),
              help="Share of this user's 250 impostor attempts let in")
    m4.metric("False rejects at this limit", pct(float((gen > cut).mean())),
              help="Share of this user's 200 real test attempts blocked")
    st.caption(f"{user_label(claimed)}'s EER with the LSTM autoencoder: {pct(eer(-gen, -imp))}. "
               "The trained threshold is the 95th percentile of errors on held-out training data (session 4).")

    c1, c2 = st.columns(2)
    with c1:
        show_chart(score_hist(np.log10(gen), np.log10(imp), float(np.log10(cut)), float(np.log10(err)),
                              "log10 reconstruction error (lower = more like the user)", "Limit")
                   .properties(title=f"{user_label(claimed)}: real attempts vs impostors"))
    with c2:
        pk = pd.DataFrame({"key": key_labels(feat_cols), "order": range(11), "error": _per_key_error(det, x)})
        bars = alt.Chart(pk, title="Which keys looked unusual? (error per keystroke)").mark_bar(
            color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("key:N", sort=alt.SortField("order"), title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("error:Q", title="reconstruction error"),
            tooltip=[alt.Tooltip("key:N", title="Key"), alt.Tooltip("error:Q", title="Error", format=".3f")])
        show_chart(bars.properties(height=260))
    show_chart(rhythm_chart(row, feat_cols, title=f"Typing rhythm: {user_label(int(row.subject))}, "
                                                   f"session {int(row.sessionIndex)}, rep {int(row.rep)}"))


def _results_section():
    st.subheader("Deep vs classical detectors")
    res = load_results()
    pick = [("kNN (scaled Manhattan)", "task2_classical", "Classical (B)"),
            ("Isolation Forest", "task2_classical", "Classical (B)"),
            ("One-class SVM", "task2_classical", "Classical (B)"),
            ("Manhattan (scaled)", "task2_classical", "Classical (B)"),
            ("dense_ae", "task2_deep", "Deep (C)"), ("lstm_ae", "task2_deep", "Deep (C)")]
    names = {"dense_ae": "Dense autoencoder", "lstm_ae": "LSTM autoencoder"}
    rows = []
    for model, task, family in pick:
        v = res[(res.model == model) & (res.task == task) & (res.metric == "eer_mean")].value
        rows.append(dict(model=names.get(model, model), family=family, eer=float(v.iloc[0])))
    t = pd.DataFrame(rows).sort_values("eer")
    order = t.model.tolist()
    bars = alt.Chart(t, title="Mean EER over 51 users (lower is better)").mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("eer:Q", title="equal error rate", axis=alt.Axis(format="%")),
        y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
        color=alt.Color("family:N", title=None, scale=alt.Scale(domain=["Classical (B)", "Deep (C)"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("model:N", title="Detector"), alt.Tooltip("family:N", title="Family"),
                 alt.Tooltip("eer:Q", title="EER", format=".1%")])
    labels = alt.Chart(t).mark_text(align="left", dx=4, fontSize=11, color=LABEL_INK).encode(
        x="eer:Q", y=alt.Y("model:N", sort=order), text=alt.Text("eer:Q", format=".1%"))
    show_chart((bars + labels).properties(height=300))
    st.caption("With only 200 training samples per user, simple distance-based detectors beat the deep models. "
               "The dense autoencoder is statistically tied with scaled Manhattan; the LSTM autoencoder is worse.")


def _drift_section(df, feat_cols):
    st.subheader("Task 3 · Does typing change over time?")
    DD = [c for c in feat_cols if c.startswith("DD.")]
    tt = df.assign(total=df[DD].sum(1) + df["H.Return"]).groupby(["subject", "sessionIndex"]).total.mean().unstack()
    rel = tt.div(tt[1], axis=0).mean().reset_index()
    rel.columns = ["session", "relative"]

    res = load_results()
    d = res[(res.task == "task3_drift") & res.metric.str.startswith("eer_mean_s")].copy()
    d["session"] = d.metric.str[-1].astype(int)
    d["detector"] = d.model.str.replace("_sliding", "").str.replace("_static", "").map(
        {"lstm_ae": "LSTM autoencoder", "ref_manhattan_scaled": "Scaled Manhattan"})
    d["mode"] = np.where(d.model.str.endswith("sliding"), "retrained each session", "trained once")
    d["series"] = d.detector + ", " + d["mode"]
    series = [f"{m}, {mode}" for m in ("Scaled Manhattan", "LSTM autoencoder")
              for mode in ("trained once", "retrained each session")]

    c1, c2 = st.columns(2)
    with c1:
        line = alt.Chart(rel, title="Typing time relative to each user's session 1").mark_line(
            color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=80, color=BLUE)).encode(
            x=alt.X("session:O", title="session", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("relative:Q", title="typing time (session 1 = 100%)", axis=alt.Axis(format="%"),
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("session:O", title="Session"), alt.Tooltip("relative:Q", title="Relative time", format=".1%")])
        show_chart(line.properties(height=300))
        st.caption(f"By session 8 people type the password {pct(1 - rel.relative.iloc[-1], 0)} faster than in session 1.")
    with c2:
        lines = alt.Chart(d, title="EER on each later session").mark_line(
            strokeWidth=2, point=alt.OverlayMarkDef(size=70)).encode(
            x=alt.X("session:O", title="test session", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="mean EER", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title=None, scale=alt.Scale(domain=series, range=[BLUE, BLUE, ORANGE, ORANGE]),
                            legend=alt.Legend(orient="bottom", columns=2, symbolType="stroke", labelLimit=300)),
            strokeDash=alt.StrokeDash("series:N", title=None, scale=alt.Scale(
                domain=series, range=[[6, 4], [1, 0], [6, 4], [1, 0]])),
            tooltip=[alt.Tooltip("detector:N", title="Detector"), alt.Tooltip("mode:N", title="Mode"),
                     alt.Tooltip("session:O", title="Session"), alt.Tooltip("value:Q", title="EER", format=".1%")])
        show_chart(lines.properties(height=380))
        st.caption("Trained once = trained on sessions 1-2 and never updated (dashed): it gets worse as typing drifts. "
                   "Retrained = retrained on the 2 most recent sessions before each test (solid): the error goes down instead.")
    with st.expander("Table view"):
        tv = d.pivot_table(index=["detector", "mode"], columns="session", values="value").reset_index()
        tv["mode"] = tv["mode"].str.capitalize()
        for s in range(3, 9):
            tv[s] = tv[s].map(pct)
        show_table(tv.rename(columns={"detector": "Detector", "mode": "Mode", **{s: f"Session {s}" for s in range(3, 9)}}))


def render():
    st.header("Task 2/3 · Deep authentication and session drift")
    st.markdown("Each user gets their own **LSTM autoencoder**, trained only on their sessions 1-4 to rebuild their "
                "typing. If it cannot rebuild a sample well (high reconstruction error), the sample is probably an impostor.")
    if not os.path.isdir(os.path.join(ART, "lstm_ae")):
        st.warning("Saved models not found: run notebooks/task2_deep_task3_drift.ipynb to create artifacts/lstm_ae/.")
        return
    df, feat_cols = load_data()
    _verify_section(df, feat_cols)
    st.divider()
    _results_section()
    st.divider()
    _drift_section(df, feat_cols)
