"""Person A's tab: Task 1, identification. Given one typing sample, which of the 51 users typed it?

Model: the 1D-CNN from notebooks/task1_identification.ipynb, retrained for the demo by
scripts/train_demo_identifier.py and saved to models/task1_demo_cnn.pt.
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (BLUE, CFG, DEFAULT_USER, LABEL_INK, ROOT, load_data, load_results, pct, rhythm_chart, sample_picker, show_chart,
                    show_table, user_label)
from task1_model import Identifier

MODEL = os.path.join(ROOT, "models", "task1_demo_cnn.pt")


@st.cache_resource
def _identifier():
    return Identifier.load(MODEL)


@st.cache_data
def _test_predictions():
    """Demo model's predictions on the whole test set (sessions 5-8), for the per-session chart."""
    df, feat_cols = load_data()
    ident = _identifier()
    test = df[df.sessionIndex.isin(CFG["TEST_SESSIONS"])]
    pred = np.array(ident.classes)[ident.proba(test[feat_cols].values).argmax(1)]
    return pd.DataFrame({"session": test.sessionIndex.values, "correct": pred == test.subject.values})


def _verify_section(df, feat_cols, ident):
    st.subheader("Identify a typing sample")
    st.caption(f"Pick any sample from the test sessions ({', '.join(map(str, CFG['TEST_SESSIONS']))}). "
               "The model never saw these during training.")
    row = sample_picker("a", df, ident.classes, CFG["TEST_SESSIONS"], default_user=DEFAULT_USER, label="Typed by")
    true_user = int(row.subject)
    p = ident.proba(row[feat_cols].values.astype(np.float32))[0]
    order = np.argsort(-p)
    pred = ident.classes[order[0]]
    rank = int(np.where(np.array(ident.classes)[order] == true_user)[0][0]) + 1

    if pred == true_user:
        st.success(f"✅ Correct: the model says **{user_label(pred)}** with {pct(p[order[0]])} confidence.")
    else:
        st.error(f"❌ Wrong: the model says **{user_label(pred)}** ({pct(p[order[0]])}), "
                 f"but **{user_label(true_user)}** typed it (ranked #{rank}).")

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted user", user_label(pred))
    m2.metric("Confidence", pct(p[order[0]]))
    m3.metric("True user's rank", f"#{rank} of {len(ident.classes)}")

    c1, c2 = st.columns(2)
    with c1:
        top = pd.DataFrame({"user": [user_label(ident.classes[i]) + ("  (true)" if ident.classes[i] == true_user else "")
                                     for i in order[:5]], "prob": p[order[:5]], "rank": range(5)})
        bars = alt.Chart(top, title="Top 5 guesses").mark_bar(color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("prob:Q", title="model confidence", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("user:N", sort=alt.SortField("rank"), title=None),
            tooltip=[alt.Tooltip("user:N", title="User"), alt.Tooltip("prob:Q", title="Confidence", format=".1%")])
        labels = bars.mark_text(align="left", dx=4, fontSize=11, color=LABEL_INK).encode(text=alt.Text("prob:Q", format=".0%"))
        show_chart((bars + labels).properties(height=300))
    with c2:
        show_chart(rhythm_chart(row, feat_cols, title=f"Typing rhythm: {user_label(true_user)}, "
                                                       f"session {int(row.sessionIndex)}, rep {int(row.rep)}"))


def _results_section(ident):
    st.subheader("How the models compare")
    res = load_results()
    t1 = res[(res.task == "task1_identification")].pivot(index="model", columns="metric", values="value").reset_index()
    t1 = t1.sort_values("accuracy", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        bars = alt.Chart(t1, title="Test accuracy, sessions 5-8 (notebook results)").mark_bar(
            color=BLUE, cornerRadiusEnd=4).encode(
            x=alt.X("accuracy:Q", title="accuracy", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
                     alt.Tooltip("macro_f1:Q", title="Macro F1", format=".3f")])
        labels = bars.mark_text(align="left", dx=4, fontSize=11, color=LABEL_INK).encode(text=alt.Text("accuracy:Q", format=".1%"))
        show_chart((bars + labels).properties(height=320))
    with c2:
        ps = _test_predictions().groupby("session").correct.mean().reset_index()
        line = alt.Chart(ps, title="Demo model accuracy by test session").mark_line(
            color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=80, color=BLUE)).encode(
            x=alt.X("session:O", title="test session", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("correct:Q", title="accuracy", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("session:O", title="Session"), alt.Tooltip("correct:Q", title="Accuracy", format=".1%")])
        show_chart(line.properties(height=320))
        st.caption("Accuracy falls on later sessions because people's typing drifts away from how they typed "
                   "in sessions 1-4.")
    with st.expander("Table view"):
        show_table(t1.rename(columns={"model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1"})
                   .assign(Accuracy=lambda d: d.Accuracy.map(pct), **{"Macro F1": lambda d: d["Macro F1"].round(3)}))
    st.caption(f"The demo uses the 1D-CNN, retrained with the notebook's recipe: {pct(ident.info['test_accuracy'])} on "
               "sessions 5-8 (the notebook run reported 85.7%; neural networks vary slightly between runs and library "
               "versions).")


def render():
    st.header("Task 1 · Who typed this?")
    st.markdown("**Identification:** given one typing of `.tie5Roanl`, pick which of the 51 users typed it. "
                "Models train on sessions 1-4 and are tested on sessions 5-8.")
    if not os.path.exists(MODEL):
        st.warning("Model not found. Run `python scripts/train_demo_identifier.py` from the repo root (about 3 min).")
        return
    df, feat_cols = load_data()
    ident = _identifier()
    _verify_section(df, feat_cols, ident)
    st.divider()
    _results_section(ident)
