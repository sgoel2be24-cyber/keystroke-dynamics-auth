"""
Keystroke dynamics demo: one tab per team member.
Run from the repo root:  streamlit run app/app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tab_a  # noqa: E402  (Person A: identification)
import tab_b  # noqa: E402  (Person B: classical authentication)
import tab_c  # noqa: E402  (Person C: deep authentication + session drift)

st.set_page_config(page_title="Keystroke Dynamics Demo", page_icon="⌨️", layout="wide")
st.title("⌨️ Keystroke Dynamics: Behavioral Biometric Authentication")
st.markdown("Can we recognise people by **how** they type? 51 people each typed the password `.tie5Roanl` "
            "400 times over 8 sessions ([CMU benchmark](https://www.cs.cmu.edu/~keystroke/), Killourhy & Maxion 2009). "
            "Every model trains on sessions 1-4 and is tested on sessions 5-8.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Identification accuracy", "85.7%", help="Task 1, 1D-CNN, 51 users (notebook result)")
m2.metric("Best authentication EER", "7.8%", "-1.8 pts vs paper's best (9.6%)", delta_color="inverse",
          help="Task 2, kNN with scaled-Manhattan distance; lower is better")
m3.metric("Deep authentication EER", "10.0%", help="Task 2, dense autoencoder (LSTM autoencoder: 11.5%)")
m4.metric("Faster typing by session 8", "27%", help="Task 3, mean typing time vs each user's session 1")

tab1, tab2, tab3 = st.tabs(["Task 1: Identification (A)",
                            "Task 2: Classical Auth (B)",
                            "Task 2/3: Deep Auth + Drift (C)"])
with tab1:
    tab_a.render()
with tab2:
    tab_b.render()
with tab3:
    tab_c.render()
