"""
Shared Streamlit demo: one tab per team member.
Run from the repo root:  streamlit run app/app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tab_c  # noqa: E402  (Person C: deep authentication + session drift)

st.set_page_config(page_title="Keystroke Dynamics: Behavioral Biometric Authentication", layout="wide")
st.title("Keystroke Dynamics: Behavioral Biometric Authentication")
st.caption("CMU Keystroke Benchmark (Killourhy & Maxion, 2009)")

tab1, tab2, tab3 = st.tabs(["Task 1: Identification (A)",
                            "Task 2: Classical Auth (B)",
                            "Task 2/3: Deep Auth + Drift (C)"])

with tab1:
    st.header("User Identification")
    st.info("Person A: sample picker + predicted user + model confidence here")

with tab2:
    st.header("Authentication: Classical Detectors")
    st.info("Person B: genuine-vs-impostor score, ROC curve, EER vs paper table here")

with tab3:
    tab_c.render()
