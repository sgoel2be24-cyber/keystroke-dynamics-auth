"""Shared contract file: written once, imported by everyone, never edited.

results.csv columns: model, task, metric, value
Score convention for EER: HIGHER score = more likely genuine.
"""
import csv
import os

import numpy as np
from sklearn.metrics import roc_curve

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(ROOT, "results.csv")
SPLIT_PATH = os.path.join(ROOT, "split.txt")
FIELDS = ["model", "task", "metric", "value"]


def load_split(path=SPLIT_PATH):
    cfg = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k] = [int(x) for x in v.split(",")] if "," in v else int(v)
    return cfg


def eer(genuine_scores, impostor_scores):
    """Equal Error Rate where false-accept rate == false-reject rate."""
    y = np.r_[np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))]
    s = np.r_[genuine_scores, impostor_scores]
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def append_result(model, task, metric, value, path=RESULTS_PATH):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"model": model, "task": task, "metric": metric, "value": round(float(value), 6)})
