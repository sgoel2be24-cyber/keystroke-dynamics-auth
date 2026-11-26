"""Train Person A's 1D-CNN for the demo and save it to models/task1_demo_cnn.pt (about 1.5 MB).

Same split, scaler, architecture and recipe as notebooks/task1_identification.ipynb:
train on sessions 1-6 (TASK1_TRAIN_SESSIONS), pick the epoch count on session 6 (inner run on sessions 1-5),
retrain on 1-6, test once on sessions 7-8 (TASK1_TEST_SESSIONS).
The notebook's own model is not saved there, and its Random Forest file is too large for GitHub.

Run from the repo root:  python scripts/train_demo_identifier.py   (about 4 min on CPU)
"""
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from metrics import load_split          # noqa: E402
from task1_model import CNN1D, to_seq   # noqa: E402

CFG = load_split(os.path.join(ROOT, "split.txt"))
SEED = CFG["SEED"]
np.random.seed(SEED); random.seed(SEED)

df = pd.read_csv(os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv"))
feat_cols = [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]
H_idx = [i for i, c in enumerate(feat_cols) if c.startswith("H.")]
DD_idx = [i for i, c in enumerate(feat_cols) if c.startswith("DD.")]
UD_idx = [i for i, c in enumerate(feat_cols) if c.startswith("UD.")]
classes = sorted(df.subject.unique())
label = {s: i for i, s in enumerate(classes)}
train = df[df.sessionIndex.isin(CFG["TASK1_TRAIN_SESSIONS"])].reset_index(drop=True)
test = df[df.sessionIndex.isin(CFG["TASK1_TEST_SESSIONS"])].reset_index(drop=True)
ytr, yte = train.subject.map(label).values, test.subject.map(label).values

scaler = StandardScaler().fit(train[feat_cols].values)
Str = to_seq(scaler.transform(train[feat_cols].values), H_idx, DD_idx, UD_idx)
Ste = to_seq(scaler.transform(test[feat_cols].values), H_idx, DD_idx, UD_idx)


def predict(m, X):
    m.eval()
    with torch.no_grad():
        return torch.cat([m(torch.as_tensor(X[i:i + 2048])).argmax(1) for i in range(0, len(X), 2048)]).numpy()


def train_net(X, y, epochs, Xv=None, yv=None, lr=2e-3, wd=1e-3, bs=128):
    torch.manual_seed(SEED)
    m = CNN1D(len(classes))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long)
    hist = []
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        if Xv is not None:
            hist.append(accuracy_score(yv, predict(m, Xv)))
    return m, hist


t = time.time()
val = train.sessionIndex.values == max(CFG["TASK1_TRAIN_SESSIONS"])
_, hist = train_net(Str[~val], ytr[~val], 80, Str[val], ytr[val])
best = int(np.argmax(hist)) + 1
m, _ = train_net(Str, ytr, best)
pred = predict(m, Ste)
acc = accuracy_score(yte, pred)
per_session = {int(s): float((pred == yte)[test.sessionIndex.values == s].mean()) for s in CFG["TASK1_TEST_SESSIONS"]}
print(f"epochs={best}  val_acc={max(hist):.3f}  test_acc={acc:.4f}  per_session={per_session}  ({time.time() - t:.0f}s)")

out = os.path.join(ROOT, "models", "task1_demo_cnn.pt")
torch.save({"state": m.state_dict(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "classes": [int(c) for c in classes], "feat_cols": feat_cols,
            "H_idx": H_idx, "DD_idx": DD_idx, "UD_idx": UD_idx,
            "epochs": best, "test_accuracy": float(acc), "per_session_accuracy": per_session}, out)
print("saved", out)
