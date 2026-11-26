"""Person A's 1D-CNN identifier (same architecture and training recipe as notebooks/task1_identification.ipynb).

Used by scripts/train_demo_identifier.py (training) and app/tab_a.py (demo).
"""
import numpy as np
import torch
import torch.nn as nn

FEATURES = 31


def to_seq(X, H_idx, DD_idx, UD_idx):
    """(N, 31) scaled features -> (N, 11 keystrokes, 3 channels): hold, down-down, up-down. Last key has no next -> 0."""
    X = np.asarray(X, dtype=np.float32)
    S = np.zeros((len(X), 11, 3), dtype=np.float32)
    S[:, :, 0] = X[:, H_idx]
    S[:, :10, 1] = X[:, DD_idx]
    S[:, :10, 2] = X[:, UD_idx]
    return S


class CNN1D(nn.Module):
    def __init__(self, n_cls, p=0.3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(3, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                                  nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(p))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 11, 256), nn.ReLU(), nn.Dropout(p), nn.Linear(256, n_cls))

    def forward(self, x):
        return self.head(self.conv(x.transpose(1, 2)))


class Identifier:
    """Saved identifier: scaler + CNN + class list. Loads from models/task1_demo_cnn.pt."""

    def __init__(self, state, mean, scale, classes, feat_cols, H_idx, DD_idx, UD_idx, **info):
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.classes, self.feat_cols = list(classes), list(feat_cols)
        self.idx = (list(H_idx), list(DD_idx), list(UD_idx))
        self.info = info
        self.model = CNN1D(len(self.classes))
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def proba(self, X31):
        X = (np.atleast_2d(np.asarray(X31, dtype=np.float32)) - self.mean) / self.scale
        logits = self.model(torch.from_numpy(to_seq(X, *self.idx)))
        return torch.softmax(logits, 1).numpy()

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")        # tensors and plain Python types only
        return cls(**d)
