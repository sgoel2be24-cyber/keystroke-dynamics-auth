import copy
import numpy as np
import torch
import torch.nn as nn

# CSV feature order is H, DD, UD, H, DD, UD, ... (31 columns, 11 keystrokes).
H_IDX = list(range(0, 31, 3))     # 11 hold times
DD_IDX = list(range(1, 30, 3))    # 10 down-down
UD_IDX = list(range(2, 30, 3))    # 10 up-down

# The last keystroke (Return) has no DD/UD, so those 2 slots are padding and ignored in the loss.
MASK = np.ones((11, 3), dtype=np.float32)
MASK[10, 1:] = 0.0


def to_seq(X):
    """(N, 31) scaled features -> (N, 11, 3) float32 sequence of [H, DD, UD] per keystroke."""
    X = np.asarray(X, dtype=np.float32)
    S = np.zeros((len(X), 11, 3), dtype=np.float32)
    S[:, :, 0] = X[:, H_IDX]    
    S[:, :10, 1] = X[:, DD_IDX]
    S[:, :10, 2] = X[:, UD_IDX]
    return S


class DenseAE(nn.Module):
    def __init__(self, hidden=32, z=8, p=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(33, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, z))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(z, hidden), nn.ReLU(), nn.Linear(hidden, 33))

    def forward(self, x):
        return self.dec(self.enc(x.flatten(1))).view(-1, 11, 3)


class LSTMAE(nn.Module):
    def __init__(self, hidden=16, z=8, p=0.1):
        super().__init__()
        self.enc = nn.LSTM(3, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, z)
        self.from_z = nn.Linear(z, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 3)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(self.drop(h[-1]))
        d = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)
        o, _ = self.dec(d)
        return self.out(o)


ARCHS = {"dense_ae": DenseAE, "lstm_ae": LSTMAE}
_MASK_T = torch.from_numpy(MASK)


def masked_mse(pred, x):
    """Per-sample mean squared error over the real (unmasked) entries."""
    return (((pred - x) ** 2) * _MASK_T).sum(dim=(1, 2)) / _MASK_T.sum()


def fit_ae(make, S_fit, S_val=None, epochs=300, lr=3e-3, wd=1e-4, patience=40, seed=0):
    """Full-batch Adam. If S_val is given: early stopping, best-epoch weights restored.
    Returns (model, best_epoch). With S_val=None trains exactly `epochs` epochs."""
    torch.manual_seed(seed)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    xf = torch.from_numpy(S_fit)
    xv = None if S_val is None else torch.from_numpy(S_val)
    best_v, best_state, best_ep, since = float("inf"), None, epochs, 0
    for ep in range(1, epochs + 1):
        m.train()
        opt.zero_grad()
        masked_mse(m(xf), xf).mean().backward()
        opt.step()
        if xv is not None:
            m.eval()
            with torch.no_grad():
                v = masked_mse(m(xv), xv).mean().item()
            if v < best_v - 1e-5:
                best_v, best_state, best_ep, since = v, copy.deepcopy(m.state_dict()), ep, 0
            else:
                since += 1
                if since >= patience:
                    break
    if xv is not None and best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, best_ep


@torch.no_grad()
def recon_err(m, S):
    m.eval()
    x = torch.from_numpy(S)
    return masked_mse(m(x), x).numpy()


class UserDetector:
    """One trained per-user detector: scaler + autoencoder + threshold. Used by the demo."""

    def __init__(self, arch, kwargs, state, mean, scale, threshold, user=None, best_epoch=None):
        self.arch, self.kwargs, self.user, self.best_epoch = arch, kwargs, user, best_epoch
        self.mean, self.scale, self.threshold = np.asarray(mean), np.asarray(scale), float(threshold)
        self.model = ARCHS[arch](**kwargs)
        self.model.load_state_dict(state)
        self.model.eval()

    def error(self, X31):
        X = (np.atleast_2d(X31) - self.mean) / self.scale
        return recon_err(self.model, to_seq(X))

    def score(self, X31):          # higher = more genuine (metrics.eer convention)
        return -self.error(X31)

    def is_genuine(self, X31):
        return self.error(X31) <= self.threshold

    def save(self, path):
        torch.save({"arch": self.arch, "kwargs": self.kwargs, "state": self.model.state_dict(),
                    "mean": torch.tensor(self.mean), "scale": torch.tensor(self.scale),
                    "threshold": self.threshold, "best_epoch": int(self.best_epoch),
                    "user": self.user.item() if hasattr(self.user, "item") else self.user}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        return cls(d["arch"], d["kwargs"], d["state"], d["mean"].numpy(), d["scale"].numpy(),
                   d["threshold"], d["user"], d["best_epoch"])
