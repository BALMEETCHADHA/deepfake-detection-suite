"""
common/calibration.py
----------------------
Score calibration (Platt scaling) + reporting metrics (ECE, ROC-AUC).

WHY THIS EXISTS
    A classifier that outputs "FAKE 70%" is only *calibrated* if, among all
    images it scores around 70%, about 70% really are fake. Pretrained
    models are usually NOT calibrated on your data. Once you have labelled
    results (06_dataset_and_evaluation/batch_evaluate.py), you can fit a tiny
    correction  p_cal = sigmoid(a * logit(p_raw) + b)  so the percentages
    mean what they say. It also shifts the effective decision boundary: if
    the model is biased toward REAL on your data, `b` absorbs that.

HOW IT IS USED
    1. batch_evaluate.py fits (a, b) on labelled data and writes
       calibration_<modality>.json.
    2. At inference, detectors call `apply_calibration(modality, fake_pct)`.
       If no calibration file exists, scores pass through UNCHANGED.

    Calibration file lookup order:
       $DEEPFAKE_CALIBRATION_DIR/calibration_<modality>.json
       <suite root>/calibration/calibration_<modality>.json
    Set DEEPFAKE_CALIBRATION=off to disable it without deleting files.

Pure numpy - no sklearn/scipy dependency, so it works in every module.
"""

from __future__ import annotations
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_EPS = 1e-6
SUITE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- #
# Platt scaling
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_platt(fake_pct: Sequence[float], is_fake: Sequence[bool],
              l2: float = 1e-3, iters: int = 200) -> Dict[str, float]:
    """Fit p_cal = sigmoid(a*logit(p_raw/100) + b) by Newton's method (IRLS).

    Returns {"a": ..., "b": ..., "n": ..., "n_fake": ...}. Needs both classes
    present. A small L2 penalty keeps `a` finite on perfectly separable data.
    """
    x = _logit(np.asarray(fake_pct, dtype=float) / 100.0)
    y = np.asarray(is_fake, dtype=float)
    if len(x) < 4 or y.min() == y.max():
        raise ValueError("Need at least 4 samples with BOTH real and fake labels to calibrate.")

    theta = np.array([1.0, 0.0])          # [a, b]
    X = np.stack([x, np.ones_like(x)], axis=1)
    for _ in range(iters):
        p = _sigmoid(X @ theta)
        w = p * (1 - p) + 1e-9
        grad = X.T @ (p - y) + l2 * np.array([theta[0] - 1.0, 0.0])   # shrink a toward 1
        H = (X * w[:, None]).T @ X + l2 * np.eye(2)
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"a": float(theta[0]), "b": float(theta[1]), "n": int(len(x)), "n_fake": int(y.sum())}


def calibrate_pct(fake_pct: float, a: float, b: float) -> float:
    z = a * float(_logit(np.array([fake_pct / 100.0]))[0]) + b
    return float(_sigmoid(np.array([z]))[0] * 100.0)


def save_calibration(path: str, params: Dict[str, float], meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**params, "meta": meta or {}}, f, indent=2)


# --------------------------------------------------------------------------- #
# Loading + applying at inference time
# --------------------------------------------------------------------------- #
_cache: Dict[str, Optional[Tuple[float, float]]] = {}


def _candidate_paths(modality: str) -> List[str]:
    paths = []
    d = os.environ.get("DEEPFAKE_CALIBRATION_DIR")
    if d:
        paths.append(os.path.join(d, f"calibration_{modality}.json"))
    paths.append(os.path.join(SUITE_ROOT, "calibration", f"calibration_{modality}.json"))
    return paths


def load_calibration(modality: str, use_cache: bool = True) -> Optional[Tuple[float, float]]:
    if os.environ.get("DEEPFAKE_CALIBRATION", "").lower() == "off":
        return None
    if use_cache and modality in _cache:
        return _cache[modality]
    found = None
    for p in _candidate_paths(modality):
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    j = json.load(f)
                found = (float(j["a"]), float(j["b"]))
                break
            except (OSError, ValueError, KeyError):
                continue
    _cache[modality] = found
    return found


def clear_cache() -> None:
    _cache.clear()


def apply_calibration(modality: str, fake_pct: float) -> Tuple[float, bool]:
    """Returns (possibly-calibrated fake_pct, was_calibrated)."""
    params = load_calibration(modality)
    if params is None:
        return fake_pct, False
    return calibrate_pct(fake_pct, *params), True


# --------------------------------------------------------------------------- #
# Metrics that are independent of any threshold
# --------------------------------------------------------------------------- #
def roc_auc(fake_pct: Sequence[float], is_fake: Sequence[bool]) -> Optional[float]:
    """Probability a random fake scores higher than a random real (ties = 0.5).
    1.0 = perfect separation, 0.5 = no better than a coin."""
    s = np.asarray(fake_pct, dtype=float)
    y = np.asarray(is_fake, dtype=bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # rank-based (Mann-Whitney U) with average ranks for ties
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sorted_vals = allv[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def reliability_bins(fake_pct: Sequence[float], is_fake: Sequence[bool], n_bins: int = 10) -> List[dict]:
    """Bins by predicted fake-probability; compares mean prediction to actual fake rate."""
    p = np.asarray(fake_pct, dtype=float) / 100.0
    y = np.asarray(is_fake, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & ((p < hi) if i < n_bins - 1 else (p <= hi))
        if m.sum() == 0:
            continue
        out.append({"bin": f"{lo*100:.0f}-{hi*100:.0f}%", "n": int(m.sum()),
                    "mean_predicted_fake_pct": round(float(p[m].mean() * 100), 2),
                    "actual_fake_pct": round(float(y[m].mean() * 100), 2)})
    return out


def expected_calibration_error(fake_pct: Sequence[float], is_fake: Sequence[bool], n_bins: int = 10) -> float:
    """Weighted average gap (in percentage points) between stated and actual fake rate. 0 = perfectly calibrated."""
    bins = reliability_bins(fake_pct, is_fake, n_bins)
    n = sum(b["n"] for b in bins)
    if n == 0:
        return 0.0
    return float(sum(b["n"] * abs(b["mean_predicted_fake_pct"] - b["actual_fake_pct"]) for b in bins) / n)
