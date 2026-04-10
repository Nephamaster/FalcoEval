# uncertainty_metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np


EPS = 1e-12
# Utilities
def _safe_probs(p: np.ndarray) -> np.ndarray:
    """Clip + renormalize to avoid log(0) and sum drift."""
    p = np.clip(p, EPS, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return p


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy (nats) along given axis."""
    p = _safe_probs(p)
    return -np.sum(p * np.log(p), axis=axis)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    oh = np.zeros((y.shape[0], num_classes), dtype=float)
    oh[np.arange(y.shape[0]), y.astype(int)] = 1.0
    return oh


# Predictive Entropy
def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """
    预测熵：衡量单次预测分布的不确定性。熵越大 => 越不确定。
    probs: [N, C]
    returns: [N]
    """
    return _entropy(probs, axis=-1)


# Expected Entropy
def expected_entropy(probs_samples: np.ndarray) -> np.ndarray:
    """
    期望熵：多次采样(如温度/Dropout)下，每次熵的平均，反映“采样内在不确定性”。
    probs_samples: [S, N, C]
    returns: [N]
    """
    ent = _entropy(probs_samples, axis=-1)  # [S, N]
    return np.mean(ent, axis=0)


# Mutual Information (Epistemic)
def mutual_information(probs_samples: np.ndarray) -> np.ndarray:
    """
    互信息：MI = H(E[p]) - E[H(p)]，常用作 epistemic uncertainty 指标。
    probs_samples: [S, N, C]
    returns: [N]
    """
    p_bar = np.mean(probs_samples, axis=0)         # [N, C]
    h_bar = _entropy(p_bar, axis=-1)               # [N]
    e_h = expected_entropy(probs_samples)          # [N]
    return h_bar - e_h


# Logit Variance / Prob Variance
def logit_variance(logits_samples: np.ndarray) -> np.ndarray:
    """
    Logit 方差：多次采样下 logits 的方差（可做均值/最大聚合），反映输出不稳定性。
    logits_samples: [S, N, C]
    returns: [N]  (默认：对C维求均值后的方差)
    """
    var = np.var(logits_samples, axis=0)   # [N, C]
    return np.mean(var, axis=-1)           # [N]


def prob_variance(probs_samples: np.ndarray) -> np.ndarray:
    """
    概率方差：多次采样下概率的方差，数值越大越不稳定。
    probs_samples: [S, N, C]
    returns: [N]
    """
    var = np.var(probs_samples, axis=0)    # [N, C]
    return np.mean(var, axis=-1)


# Self-Consistency Score
def self_consistency_score(answers_samples: List[List[str]]) -> np.ndarray:
    """
    自一致性：同一问题多次生成中，出现频率最高答案的占比。越高 => 越确定/越稳定。
    answers_samples: length N, each is a list of S answer strings
    returns: [N]
    """
    scores = []
    for answers in answers_samples:
        if len(answers) == 0:
            scores.append(0.0)
            continue
        counts: Dict[str, int] = {}
        for a in answers:
            counts[a] = counts.get(a, 0) + 1
        top = max(counts.values())
        scores.append(top / len(answers))
    return np.array(scores, dtype=float)


# Semantic Variance (string-based)
def semantic_variance_tfidf(answers_samples: List[List[str]]) -> np.ndarray:
    """
    语义方差（TF-IDF近似版）：用 TF-IDF + cosine 距离衡量多次生成语义分散度。
    越大 => 越不确定（语义层面更“飘”）。
    注：不依赖深度embedding库，通用可跑；但精度不如 SBERT。
    returns: [N]
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    out = []
    for answers in answers_samples:
        if len(answers) <= 1:
            out.append(0.0)
            continue
        vec = TfidfVectorizer().fit_transform(answers)   # [S, V]
        sim = cosine_similarity(vec)                     # [S, S]
        # 语义分散度：1 - 平均相似度（去掉对角线）
        S = sim.shape[0]
        mean_offdiag = (np.sum(sim) - np.trace(sim)) / (S * (S - 1))
        out.append(float(1.0 - mean_offdiag))
    return np.array(out, dtype=float)


# ECE
def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    ECE：按置信度分箱后，计算(准确率-平均置信度)的加权平均。越小越校准。
    probs: [N, C], y_true: [N]
    """
    probs = _safe_probs(probs)
    conf = np.max(probs, axis=-1)
    pred = np.argmax(probs, axis=-1)
    acc = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = probs.shape[0]

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        ece += (np.sum(mask) / N) * abs(bin_acc - bin_conf)

    return float(ece)


# MCE
def maximum_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    MCE：分箱后误差的最大值（worst-case calibration gap）。越小越好。
    """
    probs = _safe_probs(probs)
    conf = np.max(probs, axis=-1)
    pred = np.argmax(probs, axis=-1)
    acc = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    gaps = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(acc[mask]))
        bin_conf = float(np.mean(conf[mask]))
        gaps.append(abs(bin_acc - bin_conf))

    return float(max(gaps) if gaps else 0.0)


# Risk–Coverage Curve + AURC
@dataclass
class RiskCoverageResult:
    coverages: np.ndarray   # increasing
    risks: np.ndarray       # risk at each coverage
    aurc: float             # area under risk-coverage curve

def risk_coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray
) -> RiskCoverageResult:
    """
    Risk-Coverage：按置信度从高到低逐步“接收预测”，计算每个覆盖率下的风险(=错误率)。
    AURC 越小越好（表示模型能用不确定性把错误“排到后面”）。
    """
    probs = _safe_probs(probs)
    conf = np.max(probs, axis=-1)
    pred = np.argmax(probs, axis=-1)
    err = (pred != y_true).astype(float)

    order = np.argsort(-conf)  # high->low confidence
    err_sorted = err[order]

    N = len(y_true)
    coverages = np.arange(1, N + 1) / N
    cum_err = np.cumsum(err_sorted)
    risks = cum_err / np.arange(1, N + 1)

    # AURC via trapezoidal rule
    aurc = float(np.trapz(risks, coverages))
    return RiskCoverageResult(coverages=coverages, risks=risks, aurc=aurc)


# Selective Accuracy (given threshold)
@dataclass
class SelectiveResult:
    threshold: float
    coverage: float
    selective_accuracy: float

def selective_accuracy(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.7
) -> SelectiveResult:
    """
    Selective Accuracy：只在 conf>=threshold 的样本上算准确率，同时给出 coverage。
    适合部署（拒答/转人工）场景。
    """
    probs = _safe_probs(probs)
    conf = np.max(probs, axis=-1)
    pred = np.argmax(probs, axis=-1)

    mask = conf >= threshold
    if np.sum(mask) == 0:
        return SelectiveResult(threshold, coverage=0.0, selective_accuracy=0.0)

    acc = np.mean((pred[mask] == y_true[mask]).astype(float))
    cov = float(np.mean(mask.astype(float)))
    return SelectiveResult(threshold, coverage=cov, selective_accuracy=float(acc))


if __name__ == "__main__":
    # Example: classification with 3 classes, N=5 samples
    np.random.seed(0)
    N, C, S = 5, 3, 8

    # single-run probs
    logits = np.random.randn(N, C)
    probs = _softmax(logits)

    # multi-sample probs/logits (e.g., MC dropout / temperature samples)
    logits_samples = np.random.randn(S, N, C)
    probs_samples = _softmax(logits_samples, axis=-1)

    # fake labels
    y_true = np.random.randint(0, C, size=(N,))

    # fake multi-answer samples per item (for generative setting)
    answers_samples = [
        ["A", "A", "B", "A", "A"],
        ["yes", "no", "yes", "no"],
        ["42", "42", "42"],
        ["foo"],
        ["x", "y", "z", "x"],
    ]

    print("1 Predictive Entropy:", predictive_entropy(probs))
    print("2 Expected Entropy:", expected_entropy(probs_samples))
    print("3 Mutual Information:", mutual_information(probs_samples))
    print("4 Logit Variance:", logit_variance(logits_samples))
    print("5 Self-Consistency:", self_consistency_score(answers_samples))
    print("6 Semantic Variance (TFIDF):", semantic_variance_tfidf(answers_samples))
    print("7 ECE:", expected_calibration_error(probs, y_true))
    print("8 MCE:", maximum_calibration_error(probs, y_true))
    rc = risk_coverage_curve(probs, y_true)
    print("9 AURC:", rc.aurc)
    sel = selective_accuracy(probs, y_true, threshold=0.7)
    print("10 Selective Accuracy:", sel)