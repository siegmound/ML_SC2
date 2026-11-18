"""S10: probability-quality diagnostics on matched validation checkpoints.

Fixed-width reliability bins, ECE and Brier-skill diagnostics are descriptive.
Lower log loss / Brier alone does not establish full calibration.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def validate_binary_probabilities(labels, probabilities):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError("Expected aligned nonempty 1D binary labels and probabilities")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or (
        (p < 0) | (p > 1)
    ).any():
        raise ValueError("Invalid binary targets or probability values")
    return y.astype(np.int8), p


def reliability_bins(labels, probabilities, *, bins: int = 10):
    """Fixed equal-width [0,1] bins; the final bin includes p=1.

    Empty bins have zero count and null observed/predicted frequencies. ECE
    weights absolute binwise gaps by bin occupancy (not by confidence).
    """
    y, p = validate_binary_probabilities(labels, probabilities)
    if bins < 2 or bins > 100 or not isinstance(bins, int):
        raise ValueError("Choose 2-100 integer bins")
    bin_id = np.minimum((p * bins).astype(np.int32), bins - 1)
    sizes = np.bincount(bin_id, minlength=bins)
    prob_sums = np.bincount(bin_id, weights=p, minlength=bins)
    actual_sums = np.bincount(bin_id, weights=y, minlength=bins)
    out = []
    for k in range(bins):
        count = int(sizes[k])
        pred = float(prob_sums[k] / count) if count else None
        actual = float(actual_sums[k] / count) if count else None
        out.append({
            "bin": k, "lower": k / bins, "upper": (k + 1) / bins,
            "n": count, "predicted_p1": pred, "observed_p1": actual,
            "absolute_gap": abs(pred - actual) if count else None,
        })
    return out


def calibration_diagnostics(labels, probabilities, *, bins: int = 10):
    y, p = validate_binary_probabilities(labels, probabilities)
    reliability = reliability_bins(y, p, bins=bins)
    ece = float(sum((r["n"] / len(y)) * r["absolute_gap"]
                    for r in reliability if r["n"]))
    prevalence = float(np.mean(y))
    brier = float(brier_score_loss(y, p))
    climatology = prevalence * (1 - prevalence)
    return {
        "n": len(y), "prevalence_p1": prevalence,
        "mean_predicted_p1": float(np.mean(p)),
        "calibration_bias_p1": float(np.mean(p) - prevalence),
        "expected_calibration_error": ece,
        "brier_score": brier,
        "brier_climatology_reference": float(climatology),
        "brier_skill_vs_climatology": float(1 - brier / climatology) if climatology > 0 else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "probability_variance": float(np.var(p)),
        "reliability_bins": reliability,
    }


def paired_brier_bootstrap(labels, baseline_prob, candidate_prob, *,
                           reps: int = 1000, seed: int = 2026):
    """Replay-paired Brier difference (candidate minus baseline).

    This expects exactly ONE checkpoint prediction per replay at a time. Do not
    pass stacked checkpoints, multiple splits, or individual per-time rows.
    """
    y, base = validate_binary_probabilities(labels, baseline_prob)
    yy, cand = validate_binary_probabilities(labels, candidate_prob)
    if reps < 0:
        raise ValueError("Bootstrap repetitions cannot be negative")
    if not np.array_equal(y, yy):
        raise ValueError("Baseline and candidate labels differ")
    difference = (y - cand) ** 2 - (y - base) ** 2
    point = float(difference.mean())
    ci = None
    if reps:
        rng = np.random.default_rng(seed)
        means = np.empty(reps, dtype=np.float64)
        # Chunk to limit peak CPU RAM while retaining exact paired resampling.
        for start in range(0, reps, 64):
            end = min(start + 64, reps)
            sample = rng.integers(0, len(y), size=(end - start, len(y)))
            means[start:end] = difference[sample].mean(axis=1)
        ci = [float(x) for x in np.quantile(means, [0.025, 0.975])]
    return {
        "delta_brier_prefix_minus_last9": point,
        "delta_brier_ci95": ci,
        "replay_groups": len(y),
        "replicates": reps,
        "resampling_unit": "one_checkpoint_prediction_per_replay",
    }
