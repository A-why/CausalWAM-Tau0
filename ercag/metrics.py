"""Tie-safe rank metrics (V4-B5 §3-§5).

Fixes the V4-B3 / V4-B4 Spearman ties-bug. The old implementation ranked with an
identity-permutation ``argsort`` fallback for tied values, so a collapsed predictor
could receive an arbitrary rho in [-0.93, +0.98]. The corrected metric follows
scipy's tie convention (rankdata method='average' -> Pearson) and treats degenerate /
under-dispersed predictors explicitly.

Fixed rules (do NOT tune to S2; thresholds locked before re-evaluation):

    - constant predictor or < 2 unique predictions  -> rho = NaN, DEGENERATE_PREDICTOR
    - DISPERSION_RATIO = std(pred) / (std(target)+1e-12) < 0.10
        -> rho is RECORDED but NOT usable for PASS / checkpoint-selection / mean-rho;
           status = UNDER_DISPERSED
    - otherwise -> tie-safe Spearman, status = OK
"""
import numpy as np
from scipy.stats import rankdata

DISPERSION_RATIO_THRESHOLD = 0.10


def _tie_safe_rank(x):
    """Average-rank (scipy 'average' method) — ties get their mean rank."""
    return rankdata(np.asarray(x, dtype=float), method="average")


def spearman_tie_safe(a, b):
    """Tie-safe Spearman. Returns (rho, status).

    status in {OK, DEGENERATE_PREDICTOR}. A constant input (or <2 unique values) is a
    DEGENERATE_PREDICTOR and yields NaN — never an arbitrary identity-permutation rho.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    if n < 2:
        return float("nan"), "DEGENERATE_PREDICTOR"
    if len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan"), "DEGENERATE_PREDICTOR"
    ra = _tie_safe_rank(a)
    rb = _tie_safe_rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan"), "DEGENERATE_PREDICTOR"
    return float(np.corrcoef(ra, rb)[0, 1]), "OK"


def dispersion_status(pred, target, threshold=DISPERSION_RATIO_THRESHOLD):
    """Return (dispersion_ratio, n_unique_pred, status).

    status in {OK, UNDER_DISPERSED, DEGENERATE_PREDICTOR}. UNDER_DISPERSED means the
    predictor is not a degenerate constant but its spread is <10% of the target spread,
    so any rank correlation is still spurious and must be excluded from PASS / selection.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    ts = float(np.std(target))
    ps = float(np.std(pred))
    ratio = ps / (ts + 1e-12)
    n_unique = int(len(np.unique(pred)))
    if n_unique < 2:
        return ratio, n_unique, "DEGENERATE_PREDICTOR"
    if ratio < threshold:
        return ratio, n_unique, "UNDER_DISPERSED"
    return ratio, n_unique, "OK"


def snapshot_rank_record(pred, target, label=None):
    """One per-snapshot record: target/pred std, dispersion ratio, unique count, tie-safe rho, status.

    ``pred`` / ``target`` are 1-D arrays of equal length (the candidate values at one horizon).
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    rho, rho_status = spearman_tie_safe(pred, target)
    ratio, n_unique, disp_status = dispersion_status(pred, target)
    # effective status: degenerate > under-dispersed > ok (any degenerate is unusable)
    if rho_status == "DEGENERATE_PREDICTOR" or disp_status == "DEGENERATE_PREDICTOR":
        status = "DEGENERATE_PREDICTOR"
        rho = float("nan")
    elif disp_status == "UNDER_DISPERSED":
        status = "UNDER_DISPERSED"
    else:
        status = "OK"
    rec = {
        "label": label,
        "n": int(len(pred)),
        "target_std": float(np.std(target)),
        "pred_std": float(np.std(pred)),
        "dispersion_ratio": ratio,
        "n_unique_pred": n_unique,
        "spearman_tie_safe": rho,
        "status": status,
        "usable_for_pass": bool(status == "OK"),
    }
    return rec


def aggregate(records):
    """Aggregate per-snapshot records into mean/median/positive-count over USABLE snapshots only.

    Under-dispersed / degenerate snapshots are counted separately and NEVER mixed into the
    mean/median/positive statistics (V4-B5 §42).
    """
    usable = [r for r in records if r["status"] == "OK"]
    under = [r for r in records if r["status"] == "UNDER_DISPERSED"]
    deg = [r for r in records if r["status"] == "DEGENERATE_PREDICTOR"]
    rhos = [r["spearman_tie_safe"] for r in usable if not np.isnan(r["spearman_tie_safe"])]
    return {
        "n_total": len(records),
        "n_usable": len(usable),
        "n_under_dispersed": len(under),
        "n_degenerate": len(deg),
        "mean_rho": float(np.mean(rhos)) if rhos else float("nan"),
        "median_rho": float(np.median(rhos)) if rhos else float("nan"),
        "n_positive": int(sum(1 for r in rhos if r > 0)),
        "n_negative": int(sum(1 for r in rhos if r < 0)),
        "n_zero": int(sum(1 for r in rhos if r == 0)),
    }
