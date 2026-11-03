
import numpy as np
from math import pi
from scipy.stats import norm


def alpha_for_look(look, alpha_total=0.05):
    return alpha_total * 6.0 / (pi ** 2 * look ** 2)


def _lower_bound(diffs_by_class, z):
    est, var = 0.0, 0.0
    k = len(diffs_by_class)
    for d in diffs_by_class:
        est += d.mean() / k
        var += (d.var(ddof=1) / len(d)) / (k ** 2)
    return est, est - z * np.sqrt(max(var, 1e-12))


def sequential_gate(y, p_new, p_old, look, alpha_total=0.05,
                    min_n=200, min_per_class=30,
                    delta_overall=0.01, delta_class=0.03,
                    min_effect=0.0, cost_penalty=0.0, guard_all_classes=False):
    y, p_new, p_old = map(np.asarray, (y, p_new, p_old))
    a = alpha_for_look(look, alpha_total)
    z = norm.ppf(1 - a)

    classes = np.unique(y)
    counts = {int(c): int((y == c).sum()) for c in classes}
    enough = len(y) >= min_n and len(classes) == 2 and all(n >= min_per_class for n in counts.values())
    out = {"look": look, "alpha_spent": a, "n": len(y), "counts": counts}
    if not enough:
        out.update(decision="DEFER", A=False, B=None, C=None, D=None)
        return out

    d_all = (p_new == y).astype(float) - (p_old == y).astype(float)
    by_class = {int(c): d_all[y == c] for c in classes}
    est_ba, lb_ba = _lower_bound(list(by_class.values()), z)

    old_recall = {c: float((p_old[y == c] == c).mean()) for c in by_class}
    if guard_all_classes:
        guarded = list(by_class)
        zc = norm.ppf(1 - a / len(guarded))
    else:
        guarded = [min(old_recall, key=old_recall.get)]
        zc = z
    class_lbs = {}
    for c in guarded:
        d = by_class[c]
        class_lbs[c] = d.mean() - zc * np.sqrt(max(d.var(ddof=1) / len(d), 1e-12))
    worst_lb = min(class_lbs.values())

    B = lb_ba > -delta_overall
    C = worst_lb > -delta_class
    D = (lb_ba - cost_penalty > 0) and (est_ba >= min_effect)

    out.update(A=True, B=bool(B), C=bool(C), D=bool(D),
               est_balacc=est_ba, lb_balacc=lb_ba, lb_class_guard=worst_lb,
               guarded_classes=guarded,
               decision="ACCEPT" if (B and C and D) else "REJECT")
    return out


def block_bootstrap_macro_f1_diff(y, p_new, p_old, block=50, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    y, p_new, p_old = map(np.asarray, (y, p_new, p_old))
    n = len(y)
    n_blocks = int(np.ceil(n / block))

    def macro_f1(yy, pp):
        f = []
        for c in (0, 1):
            tp = np.sum((yy == c) & (pp == c)); fp = np.sum((yy != c) & (pp == c)); fn = np.sum((yy == c) & (pp != c))
            den = 2 * tp + fp + fn
            f.append(2 * tp / den if den else 0.0)
        return np.mean(f)

    diffs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        diffs.append(macro_f1(y[idx], p_new[idx]) - macro_f1(y[idx], p_old[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
