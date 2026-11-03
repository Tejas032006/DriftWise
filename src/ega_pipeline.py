"""DriftWise / EGA experiment pipeline.

Usage in a Colab notebook:
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/DriftWise/src")
    from ega_pipeline import *

Needs ega_gate.py (the sequential gate) in the same folder, plus numpy, pandas, scikit-learn and river.
"""
import copy
import numpy as np
import pandas as pd
from scipy import stats
from river import drift
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from ega_gate import sequential_gate as _sequential_gate_raw

# Cost counters: every model trained, every gate test, and every validation label used.
COST = {"fits": 0, "gate_calls": 0, "val_rows": 0}

def reset_cost():
    for k in COST:
        COST[k] = 0

def sequential_gate(y, *args, **kwargs):
    COST["gate_calls"] += 1
    COST["val_rows"] += len(y)
    return _sequential_gate_raw(y, *args, **kwargs)


def make_stream(seed, condition, n=10000, drift_point=5000):
    r = np.random.default_rng(seed)
    p = np.where(np.arange(n) < drift_point, 0.05, 0.20)
    y = (r.random(n) < p).astype(int)
    X = r.normal(size=(n, 2))
    if condition == "pattern_drift":
        X[:drift_point, 0] += 2.0 * y[:drift_point]
        X[drift_point:, 1] += 2.0 * y[drift_point:]
    else:
        X[:, 0] += 2.0 * y
    return X, y


def fit_model(X, y):
    COST["fits"] += 1
    sc = StandardScaler()
    m = SGDClassifier(loss="log_loss", class_weight="balanced", random_state=42)
    m.fit(sc.fit_transform(X), y)
    return sc, m


def predict(model, X):
    sc, m = model
    return m.predict(sc.transform(X))


def incremental_update(model, X, y, epochs=5):
    COST["fits"] += 1
    sc, m = model
    inc = copy.deepcopy(m)
    inc.set_params(class_weight=None)
    cnt = np.bincount(y, minlength=2)
    w = len(y) / (2.0 * np.maximum(cnt, 1))
    Xs = sc.transform(X)
    for _ in range(epochs):
        inc.partial_fit(Xs, y, classes=[0, 1], sample_weight=w[y])
    return sc, inc


def new_adwin():
    return drift.ADWIN(delta=0.002)


def adwin_step(adwin, v):
    out = adwin.update(int(v))
    return out[0] if isinstance(out, tuple) else adwin.drift_detected

def run_delayed(method, X, y, lag=0, start=4000, chunk=50, gate_every=100, T=1500,
                val_size=400, gap=50, sev_thr=0.06, prob_len=600):
    n = len(y)
    model = fit_model(X[:start], y[:start])
    preds = np.zeros(n, dtype=int)
    adwin = new_adwin()
    in_ep, alarm_t, look = False, None, 0
    prob = None                                   # dict while on probation
    updates, alarms, rollbacks = [], [], 0

    for s in range(start, n, chunk):
        e = min(s + chunk, n)
        preds[s:e] = predict(model, X[s:e])
        t = e - 1                                 # present time
        tl = t - lag                              # last record whose label has arrived

        # labels that arrive during this chunk feed ADWIN
        lo, hi = max(s - lag, start), e - lag
        fired = False
        for i in range(lo, max(hi, lo)):
            fired = adwin_step(adwin, int(preds[i] != y[i])) or fired
        if fired:
            alarms.append(t)
        if tl < start + 700:
            continue

        # ---- probation: old model in shadow, same gate with roles swapped ----
        if prob is not None:
            n_lab = tl - prob["start"] + 1
            if n_lab >= 200 and n_lab % 100 == 0:
                idx = slice(prob["start"], tl + 1)
                yv = y[idx]
                p_cur = predict(model, X[idx])
                p_old = predict(prob["old"], X[idx])
                prob["look"] += 1
                r = sequential_gate(yv, p_old, p_cur, look=prob["look"])   # is OLD reliably better?
                if r["decision"] == "ACCEPT":
                    model = prob["old"]; rollbacks += 1; updates.append((t, "rollback"))
                    prob = None; adwin = new_adwin()
                elif n_lab >= prob_len:
                    prob = None
            continue

        if method == "every_alarm":
            if fired:
                model = fit_model(X[tl - 299:tl + 1], y[tl - 299:tl + 1])
                updates.append((t, "retrain")); adwin = new_adwin()

        elif method in ("ega", "ega_prob"):
            if (not in_ep) and fired:
                in_ep, alarm_t, look = True, tl, 0
            if in_ep and (tl - alarm_t) % gate_every == 0:
                look += 1
                k = min(tl + 1, start + 500) - start
                base_err = (preds[start:start + k] != y[start:start + k]).mean()
                recent_err = (preds[tl - 199:tl + 1] != y[tl - 199:tl + 1]).mean()
                pool = ["inc"] if (recent_err - base_err) < sev_thr else ["short", "long", "inc"]

                Xv, yv = X[tl + 1 - val_size:tl + 1], y[tl + 1 - val_size:tl + 1]
                te = tl + 1 - val_size - gap
                p_cur = predict(model, Xv)
                best = None
                for name in pool:
                    if name == "short":
                        cand = fit_model(X[te - 300:te], y[te - 300:te])
                    elif name == "long":
                        cand = fit_model(X[te - 600:te], y[te - 600:te])
                    else:
                        cand = incremental_update(model, X[te - 300:te], y[te - 300:te])
                    r = sequential_gate(yv, predict(cand, Xv), p_cur, look=look,
                                        alpha_total=0.05 / len(pool))
                    if r["decision"] == "ACCEPT" and (best is None or r["est_balacc"] > best[1]["est_balacc"]):
                        best = (cand, r, name)
                if best is not None:
                    if method == "ega_prob":
                        prob = {"old": model, "start": t + 1, "look": 0}
                    model = best[0]; updates.append((t, best[2]))
                    in_ep = False; adwin = new_adwin()
                elif tl - alarm_t >= T:
                    in_ep = False; adwin = new_adwin()

    ye, pe = y[5000:], preds[5000:]
    r0 = (pe[ye == 0] == 0).mean(); r1 = (pe[ye == 1] == 1).mean()
    after = [u[0] for u in updates if u[0] >= 5000 and u[1] != "rollback"]
    return {"balanced_acc": (r0 + r1) / 2, "worst_class_recall": min(r0, r1),
            "n_updates": len([u for u in updates if u[1] != "rollback"]),
            "n_rollbacks": rollbacks,
            "first_update_delay": (after[0] - 5000) if after else np.nan}


def make_stream_gradual(seed, n=10000, start_d=5000, width=2000):
    r = np.random.default_rng(seed)
    w = np.clip((np.arange(n) - start_d) / width, 0, 1)
    y = (r.random(n) < (0.05 + 0.15 * w)).astype(int)
    X = r.normal(size=(n, 2))
    use_new = r.random(n) < w
    X[~use_new, 0] += 2.0 * y[~use_new]
    X[use_new, 1] += 2.0 * y[use_new]
    return X, y


def get_stream(seed, cond):
    return make_stream_gradual(seed) if cond == "gradual_drift" else make_stream(seed, cond)


def v1_decide(yv, p_new, p_old, rng, n_boot=500):
    """Old gate: fresh 95% paired bootstrap at every look. Returns (decision, point estimate of bal-acc gain)."""
    if len(yv) < 200 or yv.sum() < 30 or (yv == 0).sum() < 30:
        return "DEFER", 0.0
    n = len(yv)
    idx = rng.integers(0, n, (n_boot, n))
    yb = yv[idx]
    def stats(pred):
        pb = pred[idx]; f1, rec = [], []
        for c in (0, 1):
            tp = ((yb == c) & (pb == c)).sum(1)
            fp = ((yb != c) & (pb == c)).sum(1)
            fn = ((yb == c) & (pb != c)).sum(1)
            f1.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
            rec.append(tp / np.maximum(tp + fn, 1))
        return np.mean(f1, 0), np.mean(rec, 0), np.minimum(rec[0], rec[1])
    fn_, bn_, wn_ = stats(p_new); fo_, bo_, wo_ = stats(p_old)
    lb = lambda a, b: np.percentile(a - b, 2.5)
    ok = (lb(fn_, fo_) > -0.01) and (lb(wn_, wo_) > -0.03) and (lb(bn_, bo_) > 0)
    est = float(np.mean([(p_new[yv == c] == c).mean() - (p_old[yv == c] == c).mean() for c in (0, 1)]))
    return ("ACCEPT" if ok else "REJECT"), est


def run_cfg(X, y, lag=0, gate="seq", pool_mode="auto", prob=True, adwin_delta=0.002,
            val_size=400, start=4000, chunk=50, gate_every=100, T=1500, gap=50,
            sev_thr=0.06, prob_len=600, seed=0):
    n = len(y); rng = np.random.default_rng(seed)
    model = fit_model(X[:start], y[:start])
    preds = np.zeros(n, dtype=int)
    adwin = drift.ADWIN(delta=adwin_delta)
    in_ep, alarm_t, look, probation = False, None, 0, None
    updates, rollbacks = [], 0

    for s in range(start, n, chunk):
        e = min(s + chunk, n)
        preds[s:e] = predict(model, X[s:e])
        t = e - 1; tl = t - lag
        fired = False
        for i in range(max(s - lag, start), max(e - lag, max(s - lag, start))):
            fired = adwin_step(adwin, int(preds[i] != y[i])) or fired
        if tl < start + 700:
            continue

        if probation is not None:                       # old model in shadow
            n_lab = tl - probation["start"] + 1
            if n_lab >= 200 and n_lab % 100 == 0:
                idx = slice(probation["start"], tl + 1)
                probation["look"] += 1
                r = sequential_gate(y[idx], predict(probation["old"], X[idx]), predict(model, X[idx]),
                                    look=probation["look"])
                if r["decision"] == "ACCEPT":
                    model = probation["old"]; rollbacks += 1
                    probation = None; adwin = drift.ADWIN(delta=adwin_delta)
                elif n_lab >= prob_len:
                    probation = None
            continue

        if (not in_ep) and fired:
            in_ep, alarm_t, look = True, tl, 0
        if in_ep and (tl - alarm_t) % gate_every == 0:
            look += 1
            k = min(tl + 1, start + 500) - start
            base_err = (preds[start:start + k] != y[start:start + k]).mean()
            recent_err = (preds[tl - 199:tl + 1] != y[tl - 199:tl + 1]).mean()
            if pool_mode == "short_only":
                pool = ["short"]
            else:
                pool = ["inc"] if (recent_err - base_err) < sev_thr else ["short", "long", "inc"]

            Xv, yv = X[tl + 1 - val_size:tl + 1], y[tl + 1 - val_size:tl + 1]
            te = tl + 1 - val_size - gap
            p_cur = predict(model, Xv)
            best = None
            for name in pool:
                if name == "short":
                    cand = fit_model(X[te - 300:te], y[te - 300:te])
                elif name == "long":
                    cand = fit_model(X[te - 600:te], y[te - 600:te])
                else:
                    cand = incremental_update(model, X[te - 300:te], y[te - 300:te])
                p_new = predict(cand, Xv)
                if gate == "seq":
                    r = sequential_gate(yv, p_new, p_cur, look=look, alpha_total=0.05 / len(pool))
                    dec, est = r["decision"], r.get("est_balacc", 0.0)
                else:
                    dec, est = v1_decide(yv, p_new, p_cur, rng)
                if dec == "ACCEPT" and (best is None or est > best[1]):
                    best = (cand, est, name)
            if best is not None:
                if prob:
                    probation = {"old": model, "start": t + 1, "look": 0}
                model = best[0]; updates.append(t)
                in_ep = False; adwin = drift.ADWIN(delta=adwin_delta)
            elif tl - alarm_t >= T:
                in_ep = False; adwin = drift.ADWIN(delta=adwin_delta)

    ye, pe = y[5000:], preds[5000:]
    r0 = (pe[ye == 0] == 0).mean(); r1 = (pe[ye == 1] == 1).mean()
    post = [u for u in updates if u >= 5000]
    return {"balanced_acc": (r0 + r1) / 2, "worst_class_recall": min(r0, r1),
            "n_updates": len(updates), "updates_before_5000": len(updates) - len(post),
            "n_rollbacks": rollbacks, "first_update_delay": (post[0] - 5000) if post else np.nan}


def final_metrics(y, preds, updates):
    ye, pe = y[5000:], preds[5000:]
    r0 = (pe[ye == 0] == 0).mean(); r1 = (pe[ye == 1] == 1).mean()
    post = [u for u in updates if u >= 5000]
    return {"balanced_acc": (r0 + r1) / 2, "worst_class_recall": min(r0, r1),
            "n_updates": len(updates), "updates_before_5000": len(updates) - len(post),
            "n_rollbacks": 0, "first_update_delay": (post[0] - 5000) if post else np.nan}


def run_base(method, X, y, lag=0, start=4000, chunk=50):
    n = len(y)
    model = fit_model(X[:start], y[:start])
    preds = np.zeros(n, dtype=int)
    adwin = new_adwin()
    updates = []
    for s in range(start, n, chunk):
        e = min(s + chunk, n)
        preds[s:e] = predict(model, X[s:e])
        t = e - 1; tl = t - lag
        fired = False
        lo = max(s - lag, start)
        for i in range(lo, max(e - lag, lo)):
            fired = adwin_step(adwin, int(preds[i] != y[i])) or fired
        if method == "periodic" and (t + 1 - start) % 1000 == 0 and tl >= start + 500:
            model = fit_model(X[tl - 499:tl + 1], y[tl - 499:tl + 1]); updates.append(t)
        elif method == "every_alarm" and fired and tl >= start + 300:
            model = fit_model(X[tl - 299:tl + 1], y[tl - 299:tl + 1]); updates.append(t)
            adwin = new_adwin()
    return final_metrics(y, preds, updates)


def run_ega_v2(X, y, lag=0, patience=None, start=4000, chunk=50, gate_every=100, T=1500,
               val_size=400, gap=50, sev_thr=0.06):
    """Same EGA loop as run_delayed("ega"), with one option:
       patience=None -> stop after the first accepted candidate (original behaviour)
       patience=k    -> keep gating after an accept; end the episode after k looks with no further accept"""
    n = len(y)
    model = fit_model(X[:start], y[:start])
    preds = np.zeros(n, dtype=int)
    adwin = new_adwin()
    in_ep, alarm_t, look = False, None, 0
    n_acc, streak = 0, 0
    updates = []

    for s in range(start, n, chunk):
        e = min(s + chunk, n)
        preds[s:e] = predict(model, X[s:e])
        t = e - 1
        tl = t - lag

        lo, hi = max(s - lag, start), e - lag
        fired = False
        for i in range(lo, max(hi, lo)):
            fired = adwin_step(adwin, int(preds[i] != y[i])) or fired
        if tl < start + 700:
            continue

        if (not in_ep) and fired:
            in_ep, alarm_t, look, n_acc, streak = True, tl, 0, 0, 0

        if in_ep and (tl - alarm_t) % gate_every == 0:
            look += 1
            k = min(tl + 1, start + 500) - start
            base_err = (preds[start:start + k] != y[start:start + k]).mean()
            recent_err = (preds[tl - 199:tl + 1] != y[tl - 199:tl + 1]).mean()
            pool = ["inc"] if (recent_err - base_err) < sev_thr else ["short", "long", "inc"]

            Xv, yv = X[tl + 1 - val_size:tl + 1], y[tl + 1 - val_size:tl + 1]
            te = tl + 1 - val_size - gap
            p_cur = predict(model, Xv)
            best = None
            for name in pool:
                if name == "short":
                    cand = fit_model(X[te - 300:te], y[te - 300:te])
                elif name == "long":
                    cand = fit_model(X[te - 600:te], y[te - 600:te])
                else:
                    cand = incremental_update(model, X[te - 300:te], y[te - 300:te])
                r = sequential_gate(yv, predict(cand, Xv), p_cur, look=look,
                                    alpha_total=0.05 / len(pool))
                if r["decision"] == "ACCEPT" and (best is None or r["est_balacc"] > best[1]["est_balacc"]):
                    best = (cand, r, name)

            if best is not None:
                model = best[0]; updates.append(t); n_acc += 1; streak = 0
                if patience is None:
                    in_ep = False; adwin = new_adwin()
            else:
                streak += 1
                if patience is not None and n_acc > 0 and streak >= patience:
                    in_ep = False; adwin = new_adwin()
                elif tl - alarm_t >= T:
                    in_ep = False; adwin = new_adwin()
            if in_ep and tl - alarm_t >= T:
                in_ep = False; adwin = new_adwin()

    ye, pe = y[5000:], preds[5000:]
    r0 = (pe[ye == 0] == 0).mean(); r1 = (pe[ye == 1] == 1).mean()
    after = [u for u in updates if u >= 5000]
    return {"balanced_acc": (r0 + r1) / 2, "worst_class_recall": min(r0, r1),
            "n_updates": len(updates),
            "first_update_delay": (after[0] - 5000) if after else np.nan}


def run_retrain_schedule(X, y, offsets=(0, 500), lag=0, start=4000, chunk=50, window=300):
    """No gate. After an alarm, retrain on the newest `window` labels at each offset (records after the alarm)."""
    n = len(y)
    model = fit_model(X[:start], y[:start])
    preds = np.zeros(n, dtype=int)
    adwin = new_adwin()
    in_ep, alarm_t = False, None
    updates = []
    for s in range(start, n, chunk):
        e = min(s + chunk, n)
        preds[s:e] = predict(model, X[s:e])
        t = e - 1
        tl = t - lag
        lo, hi = max(s - lag, start), e - lag
        fired = False
        for i in range(lo, max(hi, lo)):
            fired = adwin_step(adwin, int(preds[i] != y[i])) or fired
        if tl < start + 700:
            continue
        if (not in_ep) and fired:
            in_ep, alarm_t = True, tl
        if in_ep:
            gone = tl - alarm_t
            if gone in offsets:
                model = fit_model(X[tl - window + 1:tl + 1], y[tl - window + 1:tl + 1])
                updates.append(t)
            if gone >= max(offsets):
                in_ep = False
                adwin = new_adwin()
    ye, pe = y[5000:], preds[5000:]
    r0 = (pe[ye == 0] == 0).mean(); r1 = (pe[ye == 1] == 1).mean()
    after = [u for u in updates if u >= 5000]
    return {"balanced_acc": (r0 + r1) / 2, "worst_class_recall": min(r0, r1),
            "n_updates": len(updates),
            "first_update_delay": (after[0] - 5000) if after else np.nan}
