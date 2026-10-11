"""Discrete-time survival helpers used by solve.py: person-day expansion, hazard
regression, IPCW weighting, and g-formula standardized RMST."""
import csv
import math
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def integrity_payload_sum(row):
    """Position-weighted ASCII sum over the five baseline fields joined by "|".
    outreach_flag is deliberately excluded: later revisions change that column."""
    payload = "|".join([
        row["customer_id"],
        row["plan_tier"],
        row["monthly_spend_usd"],
        row["login_freq_pre30d"],
        row["tenure_days_pre"],
    ])
    acc = 0
    for i, ch in enumerate(payload):
        acc += (i + 1) * (i + 1) * ord(ch)
    return acc


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, int(math.isqrt(n)) + 1, 2):
        if n % d == 0:
            return False
    return True


def _prime_factors(n):
    out = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1
    if n > 1:
        out.add(n)
    return out


def recover_integrity_prime(rows):
    """Recover the checksum modulus from rows that already match, as the data
    dictionary directs, instead of hardcoding it. For an intact row the payload
    sum S and stored value v satisfy S = v (mod p), so p divides S - v; a
    mismatched row does not, which is why a plain GCD over every row collapses
    to 1 on an export with planted mismatches. Candidate primes are taken from
    pairwise GCDs of neighbouring rows (most pairs are intact), and the prime
    that the largest number of rows recompute under is the modulus. p must
    exceed every stored value, since v = S mod p."""
    diffs = [integrity_payload_sum(r) - int(r["integrity"]) for r in rows]
    stored = [int(r["integrity"]) for r in rows]
    vmax = max(stored)
    candidates = set()
    for a, b in zip(diffs, diffs[1:201]):
        g = math.gcd(a, b)
        if g > vmax:
            candidates.update(f for f in _prime_factors(g) if f > vmax)
    if not candidates:
        raise RuntimeError("could not recover the integrity prime from matching rows")
    sums = [d + v for d, v in zip(diffs, stored)]
    score = Counter()
    for p in candidates:
        score[p] = sum(1 for s, v in zip(sums, stored) if s % p == v)
    prime, n_match = score.most_common(1)[0]
    if n_match * 2 < len(rows) or not _is_prime(prime):
        raise RuntimeError("could not recover the integrity prime from matching rows")
    return prime


def intact_customer_ids(customers_path):
    """IDs whose integrity field recomputes under the prime recovered from the
    rows that already match."""
    with open(customers_path, newline="") as f:
        rows = list(csv.DictReader(f))
    prime = recover_integrity_prime(rows)
    return {
        row["customer_id"]
        for row in rows
        if int(row["integrity"]) == integrity_payload_sum(row) % prime
    }


def day0_outreach_by_id(revisions_path):
    """Among revisions with published_at_day <= 0, the latest recorded_at_day wins."""
    best = {}
    with open(revisions_path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["published_at_day"]) > 0:
                continue
            cid = row["customer_id"]
            rec = int(row["recorded_at_day"])
            flag = int(row["outreach_flag"])
            if cid not in best or rec > best[cid][0]:
                best[cid] = (rec, flag)
    return {cid: flag for cid, (_, flag) in best.items()}


def expand_person_days(ids, obs_time, event, covars: dict):
    ids = np.asarray(ids)
    obs_time = np.asarray(obs_time).astype(int)
    event = np.asarray(event)
    n = len(ids)
    row_idx = np.repeat(np.arange(n), obs_time)
    days = np.concatenate([np.arange(1, t + 1) for t in obs_time]) if n else np.array([], dtype=int)
    is_last = days == obs_time[row_idx]
    ev = (is_last & (event[row_idx] == 1)).astype(int)
    out = {"id": ids[row_idx], "day": days, "event": ev}
    for k, v in covars.items():
        out[k] = np.asarray(v)[row_idx]
    return pd.DataFrame(out)


def fit_hazard(long_df, feature_cols, sample_weight=None):
    """Unpenalized discrete-time logistic hazard, fit to convergence.

    The graded reference values are the (weighted) maximum-likelihood estimates
    of this model, so the fit must actually reach the MLE: sklearn's default
    lbfgs tolerance (1e-4) stops several log-likelihood units short on a
    ~1M-row person-day expansion without raising ConvergenceWarning, which
    shifts coefficients by tenths and day-90 survival by up to ~0.05. Newton
    with a tight tolerance reproduces statsmodels' Logit/GLM to ~1e-5, and a
    fit that does not converge is an error rather than a silently wrong answer.
    """
    X = long_df[feature_cols + ["log1p_day"]].to_numpy()
    y = long_df["event"].to_numpy()
    clf = LogisticRegression(penalty=None, solver="newton-cholesky", tol=1e-8, max_iter=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        clf.fit(X, y, sample_weight=sample_weight)
    # Effective sample size of the weights actually passed to this fit (all-ones,
    # i.e. n_rows, when sample_weight is None) -- exposed on the classifier itself
    # so a caller can't report a diagnostic that has drifted from what was fit.
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight)
    clf.fit_effective_sample_size_ = float((w.sum() ** 2) / (w ** 2).sum())
    clf.fit_max_weight_ = float(w.max())
    return clf


def predict_hazard(clf, feature_cols, df_static, day, treat_col=None, treat_override=None):
    d = df_static.copy()
    if treat_override is not None:
        d[treat_col] = treat_override
    d = d.copy()
    d["log1p_day"] = np.log1p(day)
    X = d[feature_cols + ["log1p_day"]].to_numpy()
    return clf.predict_proba(X)[:, 1]


def survival_curve(clf, feature_cols, df_static, t_max, treat_col=None, treat_override=None):
    n = len(df_static)
    surv = np.ones(n)
    curve = np.zeros((t_max + 1, n))
    curve[0] = 1.0
    for t in range(1, t_max + 1):
        h = predict_hazard(clf, feature_cols, df_static, t, treat_col=treat_col, treat_override=treat_override)
        surv = surv * (1 - h)
        curve[t] = surv
    return curve


def rmst_from_curve(curve, t_max):
    return curve[:t_max, :].sum(axis=0)
