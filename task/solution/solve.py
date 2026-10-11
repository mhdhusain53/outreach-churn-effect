import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from surv_lib import (
    expand_person_days,
    fit_hazard,
    survival_curve,
    rmst_from_curve,
    intact_customer_ids,
    day0_outreach_by_id,
)

DATA_DIR = "/app/data"
OUT_DIR = "/app/output"
T_MAX = 180


def standardize(x):
    return (x - x.mean()) / x.std()


def build_frame():
    keep = intact_customer_ids(f"{DATA_DIR}/customers.csv")
    day0 = day0_outreach_by_id(f"{DATA_DIR}/assignment_revisions.csv")
    customers = pd.read_csv(f"{DATA_DIR}/customers.csv")
    customers = customers[customers["customer_id"].isin(keep)].copy()
    customers["outreach_flag"] = customers["customer_id"].map(day0)
    customers = customers.dropna(subset=["outreach_flag"])
    customers["outreach_flag"] = customers["outreach_flag"].astype(int)
    events = pd.read_csv(f"{DATA_DIR}/plan_events.csv")
    tickets = pd.read_csv(f"{DATA_DIR}/support_tickets.csv")
    usage = pd.read_csv(f"{DATA_DIR}/usage_summary.csv")

    tickets = tickets.drop_duplicates("ticket_id")
    ticket_counts = tickets.groupby("customer_id").size().rename("tickets_post_count")
    df = customers.merge(ticket_counts, on="customer_id", how="left")
    df["tickets_post_count"] = df["tickets_post_count"].fillna(0)
    df = df.merge(usage, on="customer_id", how="left")

    # one terminal event per customer, if any; no row -> administrative censor at T_MAX
    ev = events.sort_values("event_day").drop_duplicates("customer_id", keep="first")
    df = df.merge(ev, on="customer_id", how="left")
    df["obs_time"] = df["event_day"].fillna(T_MAX).astype(int)
    df["event"] = (df["event_type"] == "cancellation").astype(int)
    df["is_downgrade"] = (df["event_type"] == "downgrade").astype(int)

    df["spend_s"] = standardize(df["monthly_spend_usd"])
    df["login_s"] = standardize(df["login_freq_pre30d"])
    df["tenure_s"] = standardize(df["tenure_days_pre"])
    df["tickets_s"] = standardize(df["tickets_post_count"])
    df["usage_s"] = standardize(df["usage_change_pct_30d"])
    df["plan_starter"] = (df["plan_tier"] == "starter").astype(int)
    df["plan_scale"] = (df["plan_tier"] == "scale").astype(int)
    return df.reset_index(drop=True)


FIELD_TO_OUTCOME_COLS = {
    "plan_tier": ["plan_starter", "plan_scale"],
    "monthly_spend_usd": ["spend_s"],
    "login_freq_pre30d": ["login_s"],
    "tenure_days_pre": ["tenure_s"],
}


def standardized_mean_diff(x, treat):
    t, c = x[treat == 1], x[treat == 0]
    pooled = np.sqrt((t.std() ** 2 + c.std() ** 2) / 2)
    return abs((t.mean() - c.mean()) / pooled) if pooled > 0 else 0.0


def compute_adjustment_set(df, threshold=0.1):
    """A pre-treatment field belongs in the adjustment set if it's imbalanced
    between the outreach arms (the standard covariate-balance diagnostic for
    detecting confounding), not because a targeting rule says so -- targeting
    criteria were never disclosed to the agent, so the oracle doesn't get to
    use them either."""
    treat = df["outreach_flag"].to_numpy()
    included = []
    if max(standardized_mean_diff(df["plan_starter"], treat),
           standardized_mean_diff(df["plan_scale"], treat)) >= threshold:
        included.append("plan_tier")
    if standardized_mean_diff(df["monthly_spend_usd"].to_numpy(), treat) >= threshold:
        included.append("monthly_spend_usd")
    if standardized_mean_diff(df["login_freq_pre30d"].to_numpy(), treat) >= threshold:
        included.append("login_freq_pre30d")
    if standardized_mean_diff(df["tenure_days_pre"].to_numpy(), treat) >= threshold:
        included.append("tenure_days_pre")
    return included


def fit_censoring_model(df, cens_features):
    long_c = expand_person_days(
        df["customer_id"].to_numpy(), df["obs_time"].to_numpy(), df["is_downgrade"].to_numpy(),
        {c: df[c].to_numpy() for c in cens_features},
    )
    long_c["log1p_day"] = np.log1p(long_c["day"])
    clf_c = fit_hazard(long_c, cens_features)
    return clf_c


def ipcw_weights(df, clf_c, cens_features):
    surv_c_curve = survival_curve(clf_c, cens_features, df, t_max=T_MAX)
    long_o_days = df["obs_time"].to_numpy().astype(int)
    id_to_col = {cid: j for j, cid in enumerate(df["customer_id"].to_numpy())}
    return surv_c_curve, id_to_col


def fit_ipcw_weights(df, cens_features):
    clf_c = fit_censoring_model(df, cens_features)
    surv_c_curve, id_to_col = ipcw_weights(df, clf_c, cens_features)
    return clf_c, surv_c_curve, id_to_col


def fit_weighted_outcome_model(df, outcome_features, surv_c_curve, id_to_col):
    long_o = expand_person_days(
        df["customer_id"].to_numpy(), df["obs_time"].to_numpy(), df["event"].to_numpy(),
        {c: df[c].to_numpy() for c in outcome_features},
    )
    long_o["log1p_day"] = np.log1p(long_o["day"])
    cols = long_o["id"].map(id_to_col).to_numpy()
    days = long_o["day"].to_numpy()
    s_c_t = np.clip(surv_c_curve[days, cols], 0.04, 1.0)
    sw = 1.0 / s_c_t
    return fit_hazard(long_o, outcome_features, sample_weight=sw)


def gformula_rmst_diff(df, outcome_features, cens_features):
    clf_c, surv_c_curve, id_to_col = fit_ipcw_weights(df, cens_features)
    clf_o = fit_weighted_outcome_model(df, outcome_features, surv_c_curve, id_to_col)

    curve1 = survival_curve(clf_o, outcome_features, df, t_max=T_MAX, treat_col="outreach_flag", treat_override=1)
    curve0 = survival_curve(clf_o, outcome_features, df, t_max=T_MAX, treat_col="outreach_flag", treat_override=0)
    rmst1 = rmst_from_curve(curve1, T_MAX).mean()
    rmst0 = rmst_from_curve(curve0, T_MAX).mean()

    # Read the effective sample size back off the fitted outcome model itself (set
    # from whatever sample_weight it was actually fit with), not from the sw array
    # computed above -- so this diagnostic can't drift from what was really fit.
    ess = clf_o.fit_effective_sample_size_
    max_w = clf_o.fit_max_weight_
    return rmst1 - rmst0, ess, max_w, clf_c, clf_o, surv_c_curve, id_to_col


PREDICTION_CUSTOMER_IDS = [
    "cust_00409", "cust_00434", "cust_00488", "cust_00520", "cust_01424",
    "cust_01535", "cust_01679", "cust_01824", "cust_02286", "cust_03582",
    "cust_03657", "cust_03811", "cust_04012", "cust_04506", "cust_06912",
    "cust_08279", "cust_08935", "cust_09674", "cust_09863", "cust_10476",
    "cust_11087", "cust_12066", "cust_12135", "cust_12149", "cust_14617",
]
PREDICTION_DAY = 90


def customer_survival_predictions(df, outcome_features, clf_o):
    """Predicted probability of still being active at PREDICTION_DAY, at each
    customer's own observed outreach_flag (no counterfactual override) -- this
    depends on the customer's specific fitted covariates and can't be produced
    without actually running the fitted outcome model."""
    curve = survival_curve(clf_o, outcome_features, df, t_max=PREDICTION_DAY)
    id_to_row = {cid: i for i, cid in enumerate(df["customer_id"])}
    return {
        cid: round(float(curve[PREDICTION_DAY, id_to_row[cid]]), 4)
        for cid in PREDICTION_CUSTOMER_IDS
    }


REQUIRED_FIELDS = ["plan_tier", "monthly_spend_usd", "login_freq_pre30d"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build_frame()

    adjustment_set = compute_adjustment_set(df)
    adjustment_cols = sum((FIELD_TO_OUTCOME_COLS[f] for f in adjustment_set), [])
    outcome_features = adjustment_cols + ["outreach_flag"]
    cens_features = adjustment_cols + ["outreach_flag", "tickets_s", "usage_s"]

    rmst_diff, ess, max_w, clf_c, clf_o, surv_c_curve, id_to_col = gformula_rmst_diff(
        df, outcome_features, cens_features
    )
    tickets_coef = clf_c.coef_[0][cens_features.index("tickets_s")]
    tickets_hr = float(np.exp(tickets_coef))

    # Same fitted outcome model, standardized only over the scale-tier subpopulation
    # instead of everyone -- a separate g-formula RMST difference, not a re-use of
    # the pooled number above.
    scale_mask = (df["plan_tier"] == "scale").to_numpy()
    curve1_all = survival_curve(clf_o, outcome_features, df, t_max=T_MAX, treat_col="outreach_flag", treat_override=1)
    curve0_all = survival_curve(clf_o, outcome_features, df, t_max=T_MAX, treat_col="outreach_flag", treat_override=0)
    rmst1_scale = rmst_from_curve(curve1_all[:, scale_mask], T_MAX).mean()
    rmst0_scale = rmst_from_curve(curve0_all[:, scale_mask], T_MAX).mean()
    rmst_diff_scale = rmst1_scale - rmst0_scale

    # customer_predictions.json always comes from a model fit on the required
    # confounders only (plan_tier, monthly_spend_usd, login_freq_pre30d) plus
    # outreach_flag, regardless of whether tenure_days_pre made it into the
    # adjustment set used above -- instruction.md pins this so the predictions
    # are comparable across submissions no matter what they decided about that
    # optional field.
    required_cols = sum((FIELD_TO_OUTCOME_COLS[f] for f in REQUIRED_FIELDS), [])
    pred_outcome_features = required_cols + ["outreach_flag"]
    clf_o_pred = fit_weighted_outcome_model(df, pred_outcome_features, surv_c_curve, id_to_col)
    predictions = customer_survival_predictions(df, pred_outcome_features, clf_o_pred)

    with open(f"{OUT_DIR}/adjustment_set.json", "w") as f:
        json.dump({"included_covariates": adjustment_set}, f, indent=2)

    with open(f"{OUT_DIR}/effect_estimate.json", "w") as f:
        json.dump({
            "rmst_horizon_days": T_MAX,
            "rmst_difference_days": round(float(rmst_diff), 3),
            "rmst_difference_days_scale_tier": round(float(rmst_diff_scale), 3),
            "downgrade_hazard_ratio_per_sd_tickets": round(tickets_hr, 4),
            "outcome_model_effective_sample_size": round(float(ess), 1),
            "max_ipcw_weight": round(float(max_w), 3),
        }, f, indent=2)

    with open(f"{OUT_DIR}/customer_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"rmst_diff={rmst_diff:.2f} tickets_hr={tickets_hr:.3f} ess={ess:.0f} max_w={max_w:.2f}")


if __name__ == "__main__":
    main()
