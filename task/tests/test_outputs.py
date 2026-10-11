"""
Grades JSON written after tests restore verifier-owned held-out tables over
/app/data and re-run /app/output/analyze.py. Day-0 vs latest outreach assignment
and dropping integrity mismatches are graded on separate checks so fixing one
naive method cannot satisfy the rest.
"""
import csv
import hashlib
import json
import math
import os

import pytest

OUTPUT_DIR = "/app/output"
DATA_DIR = "/app/data"
INTEGRITY_MOD = 7919
HELDOUT_CUSTOMERS_SHA256 = "3ebc626dccbe1ec3af32dfcfa22ea703649b9d37368b16dbfab1385c86a7cf66"

REQUIRED_COVARIATES = {"plan_tier", "monthly_spend_usd", "login_freq_pre30d"}
ALLOWED_COVARIATES = REQUIRED_COVARIATES | {"tenure_days_pre"}

RMST_HORIZON_DAYS = 180

# Every band below was calibrated on the held-out export with each hazard model
# fit by unpenalized maximum likelihood to convergence (newton-cholesky /
# statsmodels agree to ~1e-5), after dropping integrity mismatches and using the
# day-0 assignment. Compliant implementations -- exact-MLE solvers from any
# library, tenure_days_pre in or out of the adjustment set, outreach_flag or
# tenure in or out of the downgrade model -- all land at 52.97-53.08. Using the
# latest customers.csv outreach_flag lands near 40.3, trusting every row without
# recomputing integrity near 40.2, an unadjusted arm comparison near 72, and a
# default-tolerance (unconverged) solver near 55.2.
RMST_LOWER = 50.0
RMST_UPPER = 56.0

# Same population, standardized only over customers on the scale plan instead of
# everyone: compliant implementations land at 28.8-28.9. Reusing the pooled value
# here (~53) misses by a wide margin; the latest outreach flag lands near 19.9,
# trusting every row near 26.4, and an unconverged solver near 26.5.
RMST_SCALE_LOWER = 27.0
RMST_SCALE_UPPER = 30.5

# Held-out, integrity-valid rows, converged fit: the downgrade model with every
# real predictor (plan, spend, logins, outreach, tickets, usage change) lands at
# 2.07 with or without tenure/outreach in it. Omitting usage_change_pct_30d lands
# at 1.91, skipping ticket_id deduplication at 1.65, the latest outreach flag at
# 1.68, and trusting mismatched integrity rows restores a ~7.5 ratio.
HR_LOWER = 1.97
HR_UPPER = 2.30

# instruction.md discloses this directly: with the 0.04 floor applied correctly,
# the largest weight actually used should come out at (or very near) 25 = 1/0.04.
# This is what makes the floor value a real, enforced convention rather than a
# number that happens to sit in the instructions unchecked -- a submission that
# uses a different floor (0.03 -> ~33.3, 0.1 -> ~10, or no floor at all) lands
# well outside this band even when every other field happens to land inside its
# own tolerance.
MAX_WEIGHT_LOWER = 23.0
MAX_WEIGHT_UPPER = 27.0

# An outcome-hazard fit with no informative-censoring correction assigns every
# person-day record weight 1, so its effective sample size equals the raw total
# person-days exactly. Any genuine inverse-probability-of-censoring weighting on
# this dataset measurably shrinks that below the raw total (the correct pipeline
# lands at 0.920-0.924 of it; a downgrade model missing usage change or logins
# lands near 0.96); this band brackets real reweighting on either side while
# excluding the unweighted value (ratio 1.0) by a comfortable margin.
ESS_RATIO_MIN = 0.30
ESS_RATIO_MAX = 0.95

# Predicted probability of still being active at day 90, at each customer's own
# day-0 outreach assignment, from the reference model instruction.md pins: a
# discrete-time logistic hazard on plan_tier, monthly_spend_usd, login_freq_pre30d
# and the day-0 assignment with log(1+day) as the sole time term, IPCW-weighted
# with the 0.04 floor, fit by unpenalized maximum likelihood to convergence.
# Because the reference is the MLE of a fully specified model, any correct
# implementation reproduces it: sklearn (tight tolerance, no penalty) and
# statsmodels Logit/GLM agree with these values to within 0.0005, as do the
# instruction-permitted choices elsewhere in the pipeline (tenure in or out of
# the adjustment set, outreach_flag or tenure in or out of the downgrade model,
# the G_C(t-1) weight index instead of the stated G_C(t)). Each value depends
# on that customer's real covariates run through the fitted model, so a
# submission that hardcodes plausible aggregate numbers has no way to get more
# than a few of these 25 right by chance.
REFERENCE_PREDICTIONS = {
    "cust_00409": 0.9897, "cust_00434": 0.5983, "cust_00488": 0.9294,
    "cust_00520": 0.8608, "cust_01424": 0.1150, "cust_01535": 0.1731,
    "cust_01679": 0.9617, "cust_01824": 0.3727, "cust_02286": 0.1769,
    "cust_03582": 0.9670, "cust_03657": 0.0899, "cust_03811": 0.5368,
    "cust_04012": 0.2775, "cust_04506": 0.5694, "cust_06912": 0.1061,
    "cust_08279": 0.6847, "cust_08935": 0.5899, "cust_09674": 0.6229,
    "cust_09863": 0.7510, "cust_10476": 0.6072, "cust_11087": 0.9919,
    "cust_12066": 0.9363, "cust_12135": 0.9080, "cust_12149": 0.1198,
    "cust_14617": 0.2664,
}
# Tolerance is ~20x the spread across compliant implementations (0.0005) and
# still rejects each pipeline error this file is the only check for: fitting
# tenure_days_pre into this model (12/25 within tolerance), using the latest
# outreach flag instead of the day-0 assignment (10/25), trusting integrity
# mismatches (1/25), and a solver that stops at sklearn's default tolerance
# rather than converging (18/25; its coefficients differ by tenths). Skipping
# the censoring weights or changing the floor barely moves these predictions
# (<= 0.010), so those are enforced by the ESS and max-weight checks above, not
# here. 24 of 25 leaves one value of slack for rounding at a boundary.
PREDICTION_TOLERANCE = 0.012
PREDICTION_MIN_MATCHES = 24


def _assert_real_output_file(path):
    """Reject a symlinked leaf or any symlinked parent, including /app and
    /app/output, so a parent-directory containment bypass cannot redirect grading."""
    if not os.path.isfile(path):
        pytest.fail(f"{path} is missing")
    cursor = os.path.abspath(path)
    seen = set()
    while cursor not in seen:
        seen.add(cursor)
        if os.path.islink(cursor):
            pytest.fail(f"symlink in output path: {cursor}")
        parent = os.path.dirname(cursor)
        if parent == cursor:
            break
        cursor = parent
    real_out = os.path.realpath(OUTPUT_DIR)
    real_path = os.path.realpath(path)
    try:
        common = os.path.commonpath([real_out, real_path])
    except ValueError:
        pytest.fail(f"{path} is not under {OUTPUT_DIR}")
    if common != real_out:
        pytest.fail(f"{path} is not under {OUTPUT_DIR}")


def _load(name):
    path = f"{OUTPUT_DIR}/{name}"
    _assert_real_output_file(path)
    with open(path) as f:
        return json.load(f)


def _finite_number(val, label):
    assert isinstance(val, (int, float)) and not isinstance(val, bool), f"{label} is not numeric: {val!r}"
    assert math.isfinite(val), f"{label} is not finite: {val!r}"
    return val


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _customer_integrity(row):
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
    return acc % INTEGRITY_MOD


def _intact_customer_ids(path):
    ok = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["integrity"]) == _customer_integrity(row):
                ok.add(row["customer_id"])
    return ok


def test_analyze_script_is_a_real_file():
    """analyze.py must exist as a regular file so the verifier can re-run it."""
    path = f"{OUTPUT_DIR}/analyze.py"
    _assert_real_output_file(path)


def test_heldout_customers_were_restored():
    """Grading must run on the verifier-owned held-out customers.csv, not the
    agent-visible calibration extract."""
    path = f"{DATA_DIR}/customers.csv"
    assert os.path.isfile(path) and not os.path.islink(path)
    assert _sha256_file(path) == HELDOUT_CUSTOMERS_SHA256


def _raw_total_person_days():
    """Person-days over integrity-valid customers only, from /app/data after restore."""
    customers = sorted(_intact_customer_ids(f"{DATA_DIR}/customers.csv"))

    first_event_day = {}
    with open(f"{DATA_DIR}/plan_events.csv") as f:
        for row in csv.DictReader(f):
            cid = row["customer_id"]
            day = int(row["event_day"])
            if cid not in first_event_day or day < first_event_day[cid]:
                first_event_day[cid] = day

    return sum(first_event_day.get(cid, RMST_HORIZON_DAYS) for cid in customers)


def test_adjustment_set_file_present_and_well_formed():
    """adjustment_set.json exists and is a JSON object with an included_covariates list."""
    data = _load("adjustment_set.json")
    assert isinstance(data, dict)
    assert "included_covariates" in data
    assert isinstance(data["included_covariates"], list)
    assert all(isinstance(c, str) for c in data["included_covariates"])


def test_adjustment_set_excludes_post_treatment_variables():
    """Only customers.csv baseline field names are allowed; the post-treatment ticket
    count and usage-change fields (or any other name) must not appear."""
    data = _load("adjustment_set.json")
    chosen = set(data["included_covariates"])
    unknown = chosen - ALLOWED_COVARIATES
    assert not unknown, f"adjustment set includes disallowed/post-treatment field(s): {unknown}"


def test_adjustment_set_includes_all_confounders():
    """All three fields that drove day-0 outreach targeting (plan_tier,
    monthly_spend_usd, login_freq_pre30d) must be in the adjustment set;
    tenure_days_pre is optional. Using the latest customers.csv outreach_flag
    as if it were the day-0 assignment makes login_freq_pre30d look balanced
    on the held-out export and omits that confounder."""
    data = _load("adjustment_set.json")
    chosen = set(data["included_covariates"])
    missing = REQUIRED_COVARIATES - chosen
    assert not missing, f"adjustment set is missing required confounder(s): {missing}"


def test_effect_estimate_file_present_and_well_formed():
    """effect_estimate.json exists with the required numeric fields."""
    data = _load("effect_estimate.json")
    assert isinstance(data, dict)
    for key in (
        "rmst_horizon_days",
        "rmst_difference_days",
        "rmst_difference_days_scale_tier",
        "downgrade_hazard_ratio_per_sd_tickets",
        "outcome_model_effective_sample_size",
        "max_ipcw_weight",
    ):
        assert key in data, f"missing field: {key}"
    _finite_number(data["rmst_difference_days"], "rmst_difference_days")
    _finite_number(data["rmst_difference_days_scale_tier"], "rmst_difference_days_scale_tier")
    _finite_number(data["downgrade_hazard_ratio_per_sd_tickets"], "downgrade_hazard_ratio_per_sd_tickets")
    _finite_number(data["outcome_model_effective_sample_size"], "outcome_model_effective_sample_size")
    _finite_number(data["max_ipcw_weight"], "max_ipcw_weight")


def test_effect_estimate_horizon_matches_instruction():
    """The reported horizon must be the 180-day window stated in the instructions."""
    data = _load("effect_estimate.json")
    assert data["rmst_horizon_days"] == RMST_HORIZON_DAYS


def test_rmst_difference_within_tolerance():
    """rmst_difference_days must be within RMST_BAND days of the value a correctly
    adjusted, censoring-corrected estimator recovers. An estimate built on the raw,
    unadjusted arm comparison (ignoring confounding and informative censoring) misses
    this band by a wide margin."""
    data = _load("effect_estimate.json")
    val = _finite_number(data["rmst_difference_days"], "rmst_difference_days")
    assert RMST_LOWER <= val <= RMST_UPPER, (
        f"rmst_difference_days={val} outside [{RMST_LOWER}, {RMST_UPPER}]"
    )


def test_rmst_difference_scale_tier_within_tolerance():
    """rmst_difference_days_scale_tier must be within its own band -- a different,
    separately-computed number from rmst_difference_days, not a copy of it. Outreach's
    effect on cancellation is real but substantially smaller among scale-plan
    customers than in the population overall; a submission that reports the pooled
    number here (or restandardizes over the wrong subgroup) misses this band by a
    wide margin even when rmst_difference_days itself is correct."""
    data = _load("effect_estimate.json")
    val = _finite_number(data["rmst_difference_days_scale_tier"], "rmst_difference_days_scale_tier")
    assert RMST_SCALE_LOWER <= val <= RMST_SCALE_UPPER, (
        f"rmst_difference_days_scale_tier={val} outside [{RMST_SCALE_LOWER}, {RMST_SCALE_UPPER}]"
    )


def test_downgrade_hazard_ratio_within_tolerance():
    """downgrade_hazard_ratio_per_sd_tickets must be within [HR_LOWER, HR_UPPER] of the
    value a correctly specified censoring-hazard model recovers. This is only
    reproducible by actually fitting a downgrade-hazard model that uses ticket
    activity alongside the other real downgrade predictors; an agent that skips
    informative-censoring detection has no principled way to land in this band,
    and one that forgets to deduplicate support_tickets.csv by ticket_id, or
    leaves usage_change_pct_30d out of the model, lands measurably below HR_LOWER."""
    data = _load("effect_estimate.json")
    val = _finite_number(data["downgrade_hazard_ratio_per_sd_tickets"], "downgrade_hazard_ratio_per_sd_tickets")
    assert HR_LOWER <= val <= HR_UPPER, (
        f"downgrade_hazard_ratio_per_sd_tickets={val} outside [{HR_LOWER}, {HR_UPPER}]"
    )


def test_max_ipcw_weight_reflects_the_stated_floor():
    """max_ipcw_weight must be close to 25 (= 1 / 0.04), the cap instruction.md
    states explicitly follows from the 0.04 floor. A submission that applies a
    different floor value, or no floor at all, produces a different maximum
    weight -- this is what makes the disclosed 0.04 convention an enforced
    requirement rather than an unchecked number in the instructions."""
    data = _load("effect_estimate.json")
    val = _finite_number(data["max_ipcw_weight"], "max_ipcw_weight")
    assert MAX_WEIGHT_LOWER <= val <= MAX_WEIGHT_UPPER, (
        f"max_ipcw_weight={val} outside [{MAX_WEIGHT_LOWER}, {MAX_WEIGHT_UPPER}]"
    )


def test_outcome_model_shows_real_censoring_reweighting():
    """outcome_model_effective_sample_size must be measurably below the raw total
    person-days in the data (recomputed here from customers.csv and plan_events.csv,
    not from any agent output). An outcome hazard fit that never applies inverse-
    probability-of-censoring weights - even one built on the exactly right adjustment
    set - has every weight equal to 1, so its effective sample size equals the raw
    total exactly; this catches that silently unweighted case even when the resulting
    rmst_difference_days happens to land inside its own tolerance band."""
    data = _load("effect_estimate.json")
    val = _finite_number(data["outcome_model_effective_sample_size"], "outcome_model_effective_sample_size")
    raw_total = _raw_total_person_days()
    ratio = val / raw_total
    assert ESS_RATIO_MIN <= ratio <= ESS_RATIO_MAX, (
        f"outcome_model_effective_sample_size={val} is ratio {ratio:.3f} of the raw "
        f"total person-days ({raw_total}); expected within [{ESS_RATIO_MIN}, {ESS_RATIO_MAX}] "
        f"(1.0 means no censoring reweighting was applied)"
    )


def test_customer_predictions_match_reference_model():
    """customer_predictions.json must map each of the 25 named customer_ids to a
    predicted probability (at that customer's own day-0 outreach assignment) of
    still being active at day 90, within PREDICTION_TOLERANCE of the converged
    maximum-likelihood fit of the model instruction.md specifies, for at least
    PREDICTION_MIN_MATCHES of them. Each value depends on that specific
    customer's covariates run through a correctly adjusted, converged hazard
    model on integrity-valid rows - there's no way to get most of these right
    without actually fitting one."""
    data = _load("customer_predictions.json")
    assert isinstance(data, dict)
    matches = 0
    for cid, ref in REFERENCE_PREDICTIONS.items():
        assert cid in data, f"missing prediction for {cid}"
        val = _finite_number(data[cid], f"{cid} prediction")
        assert 0.0 <= val <= 1.0, f"{cid} prediction {val} is not a probability"
        if abs(val - ref) <= PREDICTION_TOLERANCE:
            matches += 1
    assert matches >= PREDICTION_MIN_MATCHES, (
        f"only {matches}/{len(REFERENCE_PREDICTIONS)} customer predictions were within "
        f"{PREDICTION_TOLERANCE} of the reference model; need at least {PREDICTION_MIN_MATCHES}"
    )


def test_adjustment_set_and_effect_estimate_both_present():
    """Doing nothing (no output files) must fail outright."""
    for name in ("adjustment_set.json", "effect_estimate.json", "customer_predictions.json"):
        try:
            _load(name)
        except FileNotFoundError:
            pytest.fail(f"{OUTPUT_DIR}/{name} was not produced")
