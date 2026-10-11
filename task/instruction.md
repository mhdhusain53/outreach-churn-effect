You have customer-level data at `/app/data/`: `customers.csv`, `plan_events.csv`, `support_tickets.csv`, `usage_summary.csv`, `assignment_revisions.csv`. Read `/app/data/data_dictionary.md` first; it defines columns, timing (day 0, the 180-day window), and two export quirks.

Estimate the causal effect of proactive outreach on time to cancellation as restricted mean survival time (RMST) at 180 days: expected days before cancellation, capped at 180, under outreach minus the same quantity without outreach, for an otherwise identical customer.

Pre-treatment fields in `customers.csv` did not all predict who got outreach; find the ones that did from the data and adjust for them. Some fields were measured after the day-0 assignment; adjusting for those biases the estimate. `plan_events.csv` stops tracking cancellation after a free-tier downgrade; that censoring may depend on the same factors that predict cancellation, so check every post-treatment signal you have, not only ticket volume, and correct if needed.

Two export quirks. `customers.csv` has an `integrity` field on every row; drop mismatches. It is not a named digest or CRC. Join `customer_id`, `plan_tier`, `monthly_spend_usd`, `login_freq_pre30d`, and `tenure_days_pre` in that order with `|` (do not include `outreach_flag`), then take the sum over characters of (1-based position squared times that character's ASCII code), then reduce modulo a prime. Recover the prime from the rows that already match. `assignment_revisions.csv` is the outreach history. The assignment that counts is the one that was already known as of day 0, not a later correction. The `outreach_flag` column on `customers.csv` is the latest value in the export and is not always that day-0 assignment. Confounder checks, the outcome model, and the graded predictions all use the day-0 assignment.

Write four files.

`/app/output/analyze.py` - Python 3 that reads the current `/app/data` files and writes the three JSON files below, using absolute paths. Every graded number must be computed from those files, not baked in as constants. Grading copies `analyze.py` plus every sibling `.py` file in `/app/output/` to a fresh directory and runs it there, so keep any helper module as a plain `.py` file beside it.

`/app/output/adjustment_set.json` - a JSON object with one field:
- `included_covariates`: a list drawn only from `plan_tier`, `monthly_spend_usd`, `login_freq_pre30d`, `tenure_days_pre` - whichever of these you used to adjust for confounding.

`/app/output/effect_estimate.json` - a JSON object with these fields:
- `rmst_horizon_days`: 180
- `rmst_difference_days`: causal RMST difference (outreach minus no outreach), in days
- `rmst_difference_days_scale_tier`: the same quantity, standardized only over customers with `plan_tier == "scale"` as of day 0, using the same fitted outcome model
- `downgrade_hazard_ratio_per_sd_tickets`: hazard ratio (exp of the fitted coefficient) for a one-SD increase in each retained customer's total ticket count from `support_tickets.csv`; fit the downgrade hazard using every covariate that actually predicts downgrades, not just the first you check; standardize ticket count as one value per retained customer, by that retained set's mean and SD, not a per-person-day expansion
- `outcome_model_effective_sample_size`: if you weight person-day records in the cancellation fit by the inverse probability of remaining uncensored, report `(sum of weights)^2 / (sum of squared weights)` over every person-day used
- `max_ipcw_weight`: the largest weight actually applied; the `0.04` floor below should give (or nearly give) `25`

`/app/output/customer_predictions.json` - map each of these 25 `customer_id` values to a number. Fit this file's cancellation model on `plan_tier`, `monthly_spend_usd`, and `login_freq_pre30d` plus the day-0 outreach assignment only (leave `tenure_days_pre` out even if it is in `adjustment_set.json`). That fit uses the same inverse-probability-of-censoring weights, including the 0.04 floor, as the RMST cancellation model; an unweighted hazard is not the model for this file. Report each customer's predicted probability of still being active at day 90, at that customer's own day-0 assignment, not a counterfactual:
`cust_00409, cust_00434, cust_00488, cust_00520, cust_01424, cust_01535, cust_01679, cust_01824, cust_02286, cust_03582, cust_03657, cust_03811, cust_04012, cust_04506, cust_06912, cust_08279, cust_08935, cust_09674, cust_09863, cust_10476, cust_11087, cust_12066, cust_12135, cust_12149, cust_14617`

Sanity check (`cust_00001` is not one of the 25): the same model gives `cust_00001` a day-90 survival of about `0.25` on the data you were given; if yours is far off, the pipeline is wrong before the graded customers.

Modeling conventions, so numbers are comparable to a reference: every hazard model for these outputs is a discrete-time logistic regression on person-day records with `log(1 + days since day 0)` as the sole time-dependent term (no splines, polynomials, or extra day terms), fit by unpenalized maximum likelihood to convergence - a default solver tolerance or an L2 penalty stops short of the reference. The weight on a customer's day-`t` record is `1 / max(0.04, S_C(t))`, where `S_C(t)` is that customer's fitted probability of not having downgraded through day `t` (the product of one minus the downgrade hazard over days 1 to `t`), so weights cap at 25.

All seven fields in `adjustment_set.json` and `effect_estimate.json`, plus all 25 values in `customer_predictions.json`, must be valid JSON, with every numeric field as a JSON number.
