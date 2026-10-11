# Outreach churn effect

Estimate the causal effect of a retention outreach program on time to cancellation from
subscription logs that are both confounded and informatively censored.

## Approach

Merge `customers.csv`, `plan_events.csv`, `support_tickets.csv`, and `usage_summary.csv`. Use the
data dictionary's timing information to separate pre-treatment fields (real confounders,
belong in the adjustment set) from post-treatment fields (a collider and a mediator, must
stay out of the outcome model). Detect that downgrade timing depends on the same factors
that predict cancellation, correct for it with inverse-probability-of-censoring weighting,
and standardize (g-formula) to get the causal RMST difference at a 180-day horizon.

## Layout

```
task/
  instruction.md          # Problem statement
  environment/            # Docker image + input data
  solution/               # Reference solution
  tests/                  # Graders and held-out fixtures
  task.toml               # Task metadata
```

## Environment

Python 3.13 with numpy, pandas, scikit-learn, scipy, statsmodels, and lifelines. Five CSVs
and a data dictionary are copied into `/app/data`.

Pinned packages live in `requirements.txt` (same versions as the Docker image).

## Quick start

```bash
# Build the analysis image
docker build -t outreach-churn-effect task/environment

# Or install deps locally (Python 3.13 recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Reference solution entrypoint:

```bash
# Inside the container (or with data available under /app/data)
python task/solution/solve.py
```

Check that held-out fixture digests still match `restore_heldout.py`:

```bash
make verify-fixtures
# or: python scripts/verify_fixtures.py
```

## Verification

`adjustment_set.json` is checked by required-include / forbidden-exclude set membership.
`effect_estimate.json` is checked on five numbers against bands calibrated on the held-out
export. `customer_predictions.json` is compared to the converged MLE of the model pinned in
`instruction.md`. Grading re-runs `analyze.py` (with sibling helper modules) from a fresh
directory as an unprivileged user on the held-out tables.
