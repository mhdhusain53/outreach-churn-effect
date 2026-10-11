Outreach retention program - data dictionary

All files key on customer_id. Day 0 for a customer is the date the outreach
decision was made for that account. The observation window runs from day 0 to
day 180 for every customer.

customers.csv (one row per customer, 15000 rows)
- customer_id
- plan_tier: starter, growth, or scale, as of day 0
- monthly_spend_usd: billed monthly amount as of day 0
- login_freq_pre30d: average weekly logins over the 30 days before day 0
- tenure_days_pre: how many days the account had been subscribed before day 0
- outreach_flag: the latest outreach value in this export. That is not always
  the day-0 assignment; recover the day-0 assignment from assignment_revisions.csv.
  Targeting criteria were not recorded.
- integrity: a per-row checksum over customer_id, plan_tier, monthly_spend_usd,
  login_freq_pre30d, and tenure_days_pre joined in that order with |. It is the
  sum over characters of (1-based position squared times ASCII code), reduced
  modulo a prime. It is not MD5, SHA, or CRC32. Recover the prime from rows
  that already match. Drop a row whose stored value does not recompute. Do not
  include outreach_flag in the join.

assignment_revisions.csv (one or more rows per customer)
- customer_id
- outreach_flag: 1 if this revision says the team reached out, else 0
- published_at_day: when this assignment became known (0 means known as of
  day 0; a positive value is a later revision)
- recorded_at_day: when this row was written to the log
Use the assignment that was known as of day 0, not a later revision, for
confounding checks, the cancellation model, IPCW, and customer_predictions.json.

plan_events.csv (one row per terminal event that actually happened)
- customer_id
- event_type: "cancellation" (fully cancelled the paid subscription) or "downgrade"
  (moved to the free tier)
- event_day: days after day 0 that the event happened
A customer with no row here neither cancelled nor downgraded in the 180-day window.
Once an account downgrades to the free tier, this dataset stops tracking whether that
account later fully cancels - a downgrade row is the last event recorded for that
customer.

support_tickets.csv (one row per support ticket)
- ticket_id: a per-ticket identifier
- customer_id
- ticket_day: days after day 0 that the ticket was filed
Tickets were filed in the first 60 days after day 0. A customer with no rows filed no
tickets in that window. A support-system export glitch caused some customers' ticket
history to be written to this file twice; where that happened, every ticket_id for that
customer appears in two identical rows. Count each ticket_id once per customer.

usage_summary.csv (one row per customer, 15000 rows)
- customer_id
- usage_change_pct_30d: percent change in weekly active usage, comparing the 30 days
  after day 0 to the 30 days before day 0

Modeling conventions, restated here so they hold regardless of which document you're
using: every hazard model fit for these outputs is a discrete-time logistic regression
on person-day records with log(1 + days since day 0) as its sole time-dependent term,
no splines/polynomials/extra day terms, fit by unpenalized maximum likelihood to
convergence (a default solver tolerance or an L2 penalty stops short of the
reference); the inverse-probability-of-censoring weight on a customer's day-t record
is 1 / max(0.04, S_C(t)), where S_C(t) is that customer's fitted probability of not
having downgraded through day t (the product of one minus the fitted downgrade hazard
over days 1..t), so weights cap at 25; the model behind customer_predictions.json is
fit on plan_tier, monthly_spend_usd, and login_freq_pre30d plus the day-0 outreach
assignment only, leaving tenure_days_pre out of that particular fit; ticket count is
standardized once per retained customer (one value per customer_id you keep, by its
mean and standard deviation across those customers), not per person-day row.
