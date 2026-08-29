# Feature dictionary

59 modelling features (48 numeric, 11 categorical). The baseline model sees 7 of them; the improved model sees all of them.

**Information window** is the guarantee that matters. `as-at t` uses only the reporting month's own record; `t-k..t` is a backward-looking window ending at month t inclusive; `month t` is cross-sectional across loans within the same month. No feature reads a month after t. That claim is enforced, not asserted: `features.assert_no_leaky_features` hard-fails on any label-derived column, and `tests/test_leakage_controls.py` rebuilds the matrix on a panel truncated after month t and requires every rolling feature at t to be unchanged.

Rolling windows are computed on a gap-free monthly grid, so a "3-month window" is three *calendar* months even where the panel is missing a month.

## static (14)

_Fixed at origination; identical for every month of a loan's life._

| feature              | information_window   | dtype       | baseline   |   % present | definition                                     |
|:---------------------|:---------------------|:------------|:-----------|------------:|:-----------------------------------------------|
| credit_score         | origination          | numeric     | False      |         100 | Borrower FICO at origination (500-850).        |
| credit_score_band    | origination          | categorical | True       |         100 | Credit score bucket.                           |
| dti                  | origination          | numeric     | False      |         100 | Debt-to-income at origination, percent.        |
| dti_band             | origination          | categorical | False      |         100 | DTI bucket.                                    |
| interest_rate        | origination          | numeric     | True       |         100 | Annual fixed note rate, percent.               |
| loan_purpose         | origination          | categorical | False      |         100 | Purchase, rate/term refinance, or cash-out.    |
| ltv                  | origination          | numeric     | False      |         100 | Loan-to-value at origination, percent.         |
| ltv_band             | origination          | categorical | False      |         100 | LTV bucket.                                    |
| occupancy_type       | origination          | categorical | False      |         100 | Primary residence, investment, or second home. |
| original_balance     | origination          | numeric     | False      |         100 | Unpaid principal balance at origination, USD.  |
| original_term_months | origination          | numeric     | False      |         100 | Contractual term at origination.               |
| property_type        | origination          | categorical | False      |         100 | Single-family, condominium, or PUD.            |
| state                | origination          | categorical | False      |         100 | US state of the mortgaged property.            |
| vintage_year         | origination          | numeric     | False      |         100 | Calendar year of origination.                  |

## contemporaneous (17)

_The state of the loan as at the reporting month._

| feature               | information_window   | dtype       | baseline   |   % present | definition                                                                                                                                              |
|:----------------------|:---------------------|:------------|:-----------|------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| balance_gap_abs       | as-at t              | numeric     | False      |         100 | current_balance - expected_balance, in USD.                                                                                                             |
| balance_ratio         | as-at t              | numeric     | True       |         100 | current_balance / original_balance: how much principal is left.                                                                                         |
| balance_vs_expected   | as-at t              | numeric     | False      |         100 | current_balance / expected_balance. Below 1 means paying down faster than schedule; a large deviation is also the Balance Discrepancy defect signature. |
| current_balance       | as-at t              | numeric     | True       |         100 | Unpaid principal balance, USD.                                                                                                                          |
| current_status        | as-at t              | categorical | True       |         100 | Performance state: Current / 30 / 60 / 90-DPD.                                                                                                          |
| days_past_due         | as-at t              | numeric     | True       |         100 | Days delinquent at the reporting month.                                                                                                                 |
| doc_status_ordinal    | as-at t              | numeric     | False      |         100 | document_status encoded Complete < Pending < Missing.                                                                                                   |
| document_status       | as-at t              | categorical | False      |         100 | Document file completeness.                                                                                                                             |
| is_delinquent         | as-at t              | numeric     | False      |         100 | 1 where days_past_due > 0.                                                                                                                              |
| loan_age_months       | as-at t              | numeric     | True       |         100 | Months elapsed since origination.                                                                                                                       |
| month_cos             | as-at t              | numeric     | False      |         100 | Second component of the cyclical month encoding.                                                                                                        |
| month_sin             | as-at t              | numeric     | False      |         100 | Cyclical encoding of the calendar month, so December and January are adjacent.                                                                          |
| remaining_term_months | as-at t              | numeric     | False      |         100 | Contractual months remaining.                                                                                                                           |
| servicer_name         | as-at t              | categorical | False      |         100 | Institution servicing the loan this month.                                                                                                              |
| source_system         | as-at t              | categorical | False      |         100 | System of record that produced the row.                                                                                                                 |
| status_ordinal        | as-at t              | numeric     | False      |         100 | current_status encoded worst-last, so a tree can split on severity.                                                                                     |
| term_progress         | as-at t              | numeric     | False      |         100 | loan_age_months / original_term_months: position in the amortisation schedule.                                                                          |

## cross-sectional (2)

_The loan's position relative to every other loan reporting that month._

| feature                 | information_window   | dtype   | baseline   |   % present | definition                                                                                               |
|:------------------------|:---------------------|:--------|:-----------|------------:|:---------------------------------------------------------------------------------------------------------|
| balance_pctile_in_month | month t              | numeric | False      |         100 | Percentile rank of current_balance within month t.                                                       |
| rate_spread             | month t              | numeric | False      |         100 | interest_rate minus the month's median rate: the refinance incentive, and the main driver of prepayment. |

## rolling (25)

_Derived from the loan's own history up to and including month t._

| feature                    | information_window   | dtype   | baseline   |   % present | definition                                                                         |
|:---------------------------|:---------------------|:--------|:-----------|------------:|:-----------------------------------------------------------------------------------|
| delinquent_months_to_date  | 0..t                 | numeric | False      |      100    | Count of months with days_past_due > 0 so far.                                     |
| doc_status_changes_to_date | 0..t                 | numeric | False      |      100    | Count of document-status changes so far.                                           |
| dpd_delta_1m               | t-1..t               | numeric | False      |       95.25 | Change in days_past_due between month t-1 and month t.                             |
| dpd_delta_3m               | t-3..t               | numeric | False      |       88.12 | Change in days_past_due between month t-3 and month t.                             |
| dpd_delta_6m               | t-6..t               | numeric | False      |       77.99 | Change in days_past_due between month t-6 and month t.                             |
| dpd_lag_1m                 | t-1..t               | numeric | False      |       95.25 | days_past_due as at month t-1.                                                     |
| dpd_lag_3m                 | t-3..t               | numeric | False      |       88.12 | days_past_due as at month t-3.                                                     |
| dpd_lag_6m                 | t-6..t               | numeric | False      |       77.99 | days_past_due as at month t-6.                                                     |
| dpd_max_12m                | t-12..t              | numeric | False      |      100    | Maximum days_past_due over the last 12 calendar months, inclusive of t.            |
| dpd_max_3m                 | t-3..t               | numeric | False      |      100    | Maximum days_past_due over the last 3 calendar months, inclusive of t.             |
| dpd_max_6m                 | t-6..t               | numeric | False      |      100    | Maximum days_past_due over the last 6 calendar months, inclusive of t.             |
| dpd_mean_12m               | t-12..t              | numeric | False      |      100    | Mean days_past_due over the last 12 calendar months, inclusive of t.               |
| dpd_mean_3m                | t-3..t               | numeric | False      |      100    | Mean days_past_due over the last 3 calendar months, inclusive of t.                |
| dpd_mean_6m                | t-6..t               | numeric | False      |      100    | Mean days_past_due over the last 6 calendar months, inclusive of t.                |
| modified_ever              | 0..t                 | numeric | False      |      100    | 1 once the loan has received a loss-mitigation modification.                       |
| months_since_delinquency   | 0..t                 | numeric | False      |        6.78 | Months since the most recent delinquent month; 0 if delinquent now, null if never. |
| months_since_modification  | 0..t                 | numeric | False      |        1.7  | Months since the modification; null if never modified.                             |
| paydown_1m                 | t-1..t               | numeric | False      |       95    | Proportional change in current_balance between month t-1 and month t.              |
| paydown_3m                 | t-3..t               | numeric | False      |       88.12 | Proportional change in current_balance between month t-3 and month t.              |
| paydown_6m                 | t-6..t               | numeric | False      |       77.98 | Proportional change in current_balance between month t-6 and month t.              |
| servicer_transfers_to_date | 0..t                 | numeric | False      |      100    | Count of servicer changes so far.                                                  |
| source_changes_to_date     | 0..t                 | numeric | False      |      100    | Count of source-system changes so far.                                             |
| status_changed_this_month  | t-1..t               | numeric | False      |      100    | 1 where the status differs from last month.                                        |
| status_changes_to_date     | 0..t                 | numeric | False      |      100    | Count of month-on-month status transitions so far.                                 |
| worst_status_to_date       | 0..t                 | numeric | False      |      100    | Running maximum of status_ordinal.                                                 |

## quality (1)

_Phase 1 data-quality signal for the record itself._

| feature   | information_window   | dtype   | baseline   |   % present | definition                                                                                                                             |
|:----------|:---------------------|:--------|:-----------|------------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| dq_score  | as-at t              | numeric | False      |         100 | Phase 1 record-level data-quality score (100 = clean). A record the profiler distrusts is a record whose reported status may be wrong. |
