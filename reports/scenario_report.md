# Scenario & Stress Simulation Report (Task 5)

_Generated 2026-08-31 18:15:28_

## What this projects

The book's 10,000 loans at their latest observed position, re-scored under each macro scenario through the Phase 3 `improved` models. Delinquency, default and prepayment, portfolio-wide and by segment, with the movement attributed to features.


## Scope and limitations

**What these numbers are.** For each scenario and projection month, the
portfolio's features are stressed to that month's assumptions and re-scored
through the Phase 3 models. A figure at horizon 24 is the *conditional forward
rate as at that month*: given the book and the macro state two years out, the
probability a loan defaults over the following twelve months.

**What they are not.** This is a stress-sensitivity projection, not a cash-flow
run-off. The Phase 3 models predict a fixed forward window from a record's own
month; they do not compound, and this pipeline does not pretend they do. The
portfolio is held at its last observed position rather than amortised, loans
that default or prepay are not removed as the horizon extends, and no balance is
rolled forward. The question answered is "how much worse does this book look
under that macro state" -- which is what a stress test is for. The question left
open is "what are the cumulative losses", which needs the Phase 4 hazards and a
run-off engine.

**Scoring population.** The latest observed record per loan: the book as it
stands at the data cutoff. Scoring every historical month would weight the
projection towards loans that happen to have long histories.


## Scenario assumptions

Read directly from `data/macro_scenarios.csv`. Nothing here is a project assumption.

| scenario        |   horizon_month | projection_month   |   mortgage_rate |   unemployment_rate |   hpi_index |   default_multiplier |   prepayment_multiplier |
|:----------------|----------------:|:-------------------|----------------:|--------------------:|------------:|---------------------:|------------------------:|
| Baseline        |               1 | 2024-01            |          6.6    |              3.9    |    100      |               1      |                  1      |
| Baseline        |              12 | 2024-12            |          6.4011 |              3.9702 |    102.747  |               1      |                  1      |
| Baseline        |              24 | 2025-12            |          6.184  |              4.0468 |    105.829  |               1      |                  1      |
| Baseline        |              36 | 2026-12            |          5.967  |              4.1234 |    109.004  |               1      |                  1      |
| Baseline        |              48 | 2027-12            |          5.75   |              4.2    |    112.274  |               1      |                  1      |
| Adverse-Credit  |               1 | 2024-01            |          6.6    |              3.9    |    100      |               1      |                  1      |
| Adverse-Credit  |              12 | 2024-12            |          6.8574 |              5.9319 |     92.6415 |               1.774  |                  0.8947 |
| Adverse-Credit  |              24 | 2025-12            |          7.1383 |              6.8381 |     85.2302 |               2.1193 |                  0.7798 |
| Adverse-Credit  |              36 | 2026-12            |          7.4191 |              7.5244 |     78.4118 |               2.3807 |                  0.6649 |
| Adverse-Credit  |              48 | 2027-12            |          7.7    |              8.1    |     72.1388 |               2.6    |                  0.55   |
| High-Prepayment |               1 | 2024-01            |          6.6    |              3.9    |    100      |               1      |                  1      |
| High-Prepayment |              12 | 2024-12            |          6.0383 |              3.7947 |    105.03   |               0.9415 |                  1.4447 |
| High-Prepayment |              24 | 2025-12            |          5.4255 |              3.6798 |    110.807  |               0.8777 |                  1.9298 |
| High-Prepayment |              36 | 2026-12            |          4.8128 |              3.5649 |    116.901  |               0.8138 |                  2.4149 |
| High-Prepayment |              48 | 2027-12            |          4.2    |              3.45   |    123.331  |               0.75   |                  2.9    |


## Portfolio projection

Mean projected rate per scenario and horizon, with the change against the baseline scenario at the same horizon in percentage points.

| scenario        |   horizon_month | projection_month   |   delinquency_3m |   default_12m |   prepayment_12m |   credit_score_shift |   loans |   delinquency_3m_vs_baseline_pp |   default_12m_vs_baseline_pp |   prepayment_12m_vs_baseline_pp |   default_12m_stated |   prepayment_12m_stated |
|:----------------|----------------:|:-------------------|-----------------:|--------------:|-----------------:|---------------------:|--------:|--------------------------------:|-----------------------------:|--------------------------------:|---------------------:|------------------------:|
| Adverse-Credit  |               1 | 2024-01            |         0.19034  |      0.143952 |        0.0733727 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.143952 |               0.0733727 |
| Adverse-Credit  |              12 | 2024-12            |         0.236554 |      0.252433 |        0.0646544 |                -93.5 |   10000 |                        4.78406  |                    10.9987   |                       -0.874637 |             0.252699 |               0.0656717 |
| Adverse-Credit  |              24 | 2025-12            |         0.2527   |      0.284266 |        0.0608314 |               -250   |   10000 |                        6.55429  |                    14.3554   |                       -1.245    |             0.298211 |               0.0571448 |
| Adverse-Credit  |              36 | 2026-12            |         0.256786 |      0.284635 |        0.0588723 |               -250   |   10000 |                        7.13185  |                    14.5206   |                       -1.4435   |             0.331939 |               0.048742  |
| Adverse-Credit  |              48 | 2027-12            |         0.259521 |      0.28445  |        0.0570541 |               -250   |   10000 |                        7.55183  |                    14.603    |                       -1.60658  |             0.359891 |               0.0402159 |
| Baseline        |               1 | 2024-01            |         0.19034  |      0.143952 |        0.0733727 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.143952 |               0.0733727 |
| Baseline        |              12 | 2024-12            |         0.188714 |      0.142446 |        0.0734008 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.142446 |               0.0734008 |
| Baseline        |              24 | 2025-12            |         0.187157 |      0.140712 |        0.0732814 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.140712 |               0.0732814 |
| Baseline        |              36 | 2026-12            |         0.185467 |      0.139429 |        0.0733073 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.139429 |               0.0733073 |
| Baseline        |              48 | 2027-12            |         0.184003 |      0.13842  |        0.0731199 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.13842  |               0.0731199 |
| High-Prepayment |               1 | 2024-01            |         0.19034  |      0.143952 |        0.0733727 |                  0   |   10000 |                        0        |                     0        |                        0        |             0.143952 |               0.0733727 |
| High-Prepayment |              12 | 2024-12            |         0.184798 |      0.134015 |        0.074661  |                  6.5 |   10000 |                       -0.391544 |                    -0.843058 |                        0.12602  |             0.134113 |               0.106042  |
| High-Prepayment |              24 | 2025-12            |         0.178637 |      0.123539 |        0.0756018 |                 14   |   10000 |                       -0.851995 |                    -1.71731  |                        0.232042 |             0.123503 |               0.141418  |
| High-Prepayment |              36 | 2026-12            |         0.172946 |      0.113559 |        0.0766542 |                 22.5 |   10000 |                       -1.25211  |                    -2.58706  |                        0.334688 |             0.113467 |               0.17703   |
| High-Prepayment |              48 | 2027-12            |         0.166268 |      0.10347  |        0.0776771 |                 31.5 |   10000 |                       -1.77353  |                    -3.49501  |                        0.455721 |             0.103815 |               0.212048  |


## Method

**The scenario file is the only source of stress assumptions.** Nothing in this
pipeline invents an unemployment path, a rate shock or an elasticity.
`data/macro_scenarios.csv` supplies, per scenario and month, a mortgage rate, an
unemployment rate, a house price index and the scenario's own stated default and
prepayment multipliers. A missing column stops the run rather than falling back
to a default.

**Three transmission channels, in descending order of how much they assume.**

*House prices to loan-to-value* is mechanical. LTV is debt over value, so a
house price index at 72 against a starting 100 raises current LTV by a factor of
1.39. No elasticity, no fitted relationship -- arithmetic.

*Market rate to refinance incentive* is mechanical. `rate_spread` is the loan's
note rate less the prevailing market rate, and the scenario states that rate. A
borrower paying 7% in a 4.2% market has an incentive the Phase 3 model already
knows how to read.

*Labour market to credit quality* is *calibrated, not assumed*. The file gives
an unemployment path and a default multiplier but no elasticity connecting them
to credit scores. Hard-coding one -- "40 points of FICO per point of
unemployment" -- would make the projection a restatement of that invented
number. Instead the shift is solved for: find the portfolio-wide credit-score
move that makes the model reproduce the scenario's *own* stated default
multiplier. The file stays authoritative, the model supplies the transmission,
and the answer lands in a unit a credit officer recognises.

**Bounds are enforced and banded columns are rebuilt.** Every stressed feature is
clipped to a plausible range, and `credit_score_band`, `ltv_band` and `dti_band`
are recomputed from the shifted values underneath them. Moving `ltv` while
leaving `ltv_band` at its original level hands the model a record that
contradicts itself -- and the model will score it without complaint.

**One credit shift per scenario-month, shared across all three targets.** The
shift describes a state of the world, not a per-model tuning knob. Letting each
target solve its own would produce three mutually inconsistent portfolios and
call them one scenario.


## Where this projection runs out of road

**The credit channel saturates, and the projection says so.** Beyond a point,
no shift in credit score reproduces the default multiplier the scenario file
states. Even moving the entire book to the floor of the observable score range
(500) leaves the Adverse-Credit multiplier short at the longer horizons. Three
readings, all worth stating:

1. The Phase 3 model's sensitivity to credit score is bounded by the range it
   was fitted on. A 2.6x default multiplier is outside what this book's credit
   distribution can express, however hard it is pushed.
2. The scenario's severity is therefore carried by channels the model does not
   see. Unemployment at 8.1% is not a feature; it reaches the model only
   through the calibrated shift, and the shift has run out of room.
3. A naive calibration would clamp at its search bound, report a number and
   move on -- and the projection would quietly under-state the scenario. The
   saturation table names the shortfall in the file's own units instead.

This is why both projection methods are reported. The **feature-stress** column
is what the model says when its inputs are moved; the **stated-multiplier**
column is the scenario file's own view applied directly to the baseline rate.
Where they diverge, the feature-stress figure is a floor, not a forecast.

**Prepayment barely responds to the rate path.** The High-Prepayment scenario
states a 2.9x prepayment multiplier by month 48; the feature-stress projection
produces roughly 1.05x. That is consistent rather than anomalous: Task 2 found
prepayment close to unpredictable in this pack (ROC-AUC 0.52) and Task 3 found
a cause-specific Cox model barely beating a constant hazard on it (C = 0.558).
A model with no prepayment signal cannot acquire one under stress. The
stated-multiplier column is the usable projection for prepayment here.


## Calibrated credit channel

The portfolio-wide credit-score shift that makes the model reproduce each scenario's stated default multiplier. `converged = False` marks a multiplier the model cannot reach by any shift in the search range -- reported as a number rather than silently clamped.

| scenario        |   horizon_month |   credit_score_shift |   baseline_rate | converged   |     residual |   target_multiplier |   target_rate |   attainable_low |   attainable_high |   attained_rate |   attainable_multiplier |
|:----------------|----------------:|---------------------:|----------------:|:------------|-------------:|--------------------:|--------------:|-----------------:|------------------:|----------------:|------------------------:|
| Adverse-Credit  |               1 |                  0   |        0.143952 | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| Adverse-Credit  |              12 |                -93.5 |        0.142446 | True        | -0.000266051 |              1.774  |      0.252699 |         0.282657 |         0.0678621 |        0.252433 |                 1.774   |
| Adverse-Credit  |              24 |               -250   |        0.140712 | False       | -0.0139456   |              2.1193 |      0.298211 |         0.284266 |         0.0723574 |        0.284266 |                 2.02019 |
| Adverse-Credit  |              36 |               -250   |        0.139429 | False       | -0.0473037   |              2.3807 |      0.331939 |         0.284635 |         0.0742845 |        0.284635 |                 2.04143 |
| Adverse-Credit  |              48 |               -250   |        0.13842  | False       | -0.0754416   |              2.6    |      0.359891 |         0.28445  |         0.0748448 |        0.28445  |                 2.05498 |
| Baseline        |               1 |                  0   |        0.143952 | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| Baseline        |              12 |                  0   |        0.142446 | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| Baseline        |              24 |                  0   |        0.140712 | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| Baseline        |              36 |                  0   |        0.139429 | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| Baseline        |              48 |                  0   |        0.13842  | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| High-Prepayment |               1 |                  0   |        0.143952 | True        |  0           |              1      |    nan        |       nan        |       nan         |      nan        |               nan       |
| High-Prepayment |              12 |                  6.5 |        0.142446 | True        | -9.74922e-05 |              0.9415 |      0.134113 |         0.279868 |         0.05797   |        0.134015 |                 0.9415  |
| High-Prepayment |              24 |                 14   |        0.140712 | True        |  3.59916e-05 |              0.8777 |      0.123503 |         0.280179 |         0.0541968 |        0.123539 |                 0.8777  |
| High-Prepayment |              36 |                 22.5 |        0.139429 | True        |  9.1112e-05  |              0.8138 |      0.113467 |         0.279757 |         0.0514851 |        0.113559 |                 0.8138  |
| High-Prepayment |              48 |                 31.5 |        0.13842  | True        | -0.000345197 |              0.75   |      0.103815 |         0.279677 |         0.0504099 |        0.10347  |                 0.75    |


## Credit channel saturation

`reached = False` marks a stated multiplier the model cannot produce by any credit-score shift. `attainable_multiplier` is the ceiling it reaches at the floor of the observable score range, and `shortfall` is the gap the feature-stress projection therefore under-states by.

| scenario        |   horizon_month |   stated_multiplier |   attainable_multiplier |   credit_score_shift | reached   |   shortfall |
|:----------------|----------------:|--------------------:|------------------------:|---------------------:|:----------|------------:|
| Adverse-Credit  |               1 |              1      |               nan       |                  0   | True      | nan         |
| Adverse-Credit  |              12 |              1.774  |                 1.774   |                -93.5 | True      |   0         |
| Adverse-Credit  |              24 |              2.1193 |                 2.02019 |               -250   | False     |   0.0991073 |
| Adverse-Credit  |              36 |              2.3807 |                 2.04143 |               -250   | False     |   0.339267  |
| Adverse-Credit  |              48 |              2.6    |                 2.05498 |               -250   | False     |   0.545021  |
| High-Prepayment |               1 |              1      |               nan       |                  0   | True      | nan         |
| High-Prepayment |              12 |              0.9415 |                 0.9415  |                  6.5 | True      |   0         |
| High-Prepayment |              24 |              0.8777 |                 0.8777  |                 14   | True      |   0         |
| High-Prepayment |              36 |              0.8138 |                 0.8138  |                 22.5 | True      |   0         |
| High-Prepayment |              48 |              0.75   |                 0.75    |                 31.5 | True      |   0         |


## Stated vs. modelled multipliers

Two independent views of the same scenario. Default agrees by construction -- the credit channel was calibrated to make it agree. **Prepayment was not calibrated against anything**, so its column is the model's own view of the refinance response, derived only from the rate path. Where it disagrees with the file, that gap is a finding about the model, not an error in the projection.

| scenario        |   horizon_month | measure        |   stated_multiplier |   model_multiplier |          gap | calibrated   |
|:----------------|----------------:|:---------------|--------------------:|-------------------:|-------------:|:-------------|
| Adverse-Credit  |               1 | default_12m    |              1      |           1        |  0           | True         |
| Adverse-Credit  |               1 | prepayment_12m |              1      |           1        |  0           | False        |
| Adverse-Credit  |              12 | default_12m    |              1.774  |           1.77213  | -0.00186773  | True         |
| Adverse-Credit  |              12 | prepayment_12m |              0.8947 |           0.880841 | -0.0138591   | False        |
| Adverse-Credit  |              24 | default_12m    |              2.1193 |           2.02019  | -0.0991073   | True         |
| Adverse-Credit  |              24 | prepayment_12m |              0.7798 |           0.830107 |  0.050307    | False        |
| Adverse-Credit  |              36 | default_12m    |              2.3807 |           2.04143  | -0.339267    | True         |
| Adverse-Credit  |              36 | prepayment_12m |              0.6649 |           0.803089 |  0.138189    | False        |
| Adverse-Credit  |              48 | default_12m    |              2.6    |           2.05498  | -0.545021    | True         |
| Adverse-Credit  |              48 | prepayment_12m |              0.55   |           0.780281 |  0.230281    | False        |
| Baseline        |               1 | default_12m    |              1      |           1        |  0           | True         |
| Baseline        |               1 | prepayment_12m |              1      |           1        |  0           | False        |
| Baseline        |              12 | default_12m    |              1      |           1        |  0           | True         |
| Baseline        |              12 | prepayment_12m |              1      |           1        |  0           | False        |
| Baseline        |              24 | default_12m    |              1      |           1        |  0           | True         |
| Baseline        |              24 | prepayment_12m |              1      |           1        |  0           | False        |
| Baseline        |              36 | default_12m    |              1      |           1        |  0           | True         |
| Baseline        |              36 | prepayment_12m |              1      |           1        |  0           | False        |
| Baseline        |              48 | default_12m    |              1      |           1        |  0           | True         |
| Baseline        |              48 | prepayment_12m |              1      |           1        |  0           | False        |
| High-Prepayment |               1 | default_12m    |              1      |           1        |  0           | True         |
| High-Prepayment |               1 | prepayment_12m |              1      |           1        |  0           | False        |
| High-Prepayment |              12 | default_12m    |              0.9415 |           0.940816 | -0.000684416 | True         |
| High-Prepayment |              12 | prepayment_12m |              1.4447 |           1.01717  | -0.427531    | False        |
| High-Prepayment |              24 | default_12m    |              0.8777 |           0.877956 |  0.000255782 | True         |
| High-Prepayment |              24 | prepayment_12m |              1.9298 |           1.03166  | -0.898135    | False        |
| High-Prepayment |              36 | default_12m    |              0.8138 |           0.814453 |  0.000653465 | True         |
| High-Prepayment |              36 | prepayment_12m |              2.4149 |           1.04566  | -1.36924     | False        |
| High-Prepayment |              48 | default_12m    |              0.75   |           0.747506 | -0.00249384  | True         |
| High-Prepayment |              48 | prepayment_12m |              2.9    |           1.06233  | -1.83767     | False        |


### Impact by vintage year

A portfolio-level move spread evenly is a different problem from the same move concentrated in one segment.

| scenario        |   horizon_month |   vintage_year |   delinquency_3m |   default_12m |   prepayment_12m |   loans |   delinquency_3m_vs_baseline_pp |   default_12m_vs_baseline_pp |   prepayment_12m_vs_baseline_pp |
|:----------------|----------------:|---------------:|-----------------:|--------------:|-----------------:|--------:|--------------------------------:|-----------------------------:|--------------------------------:|
| Adverse-Credit  |               1 |           2018 |         0.195859 |      0.143443 |        0.0701937 |    1717 |                         0       |                       0      |                        0        |
| Adverse-Credit  |               1 |           2019 |         0.207966 |      0.156062 |        0.0720127 |    1675 |                         0       |                       0      |                        0        |
| Adverse-Credit  |               1 |           2020 |         0.203838 |      0.160304 |        0.0746378 |    1650 |                         0       |                       0      |                        0        |
| Adverse-Credit  |               1 |           2021 |         0.178277 |      0.142508 |        0.0768422 |    1658 |                         0       |                       0      |                        0        |
| Adverse-Credit  |               1 |           2022 |         0.185026 |      0.136055 |        0.0729287 |    1698 |                         0       |                       0      |                        0        |
| Adverse-Credit  |               1 |           2023 |         0.170213 |      0.12486  |        0.0737785 |    1602 |                         0       |                       0      |                        0        |
| Baseline        |               1 |           2018 |         0.195859 |      0.143443 |        0.0701937 |    1717 |                         0       |                       0      |                        0        |
| Baseline        |               1 |           2019 |         0.207966 |      0.156062 |        0.0720127 |    1675 |                         0       |                       0      |                        0        |
| Baseline        |               1 |           2020 |         0.203838 |      0.160304 |        0.0746378 |    1650 |                         0       |                       0      |                        0        |
| Baseline        |               1 |           2021 |         0.178277 |      0.142508 |        0.0768422 |    1658 |                         0       |                       0      |                        0        |
| Baseline        |               1 |           2022 |         0.185026 |      0.136055 |        0.0729287 |    1698 |                         0       |                       0      |                        0        |
| Baseline        |               1 |           2023 |         0.170213 |      0.12486  |        0.0737785 |    1602 |                         0       |                       0      |                        0        |
| High-Prepayment |               1 |           2018 |         0.195859 |      0.143443 |        0.0701937 |    1717 |                         0       |                       0      |                        0        |
| High-Prepayment |               1 |           2019 |         0.207966 |      0.156062 |        0.0720127 |    1675 |                         0       |                       0      |                        0        |
| High-Prepayment |               1 |           2020 |         0.203838 |      0.160304 |        0.0746378 |    1650 |                         0       |                       0      |                        0        |
| High-Prepayment |               1 |           2021 |         0.178277 |      0.142508 |        0.0768422 |    1658 |                         0       |                       0      |                        0        |
| High-Prepayment |               1 |           2022 |         0.185026 |      0.136055 |        0.0729287 |    1698 |                         0       |                       0      |                        0        |
| High-Prepayment |               1 |           2023 |         0.170213 |      0.12486  |        0.0737785 |    1602 |                         0       |                       0      |                        0        |
| Adverse-Credit  |              12 |           2018 |         0.240345 |      0.248085 |        0.0581234 |    1717 |                         4.6705  |                      10.5562 |                       -1.16672  |
| Adverse-Credit  |              12 |           2019 |         0.252241 |      0.260187 |        0.063774  |    1675 |                         4.62564 |                      10.5172 |                       -0.823373 |
| Adverse-Credit  |              12 |           2020 |         0.24305  |      0.255052 |        0.068516  |    1650 |                         4.0866  |                       9.6135 |                       -0.589858 |
| Adverse-Credit  |              12 |           2021 |         0.218792 |      0.247111 |        0.070685  |    1658 |                         4.178   |                      10.6578 |                       -0.627346 |
| Adverse-Credit  |              12 |           2022 |         0.236426 |      0.249171 |        0.06395   |    1698 |                         5.36809 |                      11.4721 |                       -0.920036 |
| Adverse-Credit  |              12 |           2023 |         0.227917 |      0.255253 |        0.0631026 |    1602 |                         5.79795 |                      13.2542 |                       -1.11631  |
| Baseline        |              12 |           2018 |         0.19364  |      0.142523 |        0.0697906 |    1717 |                         0       |                       0      |                        0        |
| Baseline        |              12 |           2019 |         0.205985 |      0.155015 |        0.0720077 |    1675 |                         0       |                       0      |                        0        |
| Baseline        |              12 |           2020 |         0.202184 |      0.158917 |        0.0744146 |    1650 |                         0       |                       0      |                        0        |
| Baseline        |              12 |           2021 |         0.177012 |      0.140534 |        0.0769585 |    1658 |                         0       |                       0      |                        0        |
| Baseline        |              12 |           2022 |         0.182745 |      0.13445  |        0.0731504 |    1698 |                         0       |                       0      |                        0        |
| Baseline        |              12 |           2023 |         0.169938 |      0.122711 |        0.0742658 |    1602 |                         0       |                       0      |                        0        |

_Showing 30 of 90 rows._


### Impact by credit score band

A portfolio-level move spread evenly is a different problem from the same move concentrated in one segment.

| scenario        |   horizon_month | credit_score_band   |   delinquency_3m |   default_12m |   prepayment_12m |   loans |   delinquency_3m_vs_baseline_pp |   default_12m_vs_baseline_pp |   prepayment_12m_vs_baseline_pp |
|:----------------|----------------:|:--------------------|-----------------:|--------------:|-----------------:|--------:|--------------------------------:|-----------------------------:|--------------------------------:|
| Adverse-Credit  |               1 | 620-659             |        0.410624  |    0.362905   |        0.0500349 |    1531 |                        0        |                     0        |                       0         |
| Adverse-Credit  |               1 | 660-699             |        0.218319  |    0.173845   |        0.0735544 |    2899 |                        0        |                     0        |                       0         |
| Adverse-Credit  |               1 | 700-739             |        0.0976616 |    0.0423067  |        0.0800408 |    2838 |                        0        |                     0        |                       0         |
| Adverse-Credit  |               1 | 740-799             |        0.0490421 |    0.014397   |        0.0896517 |    1946 |                        0        |                     0        |                       0         |
| Adverse-Credit  |               1 | 800+                |        0.0268239 |    0.00508974 |        0.0824787 |     247 |                        0        |                     0        |                       0         |
| Adverse-Credit  |               1 | <620                |        0.487209  |    0.427829   |        0.0406288 |     539 |                        0        |                     0        |                       0         |
| Baseline        |               1 | 620-659             |        0.410624  |    0.362905   |        0.0500349 |    1531 |                        0        |                     0        |                       0         |
| Baseline        |               1 | 660-699             |        0.218319  |    0.173845   |        0.0735544 |    2899 |                        0        |                     0        |                       0         |
| Baseline        |               1 | 700-739             |        0.0976616 |    0.0423067  |        0.0800408 |    2838 |                        0        |                     0        |                       0         |
| Baseline        |               1 | 740-799             |        0.0490421 |    0.014397   |        0.0896517 |    1946 |                        0        |                     0        |                       0         |
| Baseline        |               1 | 800+                |        0.0268239 |    0.00508974 |        0.0824787 |     247 |                        0        |                     0        |                       0         |
| Baseline        |               1 | <620                |        0.487209  |    0.427829   |        0.0406288 |     539 |                        0        |                     0        |                       0         |
| High-Prepayment |               1 | 620-659             |        0.410624  |    0.362905   |        0.0500349 |    1531 |                        0        |                     0        |                       0         |
| High-Prepayment |               1 | 660-699             |        0.218319  |    0.173845   |        0.0735544 |    2899 |                        0        |                     0        |                       0         |
| High-Prepayment |               1 | 700-739             |        0.0976616 |    0.0423067  |        0.0800408 |    2838 |                        0        |                     0        |                       0         |
| High-Prepayment |               1 | 740-799             |        0.0490421 |    0.014397   |        0.0896517 |    1946 |                        0        |                     0        |                       0         |
| High-Prepayment |               1 | 800+                |        0.0268239 |    0.00508974 |        0.0824787 |     247 |                        0        |                     0        |                       0         |
| High-Prepayment |               1 | <620                |        0.487209  |    0.427829   |        0.0406288 |     539 |                        0        |                     0        |                       0         |
| Adverse-Credit  |              12 | 620-659             |        0.421916  |    0.383257   |        0.0476832 |    1531 |                        0        |                     0        |                       0         |
| Adverse-Credit  |              12 | 660-699             |        0.269044  |    0.296249   |        0.0624666 |    2899 |                        0        |                     0        |                       0         |
| Adverse-Credit  |              12 | 700-739             |        0.169414  |    0.227921   |        0.0708886 |    2838 |                        0        |                     0        |                       0         |
| Adverse-Credit  |              12 | 740-799             |        0.0934353 |    0.100912   |        0.0771651 |    1946 |                        0        |                     0        |                       0         |
| Adverse-Credit  |              12 | 800+                |        0.0458069 |    0.0155763  |        0.0801098 |     247 |                        0        |                     0        |                       0         |
| Adverse-Credit  |              12 | <620                |        0.492941  |    0.42982    |        0.0395505 |     539 |                        0        |                     0        |                       0         |
| Baseline        |              12 | 620-659             |        0.407852  |    0.360933   |        0.0502742 |    1531 |                       -1.40641  |                    -2.23243  |                       0.259097  |
| Baseline        |              12 | 660-699             |        0.215907  |    0.172174   |        0.0737097 |    2899 |                       -5.31369  |                   -12.4075   |                       1.12431   |
| Baseline        |              12 | 700-739             |        0.0968548 |    0.0405057  |        0.0801269 |    2838 |                       -7.25588  |                   -18.7416   |                       0.923831  |
| Baseline        |              12 | 740-799             |        0.0482808 |    0.0134007  |        0.0891938 |    1946 |                       -4.51545  |                    -8.75117  |                       1.20286   |
| Baseline        |              12 | 800+                |        0.0263193 |    0.00468109 |        0.0839345 |     247 |                       -1.94876  |                    -1.08953  |                       0.382472  |
| Baseline        |              12 | <620                |        0.485105  |    0.427736   |        0.0401674 |     539 |                       -0.783583 |                    -0.208464 |                       0.0616837 |

_Showing 30 of 90 rows._


### Impact by state

A portfolio-level move spread evenly is a different problem from the same move concentrated in one segment.

| scenario       |   horizon_month | state   |   delinquency_3m |   default_12m |   prepayment_12m |   loans |   delinquency_3m_vs_baseline_pp |   default_12m_vs_baseline_pp |   prepayment_12m_vs_baseline_pp |
|:---------------|----------------:|:--------|-----------------:|--------------:|-----------------:|--------:|--------------------------------:|-----------------------------:|--------------------------------:|
| Adverse-Credit |               1 | AL      |         0.206056 |      0.139835 |        0.0680783 |     143 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | AZ      |         0.228286 |      0.174343 |        0.0757705 |     266 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | CA      |         0.193613 |      0.149562 |        0.0759933 |    1290 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | CO      |         0.194999 |      0.150504 |        0.0791216 |     176 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | CT      |         0.20877  |      0.156119 |        0.0710712 |     109 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | FL      |         0.181874 |      0.142642 |        0.0719414 |     972 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | GA      |         0.196991 |      0.139511 |        0.0706848 |     383 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | IL      |         0.181757 |      0.131933 |        0.0803706 |     412 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | IN      |         0.162226 |      0.118805 |        0.0772715 |     210 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | KY      |         0.223874 |      0.1643   |        0.0713025 |     152 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | LA      |         0.20957  |      0.157345 |        0.0692294 |     133 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | MA      |         0.177282 |      0.14285  |        0.0682086 |     236 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | MD      |         0.185073 |      0.134208 |        0.0783397 |     199 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | MI      |         0.206838 |      0.160949 |        0.0685165 |     313 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | MN      |         0.158867 |      0.121612 |        0.0830356 |     179 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | MO      |         0.192553 |      0.148662 |        0.074336  |     214 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | NC      |         0.205388 |      0.14654  |        0.0739408 |     367 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | NJ      |         0.172678 |      0.114352 |        0.0731286 |     288 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | NY      |         0.199569 |      0.148748 |        0.0732316 |     589 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | OH      |         0.189704 |      0.152138 |        0.0709667 |     419 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | OK      |         0.152744 |      0.107053 |        0.0664918 |     119 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | OR      |         0.1565   |      0.123831 |        0.0588142 |     167 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | PA      |         0.200522 |      0.159295 |        0.0721924 |     419 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | SC      |         0.208975 |      0.152467 |        0.0744546 |     147 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | TN      |         0.191374 |      0.142109 |        0.0718141 |     242 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | TX      |         0.190461 |      0.144183 |        0.077222  |    1000 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | UT      |         0.195983 |      0.150595 |        0.0677134 |     100 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | VA      |         0.161397 |      0.119039 |        0.0790733 |     295 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | WA      |         0.203444 |      0.156852 |        0.0583066 |     279 |                               0 |                            0 |                               0 |
| Adverse-Credit |               1 | WI      |         0.155529 |      0.120254 |        0.0707925 |     182 |                               0 |                            0 |                               0 |

_Showing 30 of 450 rows._


### Impact by servicer name

A portfolio-level move spread evenly is a different problem from the same move concentrated in one segment.

| scenario        |   horizon_month | servicer_name                 |   delinquency_3m |   default_12m |   prepayment_12m |   loans |   delinquency_3m_vs_baseline_pp |   default_12m_vs_baseline_pp |   prepayment_12m_vs_baseline_pp |
|:----------------|----------------:|:------------------------------|-----------------:|--------------:|-----------------:|--------:|--------------------------------:|-----------------------------:|--------------------------------:|
| Adverse-Credit  |               1 | Atlas Mortgage Services       |         0.186354 |      0.141468 |        0.0732396 |    2019 |                        0        |                     0        |                        0        |
| Adverse-Credit  |               1 | Beacon Home Loans             |         0.192456 |      0.14576  |        0.0728025 |    1964 |                        0        |                     0        |                        0        |
| Adverse-Credit  |               1 | Cornerstone Loan Servicing    |         0.202367 |      0.154597 |        0.0740155 |    2049 |                        0        |                     0        |                        0        |
| Adverse-Credit  |               1 | Meridian Residential Capital  |         0.186414 |      0.139425 |        0.0740326 |    2033 |                        0        |                     0        |                        0        |
| Adverse-Credit  |               1 | Northgate Financial Servicing |         0.183742 |      0.138195 |        0.0727161 |    1935 |                        0        |                     0        |                        0        |
| Baseline        |               1 | Atlas Mortgage Services       |         0.186354 |      0.141468 |        0.0732396 |    2019 |                        0        |                     0        |                        0        |
| Baseline        |               1 | Beacon Home Loans             |         0.192456 |      0.14576  |        0.0728025 |    1964 |                        0        |                     0        |                        0        |
| Baseline        |               1 | Cornerstone Loan Servicing    |         0.202367 |      0.154597 |        0.0740155 |    2049 |                        0        |                     0        |                        0        |
| Baseline        |               1 | Meridian Residential Capital  |         0.186414 |      0.139425 |        0.0740326 |    2033 |                        0        |                     0        |                        0        |
| Baseline        |               1 | Northgate Financial Servicing |         0.183742 |      0.138195 |        0.0727161 |    1935 |                        0        |                     0        |                        0        |
| High-Prepayment |               1 | Atlas Mortgage Services       |         0.186354 |      0.141468 |        0.0732396 |    2019 |                        0        |                     0        |                        0        |
| High-Prepayment |               1 | Beacon Home Loans             |         0.192456 |      0.14576  |        0.0728025 |    1964 |                        0        |                     0        |                        0        |
| High-Prepayment |               1 | Cornerstone Loan Servicing    |         0.202367 |      0.154597 |        0.0740155 |    2049 |                        0        |                     0        |                        0        |
| High-Prepayment |               1 | Meridian Residential Capital  |         0.186414 |      0.139425 |        0.0740326 |    2033 |                        0        |                     0        |                        0        |
| High-Prepayment |               1 | Northgate Financial Servicing |         0.183742 |      0.138195 |        0.0727161 |    1935 |                        0        |                     0        |                        0        |
| Adverse-Credit  |              12 | Atlas Mortgage Services       |         0.232373 |      0.255781 |        0.0643106 |    2019 |                        4.77654  |                    11.5828   |                       -0.902257 |
| Adverse-Credit  |              12 | Beacon Home Loans             |         0.239294 |      0.260781 |        0.0655988 |    1964 |                        4.86289  |                    11.6416   |                       -0.740389 |
| Adverse-Credit  |              12 | Cornerstone Loan Servicing    |         0.248005 |      0.259538 |        0.0652617 |    2049 |                        4.71335  |                    10.6267   |                       -0.862903 |
| Adverse-Credit  |              12 | Meridian Residential Capital  |         0.232418 |      0.244889 |        0.0635474 |    2033 |                        4.77479  |                    10.7502   |                       -1.0289   |
| Adverse-Credit  |              12 | Northgate Financial Servicing |         0.230356 |      0.240868 |        0.0645746 |    1935 |                        4.79649  |                    10.3918   |                       -0.832423 |
| Baseline        |              12 | Atlas Mortgage Services       |         0.184607 |      0.139954 |        0.0733331 |    2019 |                        0        |                     0        |                        0        |
| Baseline        |              12 | Beacon Home Loans             |         0.190665 |      0.144365 |        0.0730027 |    1964 |                        0        |                     0        |                        0        |
| Baseline        |              12 | Cornerstone Loan Servicing    |         0.200872 |      0.153271 |        0.0738907 |    2049 |                        0        |                     0        |                        0        |
| Baseline        |              12 | Meridian Residential Capital  |         0.18467  |      0.137387 |        0.0738364 |    2033 |                        0        |                     0        |                        0        |
| Baseline        |              12 | Northgate Financial Servicing |         0.182392 |      0.136951 |        0.0728988 |    1935 |                        0        |                     0        |                        0        |
| High-Prepayment |              12 | Atlas Mortgage Services       |         0.180462 |      0.130491 |        0.0745311 |    2019 |                       -0.414572 |                    -0.946222 |                        0.119792 |
| High-Prepayment |              12 | Beacon Home Loans             |         0.186895 |      0.136665 |        0.0740328 |    1964 |                       -0.376969 |                    -0.769997 |                        0.103013 |
| High-Prepayment |              12 | Cornerstone Loan Servicing    |         0.19658  |      0.144835 |        0.0753584 |    2049 |                       -0.429158 |                    -0.843572 |                        0.146773 |
| High-Prepayment |              12 | Meridian Residential Capital  |         0.180895 |      0.129102 |        0.0751975 |    2033 |                       -0.377441 |                    -0.828533 |                        0.136111 |
| High-Prepayment |              12 | Northgate Financial Servicing |         0.178819 |      0.128708 |        0.0741317 |    1935 |                       -0.357297 |                    -0.824287 |                        0.123291 |

_Showing 30 of 75 rows._


## Scenario drivers

Generated from the model's own per-feature contributions: the change in each feature's mean contribution between the baseline and stressed portfolios is that feature's share of the change in the rate.

### Adverse-Credit / delinquency_3m

The scenario raises the projected rate by 7.55 percentage points. Attribution over the portfolio:
- **credit score** pushes it up, 43% of the total movement (700.61 -> 504.26, -28.0%).
- **ltv** pushes it up, 20% of the total movement (66.72 -> 103.84, +55.6%).
- **credit score band** pushes it up, 14% of the total movement.
- **rate spread** pushes it down, 2% of the total movement (-0.76 -> -2.71, +258.1%).
- **current balance** pushes it up, 2% of the total movement.

### Adverse-Credit / default_12m

The scenario raises the projected rate by 14.60 percentage points. Attribution over the portfolio:
- **credit score** pushes it up, 56% of the total movement (700.61 -> 504.26, -28.0%).
- **credit score band** pushes it up, 14% of the total movement.
- **ltv** pushes it up, 11% of the total movement (66.72 -> 103.84, +55.6%).
- **ltv band** pushes it up, 6% of the total movement.
- **balance ratio** pushes it down, 2% of the total movement.

### Adverse-Credit / prepayment_12m

The scenario lowers the projected rate by 1.61 percentage points. Attribution over the portfolio:
- **credit score** pushes it down, 51% of the total movement (700.61 -> 504.26, -28.0%).
- **ltv** pushes it down, 23% of the total movement (66.72 -> 103.84, +55.6%).
- **balance gap abs** pushes it up, 6% of the total movement.
- **dq score** pushes it up, 4% of the total movement.
- **interest rate** pushes it up, 3% of the total movement (4.99 -> 4.99, +0.0%).

### High-Prepayment / delinquency_3m

The scenario lowers the projected rate by 1.77 percentage points. Attribution over the portfolio:
- **credit score** pushes it down, 40% of the total movement (700.61 -> 732.01, +4.5%).
- **credit score band** pushes it down, 18% of the total movement.
- **ltv** pushes it down, 14% of the total movement (66.72 -> 60.74, -9.0%).
- **rate spread** pushes it up, 7% of the total movement (-0.76 -> 0.79, -205.2%).
- **dpd mean 6m** pushes it up, 5% of the total movement.

### High-Prepayment / default_12m

The scenario lowers the projected rate by 3.50 percentage points. Attribution over the portfolio:
- **credit score** pushes it down, 54% of the total movement (700.61 -> 732.01, +4.5%).
- **credit score band** pushes it down, 16% of the total movement.
- **ltv** pushes it down, 6% of the total movement (66.72 -> 60.74, -9.0%).
- **ltv band** pushes it down, 5% of the total movement.
- **days past due** pushes it up, 4% of the total movement.

### High-Prepayment / prepayment_12m

The scenario raises the projected rate by 0.46 percentage points. Attribution over the portfolio:
- **credit score** pushes it up, 51% of the total movement (700.61 -> 732.01, +4.5%).
- **rate spread** pushes it up, 10% of the total movement (-0.76 -> 0.79, -205.2%).
- **ltv** pushes it down, 9% of the total movement (66.72 -> 60.74, -9.0%).
- **interest rate** pushes it down, 6% of the total movement (4.99 -> 4.99, +0.0%).
- **balance gap abs** pushes it down, 4% of the total movement.


### Driver attribution

| scenario       |   horizon_month | measure        | feature                  |   baseline_contribution |   stressed_contribution |   delta_contribution |   share_of_movement |
|:---------------|----------------:|:---------------|:-------------------------|------------------------:|------------------------:|---------------------:|--------------------:|
| Adverse-Credit |               1 | delinquency_3m | credit_score             |            -0.0212859   |            -0.0212859   |           0          |         nan         |
| Adverse-Credit |               1 | delinquency_3m | ltv                      |             0.0138155   |             0.0138155   |           0          |         nan         |
| Adverse-Credit |               1 | delinquency_3m | dti                      |             0.00500124  |             0.00500124  |           0          |         nan         |
| Adverse-Credit |               1 | delinquency_3m | original_balance         |            -0.000548463 |            -0.000548463 |           0          |         nan         |
| Adverse-Credit |               1 | delinquency_3m | original_term_months     |             7.52098e-05 |             7.52098e-05 |           0          |         nan         |
| Adverse-Credit |               1 | default_12m    | credit_score             |            -0.0395236   |            -0.0395236   |           0          |         nan         |
| Adverse-Credit |               1 | default_12m    | ltv                      |            -0.000172543 |            -0.000172543 |           0          |         nan         |
| Adverse-Credit |               1 | default_12m    | dti                      |             0.0241869   |             0.0241869   |           0          |         nan         |
| Adverse-Credit |               1 | default_12m    | original_balance         |             0.0158715   |             0.0158715   |           0          |         nan         |
| Adverse-Credit |               1 | default_12m    | original_term_months     |             0.000424471 |             0.000424471 |           0          |         nan         |
| Adverse-Credit |               1 | prepayment_12m | credit_score             |            -0.0333167   |            -0.0333167   |           0          |         nan         |
| Adverse-Credit |               1 | prepayment_12m | ltv                      |             0.00907896  |             0.00907896  |           0          |         nan         |
| Adverse-Credit |               1 | prepayment_12m | dti                      |             0.000211006 |             0.000211006 |           0          |         nan         |
| Adverse-Credit |               1 | prepayment_12m | original_balance         |             0.00407749  |             0.00407749  |           0          |         nan         |
| Adverse-Credit |               1 | prepayment_12m | original_term_months     |             0.000635169 |             0.000635169 |           0          |         nan         |
| Adverse-Credit |              12 | delinquency_3m | credit_score             |            -0.0210317   |             0.527624    |           0.548656   |           0.535093  |
| Adverse-Credit |              12 | delinquency_3m | credit_score_band        |            -0.0139755   |             0.18154     |           0.195516   |           0.190682  |
| Adverse-Credit |              12 | delinquency_3m | ltv                      |            -0.0136089   |             0.0703349   |           0.0839438  |           0.0818686 |
| Adverse-Credit |              12 | delinquency_3m | paydown_3m               |            -0.139322    |            -0.117344    |           0.0219788  |           0.0214355 |
| Adverse-Credit |              12 | delinquency_3m | dpd_mean_6m              |            -0.101449    |            -0.122684    |          -0.0212352  |           0.0207103 |
| Adverse-Credit |              12 | default_12m    | credit_score             |            -0.0360379   |             1.68826     |           1.7243     |           0.638811  |
| Adverse-Credit |              12 | default_12m    | credit_score_band        |            -0.0100846   |             0.440032    |           0.450117   |           0.166758  |
| Adverse-Credit |              12 | default_12m    | ltv                      |            -0.0467888   |             0.0898129   |           0.136602   |           0.0506077 |
| Adverse-Credit |              12 | default_12m    | ltv_band                 |            -0.0119439   |             0.0611716   |           0.0731155  |           0.0270876 |
| Adverse-Credit |              12 | default_12m    | days_past_due            |             0.512434    |             0.455736    |          -0.056698   |           0.0210053 |
| Adverse-Credit |              12 | prepayment_12m | credit_score             |            -0.0328086   |            -0.380776    |          -0.347967   |           0.674765  |
| Adverse-Credit |              12 | prepayment_12m | balance_gap_abs          |            -0.123986    |            -0.0873559   |           0.0366298  |           0.0710312 |
| Adverse-Credit |              12 | prepayment_12m | interest_rate            |            -0.038697    |            -0.0100185   |           0.0286786  |           0.0556124 |
| Adverse-Credit |              12 | prepayment_12m | dq_score                 |             0.016493    |             0.0384386   |           0.0219455  |           0.042556  |
| Adverse-Credit |              12 | prepayment_12m | dpd_max_6m               |            -0.0211838   |            -0.0296295   |          -0.00844573 |           0.0163776 |
| Adverse-Credit |              24 | delinquency_3m | credit_score             |            -0.0204952   |             0.660749    |           0.681245   |           0.507245  |
| Adverse-Credit |              24 | delinquency_3m | credit_score_band        |            -0.0142265   |             0.201923    |           0.21615    |           0.160942  |
| Adverse-Credit |              24 | delinquency_3m | ltv                      |            -0.0399047   |             0.143743    |           0.183648   |           0.136742  |
| Adverse-Credit |              24 | delinquency_3m | paydown_3m               |            -0.138552    |            -0.11141     |           0.0271418  |           0.0202094 |
| Adverse-Credit |              24 | delinquency_3m | months_since_delinquency |             0.356871    |             0.335804    |          -0.0210669  |           0.0156861 |
| Adverse-Credit |              24 | default_12m    | credit_score             |            -0.033233    |             2.01199     |           2.04522    |           0.611522  |
| Adverse-Credit |              24 | default_12m    | credit_score_band        |            -0.0112083   |             0.496544    |           0.507753   |           0.151818  |
| Adverse-Credit |              24 | default_12m    | ltv                      |            -0.0948514   |             0.176664    |           0.271515   |           0.0811832 |
| Adverse-Credit |              24 | default_12m    | ltv_band                 |            -0.0432815   |             0.0947038   |           0.137985   |           0.0412577 |
| Adverse-Credit |              24 | default_12m    | balance_ratio            |            -0.0760288   |            -0.134738    |          -0.0587097  |           0.0175542 |

_Showing 40 of 150 rows._


### How far each stressed feature moved

The attribution says which feature mattered; this says what happened to it.

| scenario        |   horizon_month | feature       |   baseline_mean |   stressed_mean |     change |   pct_change |
|:----------------|----------------:|:--------------|----------------:|----------------:|-----------:|-------------:|
| Adverse-Credit  |               1 | credit_score  |      700.612    |      700.612    |    0       |   0          |
| Adverse-Credit  |               1 | ltv           |       74.9118   |       74.9118   |    0       |   0          |
| Adverse-Credit  |               1 | rate_spread   |       -1.60551  |       -1.60551  |    0       |   0          |
| Adverse-Credit  |               1 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| Adverse-Credit  |               1 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| Adverse-Credit  |              12 | credit_score  |      700.612    |      607.396    |  -93.2159  |  -0.133049   |
| Adverse-Credit  |              12 | ltv           |       72.9093   |       80.862    |    7.95276 |   0.109077   |
| Adverse-Credit  |              12 | rate_spread   |       -1.40661  |       -1.86291  |   -0.4563  |   0.324396   |
| Adverse-Credit  |              12 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| Adverse-Credit  |              12 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| Adverse-Credit  |              24 | credit_score  |      700.612    |      504.265    | -196.348   |  -0.280251   |
| Adverse-Credit  |              24 | ltv           |       70.7857   |       87.8935   |   17.1078  |   0.241684   |
| Adverse-Credit  |              24 | rate_spread   |       -1.18951  |       -2.14381  |   -0.9543  |   0.802261   |
| Adverse-Credit  |              24 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| Adverse-Credit  |              24 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| Adverse-Credit  |              36 | credit_score  |      700.612    |      504.265    | -196.348   |  -0.280251   |
| Adverse-Credit  |              36 | ltv           |       68.724    |       95.5364   |   26.8124  |   0.390147   |
| Adverse-Credit  |              36 | rate_spread   |       -0.972513 |       -2.42461  |   -1.4521  |   1.49314    |
| Adverse-Credit  |              36 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| Adverse-Credit  |              36 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| Adverse-Credit  |              48 | credit_score  |      700.612    |      504.265    | -196.348   |  -0.280251   |
| Adverse-Credit  |              48 | ltv           |       66.7223   |      103.844    |   37.1217  |   0.556361   |
| Adverse-Credit  |              48 | rate_spread   |       -0.755513 |       -2.70551  |   -1.95    |   2.58103    |
| Adverse-Credit  |              48 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| Adverse-Credit  |              48 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| High-Prepayment |               1 | credit_score  |      700.612    |      700.612    |    0       |   0          |
| High-Prepayment |               1 | ltv           |       74.9118   |       74.9118   |    0       |   0          |
| High-Prepayment |               1 | rate_spread   |       -1.60551  |       -1.60551  |    0       |   0          |
| High-Prepayment |               1 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| High-Prepayment |               1 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| High-Prepayment |              12 | credit_score  |      700.612    |      707.11     |    6.49765 |   0.00927425 |
| High-Prepayment |              12 | ltv           |       72.9093   |       71.324    |   -1.58528 |  -0.0217432  |
| High-Prepayment |              12 | rate_spread   |       -1.40661  |       -1.04381  |    0.3628  |  -0.257924   |
| High-Prepayment |              12 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| High-Prepayment |              12 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |
| High-Prepayment |              24 | credit_score  |      700.612    |      714.6      |   13.9874  |   0.0199645  |
| High-Prepayment |              24 | ltv           |       70.7857   |       67.6056   |   -3.18004 |  -0.044925   |
| High-Prepayment |              24 | rate_spread   |       -1.18951  |       -0.431013 |    0.7585  |  -0.637656   |
| High-Prepayment |              24 | dti           |       35.8055   |       35.8055   |    0       |   0          |
| High-Prepayment |              24 | interest_rate |        4.99449  |        4.99449  |    0       |   0          |

_Showing 40 of 50 rows._


## Figures

**Projected rates by scenario**

![Projected rates by scenario](scenario/projection_paths.png)

**Segment impact under stress**

![Segment impact under stress](scenario/segment_impact.png)

**What moves each scenario**

![What moves each scenario](scenario/scenario_drivers.png)

