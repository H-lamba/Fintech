# Explainability & Responsible AI Report (Task 6)

_Generated 2026-08-31 18:15:56_

## What this covers

Global and local explanations for the three Phase 3 binary models, an analysis of where they are wrong and on what kind of loan, reliability of the predicted probabilities, and a disparity screen across borrower segments.


## Scope

**What SHAP explains here.** `TreeExplainer` decomposes the **booster's
log-odds**, not the calibrated probability the pipeline deploys. The calibrator
is a monotone transform fitted on top, so it cannot reorder feature
contributions -- an attribution that says credit score dominates is true of the
deployed model too -- but the additive decomposition sums to the base model's
log-odds. A report that claimed these values explain the deployed probability
would be claiming something the arithmetic does not support.

**What runs on the deployed model instead.** Error analysis, reliability and
the disparity screen all use the **calibrated probability at the threshold Task
2 tuned on validation**, because a false positive is a decision and the
decision is made there.

**On sampling.** `tree_path_dependent` perturbation needs no background
dataset -- it walks the trees using cover counts already stored in the model.
The full 58k-row test set computes in about three seconds. Rows are still
sampled, for scale headroom on a larger pack and because a beeswarm of 58,000
points is a solid block of ink, not because of a memory limit at this size.
Sampling is stratified on the outcome: the positives are 8-11% of the panel,
and a uniform sample of a rare class explains mostly non-events.


## SHAP values cross-checked against the booster

`shap` and LightGBM's own `pred_contrib` both run TreeSHAP, so they must agree exactly. Where they do not, one is being handed a different matrix than the other -- a stale category encoding, a reordered column -- and every attribution in this report would be describing a model that was never scored.

| model       |   rows_explained | checked   | agrees   |   max_abs_difference |
|:------------|-----------------:|:----------|:---------|---------------------:|
| delinquency |            20000 | True      | True     |                    0 |
| default     |            20000 | True      | True     |                    0 |
| prepayment  |            20000 | True      | True     |                    0 |


## Global feature importance

Mean absolute SHAP per feature. `mean_signed_shap` carries direction: a feature can be important and directionless.

| model       | feature                  |   mean_abs_shap |   mean_signed_shap |   max_abs_shap |      share |
|:------------|:-------------------------|----------------:|-------------------:|---------------:|-----------:|
| delinquency | credit_score             |       0.261831  |       -0.0371765   |       1.0129   | 0.150367   |
| delinquency | months_since_delinquency |       0.19881   |        0.00425483  |       2.23804  | 0.114175   |
| delinquency | credit_score_band        |       0.192148  |       -0.0126956   |       0.436958 | 0.110349   |
| delinquency | ltv                      |       0.124331  |        0.00481164  |       0.649523 | 0.0714023  |
| delinquency | state                    |       0.104798  |        0.00334655  |       0.513182 | 0.0601847  |
| delinquency | days_past_due            |       0.10244   |        0.00301615  |       1.36508  | 0.0588307  |
| delinquency | interest_rate            |       0.0523732 |        0.0149828   |       0.300945 | 0.0300775  |
| delinquency | rate_spread              |       0.0467605 |        0.0172318   |       0.33263  | 0.0268542  |
| delinquency | paydown_6m               |       0.0435744 |       -0.0172271   |       0.321473 | 0.0250244  |
| delinquency | paydown_3m               |       0.0359894 |       -0.0212244   |       0.540418 | 0.0206684  |
| delinquency | balance_vs_expected      |       0.0336978 |       -0.0113472   |       0.32494  | 0.0193524  |
| delinquency | status_ordinal           |       0.0328398 |       -0.00220856  |       0.407635 | 0.0188597  |
| delinquency | dti                      |       0.0291384 |        0.00468483  |       0.252283 | 0.016734   |
| delinquency | status_changes_to_date   |       0.0246533 |       -0.0123508   |       0.312543 | 0.0141582  |
| delinquency | paydown_1m               |       0.0237417 |       -0.00287259  |       0.20173  | 0.0136347  |
| default     | credit_score             |       0.994727  |       -0.117112    |       3.13012  | 0.282578   |
| default     | credit_score_band        |       0.355831  |       -0.0334042   |       0.754554 | 0.101083   |
| default     | ltv                      |       0.304477  |       -0.0108322   |       1.15297  | 0.0864944  |
| default     | state                    |       0.258049  |        0.0197637   |       1.41024  | 0.0733054  |
| default     | days_past_due            |       0.252107  |        0.0028738   |       3.77149  | 0.0716174  |
| default     | ltv_band                 |       0.211241  |        0.0188715   |       0.818437 | 0.0600084  |
| default     | interest_rate            |       0.132073  |       -0.0458953   |       0.748477 | 0.0375188  |
| default     | loan_age_months          |       0.115808  |        0.0776809   |       0.389903 | 0.0328981  |
| default     | dti                      |       0.0938003 |        0.0229448   |       0.665044 | 0.0266464  |
| default     | rate_spread              |       0.0687858 |        0.0401955   |       0.395147 | 0.0195404  |
| default     | status_ordinal           |       0.0645453 |        0.000598767 |       0.875162 | 0.0183357  |
| default     | current_balance          |       0.0539266 |        0.00310558  |       0.503197 | 0.0153192  |
| default     | original_balance         |       0.0444782 |        0.00871413  |       0.731696 | 0.0126352  |
| default     | paydown_3m               |       0.0425445 |       -0.0286169   |       0.269124 | 0.0120859  |
| default     | paydown_6m               |       0.0340397 |       -0.0163023   |       0.231235 | 0.00966984 |
| prepayment  | credit_score             |       0.209594  |        0.00635883  |       1.08457  | 0.191998   |
| prepayment  | state                    |       0.175027  |       -0.00389735  |       0.94612  | 0.160333   |
| prepayment  | interest_rate            |       0.110878  |       -0.0721465   |       0.552104 | 0.101569   |
| prepayment  | ltv                      |       0.0738733 |        0.00865332  |       0.777751 | 0.0676713  |
| prepayment  | dti                      |       0.0624214 |       -0.000388146 |       0.558417 | 0.0571808  |
| prepayment  | original_balance         |       0.0532678 |        0.00407071  |       0.465757 | 0.0487958  |
| prepayment  | dpd_max_12m              |       0.0399607 |       -0.00175381  |       0.476806 | 0.0366058  |
| prepayment  | rate_spread              |       0.0323703 |       -0.00474202  |       0.304977 | 0.0296527  |
| prepayment  | current_balance          |       0.0313384 |       -0.00309711  |       0.533851 | 0.0287073  |
| prepayment  | loan_purpose             |       0.0286223 |        0.000813348 |       0.231476 | 0.0262193  |
| prepayment  | balance_pctile_in_month  |       0.021153  |        0.00177294  |       0.180882 | 0.0193771  |
| prepayment  | dpd_mean_12m             |       0.0210422 |        0.00146339  |       0.444628 | 0.0192756  |
| prepayment  | paydown_6m               |       0.0193005 |       -0.00892222  |       0.214885 | 0.0176801  |
| prepayment  | term_progress            |       0.0185289 |        0.0163174   |       0.197244 | 0.0169733  |
| prepayment  | servicer_name            |       0.0177533 |       -8.16107e-05 |       0.099533 | 0.0162628  |


## Loans selected for local explanation

Deliberately not a highlight reel: a confident hit, a confident false positive, a missed event and a borderline case. Showing only the confident hit would demonstrate the model on the records where nothing was in doubt.

| model       | case                          |   position | loan_id      | reporting_month     |   predicted_probability |   actual_outcome |
|:------------|:------------------------------|-----------:|:-------------|:--------------------|------------------------:|-----------------:|
| delinquency | confident true positive       |       2364 | 4HXZZPFHCDEG | 2023-02-01 00:00:00 |               1         |                1 |
| delinquency | confident false positive      |       1890 | 3MHIKIMSXP0D | 2023-08-01 00:00:00 |               0.967347  |                0 |
| delinquency | missed event (false negative) |       1657 | 35ZTHLVR00QE | 2023-10-01 00:00:00 |               0.0254777 |                1 |
| delinquency | borderline                    |      13941 | PE5PIWZQ3NS9 | 2023-04-01 00:00:00 |               0.524301  |                0 |
| default     | confident true positive       |        877 | 1R1YNCRG5YFD | 2023-05-01 00:00:00 |               1         |                1 |
| default     | confident false positive      |      11230 | KHK4PGKG9F2Q | 2023-01-01 00:00:00 |               0.982456  |                0 |
| default     | missed event (false negative) |       5415 | 9X5K386P97VI | 2023-06-01 00:00:00 |               0         |                1 |
| default     | borderline                    |         68 | 04BINLNBCBQ9 | 2023-02-01 00:00:00 |               0.470588  |                0 |
| prepayment  | missed event (false negative) |       6616 | CD0OSLUPND2V | 2023-10-01 00:00:00 |               0.0128205 |                1 |


### Local explanation: delinquency -- confident true positive

| feature                  | feature_value   |   shap_value | direction   |
|:-------------------------|:----------------|-------------:|:------------|
| months_since_delinquency | 0.0             |    1.82439   | raises risk |
| days_past_due            | 111.0           |    0.976318  | raises risk |
| status_ordinal           | 4.0             |    0.285202  | raises risk |
| credit_score             | 660.0           |    0.185628  | raises risk |
| dpd_lag_1m               | 82.0            |    0.167274  | raises risk |
| is_delinquent            | 1.0             |    0.146762  | raises risk |
| dpd_mean_12m             | 31.5            |    0.118817  | raises risk |
| ltv                      | 82.14           |    0.114697  | raises risk |
| credit_score_band        | 620-659         |    0.101023  | raises risk |
| dpd_delta_1m             | 29.0            |    0.0888188 | raises risk |
| modified_ever            | 1.0             |    0.071197  | raises risk |
| dpd_mean_3m              | 76.0            |    0.0649747 | raises risk |


### Local explanation: delinquency -- confident false positive

| feature                  | feature_value   |   shap_value | direction   |
|:-------------------------|:----------------|-------------:|:------------|
| months_since_delinquency | 0.0             |    1.7892    | raises risk |
| days_past_due            | 104.0           |    1.10739   | raises risk |
| status_ordinal           | 4.0             |    0.277659  | raises risk |
| is_delinquent            | 1.0             |    0.217951  | raises risk |
| ltv                      | 92.3            |    0.177893  | raises risk |
| dpd_lag_1m               | 78.0            |    0.167451  | raises risk |
| dpd_mean_3m              | 74.0            |    0.117245  | raises risk |
| dpd_delta_1m             | 26.0            |    0.0912602 | raises risk |
| credit_score_band        | 660-699         |    0.0897913 | raises risk |
| credit_score             | 668.0           |    0.0677566 | raises risk |
| paydown_1m               | 0.0             |    0.0627704 | raises risk |
| dpd_max_12m              | 104.0           |   -0.0601744 | lowers risk |


### Local explanation: delinquency -- missed event (false negative)

| feature                  | feature_value         |   shap_value | direction   |
|:-------------------------|:----------------------|-------------:|:------------|
| paydown_6m               | -0.006773996037529706 |   -0.271823  | lowers risk |
| credit_score_band        | 700-739               |   -0.212108  | lowers risk |
| state                    | MN                    |   -0.196792  | lowers risk |
| credit_score             | 704.0                 |   -0.160116  | lowers risk |
| months_since_delinquency | nan                   |   -0.119564  | lowers risk |
| ltv                      | 68.1                  |   -0.108184  | lowers risk |
| dpd_mean_6m              | 15.5                  |   -0.0729915 | lowers risk |
| dpd_max_12m              | 52.0                  |   -0.0621524 | lowers risk |
| days_past_due            | 0.0                   |   -0.0607553 | lowers risk |
| dpd_max_6m               | 52.0                  |   -0.0597438 | lowers risk |
| balance_pctile_in_month  | 0.2424511545293073    |   -0.0383285 | lowers risk |
| paydown_3m               | -0.005098292000316529 |   -0.0327646 | lowers risk |


### Local explanation: delinquency -- borderline

| feature                  | feature_value   |   shap_value | direction   |
|:-------------------------|:----------------|-------------:|:------------|
| months_since_delinquency | 0.0             |    1.7918    | raises risk |
| days_past_due            | 49.0            |    0.742902  | raises risk |
| credit_score             | 745.0           |   -0.3666    | lowers risk |
| ltv                      | 85.13           |    0.203199  | raises risk |
| status_ordinal           | 2.0             |    0.186236  | raises risk |
| is_delinquent            | 1.0             |    0.16989   | raises risk |
| credit_score_band        | 740-799         |   -0.120992  | lowers risk |
| dpd_delta_1m             | 3.0             |    0.0932278 | raises risk |
| paydown_1m               | 0.0             |    0.0603657 | raises risk |
| state                    | CA              |    0.0592597 | raises risk |
| dpd_lag_1m               | 46.0            |    0.0565237 | raises risk |
| current_status           | 30-DPD          |    0.0562841 | raises risk |


### Local explanation: default -- confident true positive

| feature           | feature_value   |   shap_value | direction   |
|:------------------|:----------------|-------------:|:------------|
| days_past_due     | 99.0            |    2.29895   | raises risk |
| credit_score      | 642.0           |    1.29285   | raises risk |
| status_ordinal    | 4.0             |    0.459638  | raises risk |
| credit_score_band | 620-659         |    0.33767   | raises risk |
| ltv               | 86.27           |    0.263322  | raises risk |
| dpd_lag_1m        | 66.0            |    0.141893  | raises risk |
| is_delinquent     | 1.0             |    0.118926  | raises risk |
| dpd_delta_3m      | 99.0            |    0.110123  | raises risk |
| dpd_delta_1m      | 33.0            |    0.104553  | raises risk |
| ltv_band          | 80-90%          |    0.0926019 | raises risk |
| interest_rate     | 6.33            |   -0.0691723 | lowers risk |
| paydown_1m        | 0.0             |    0.058505  | raises risk |


### Local explanation: default -- confident false positive

| feature           | feature_value   |   shap_value | direction   |
|:------------------|:----------------|-------------:|:------------|
| days_past_due     | 88.0            |    2.1025    | raises risk |
| credit_score      | 585.0           |    1.56921   | raises risk |
| status_ordinal    | 3.0             |    0.471213  | raises risk |
| credit_score_band | <620            |    0.38518   | raises risk |
| is_delinquent     | 1.0             |    0.130558  | raises risk |
| dpd_delta_3m      | 88.0            |    0.105048  | raises risk |
| ltv               | 74.89           |   -0.0902492 | lowers risk |
| dpd_delta_1m      | 49.0            |    0.0845844 | raises risk |
| dpd_lag_1m        | 39.0            |    0.0791556 | raises risk |
| state             | TX              |    0.0773258 | raises risk |
| paydown_1m        | 0.0             |    0.0712299 | raises risk |
| dti               | 48.24           |    0.0708363 | raises risk |


### Local explanation: default -- missed event (false negative)

| feature           | feature_value       |   shap_value | direction   |
|:------------------|:--------------------|-------------:|:------------|
| credit_score      | 707.0               |   -0.858878  | lowers risk |
| state             | AL                  |   -0.61348   | lowers risk |
| credit_score_band | 700-739             |   -0.283511  | lowers risk |
| ltv               | 69.63               |   -0.259654  | lowers risk |
| interest_rate     | 7.172               |   -0.21289   | lowers risk |
| ltv_band          | 60-75%              |   -0.211729  | lowers risk |
| rate_spread       | 2.6125              |    0.170481  | raises risk |
| days_past_due     | 0.0                 |   -0.13883   | lowers risk |
| dti               | 32.51               |   -0.0833961 | lowers risk |
| loan_purpose      | Rate/Term Refinance |   -0.0583576 | lowers risk |
| loan_age_months   | 1.0                 |   -0.0556295 | lowers risk |
| balance_ratio     | 0.9992077205882353  |    0.0399635 | raises risk |


### Local explanation: default -- borderline

| feature                | feature_value          |   shap_value | direction   |
|:-----------------------|:-----------------------|-------------:|:------------|
| days_past_due          | 49.0                   |    1.45987   | raises risk |
| credit_score           | 660.0                  |    1.44369   | raises risk |
| status_ordinal         | 2.0                    |    0.444955  | raises risk |
| credit_score_band      | 620-659                |    0.401746  | raises risk |
| state                  | AZ                     |    0.220722  | raises risk |
| interest_rate          | 6.255                  |   -0.164743  | lowers risk |
| ltv                    | 73.14                  |   -0.131708  | lowers risk |
| is_delinquent          | 1.0                    |    0.115869  | raises risk |
| dpd_delta_3m           | 49.0                   |    0.07474   | raises risk |
| paydown_1m             | -0.0010083080590768123 |    0.0612568 | raises risk |
| rate_spread            | 1.8149999999999995     |    0.0606148 | raises risk |
| status_changes_to_date | 3.0                    |    0.0603587 | raises risk |


### Local explanation: prepayment -- missed event (false negative)

| feature          | feature_value          |   shap_value | direction   |
|:-----------------|:-----------------------|-------------:|:------------|
| credit_score     | 638.0                  |   -0.91551   | lowers risk |
| dpd_mean_12m     | 31.583333333333332     |   -0.317379  | lowers risk |
| state            | MN                     |    0.259583  | raises risk |
| dpd_max_6m       | 84.0                   |   -0.251222  | lowers risk |
| dpd_max_12m      | 84.0                   |   -0.22652   | lowers risk |
| interest_rate    | 6.18                   |   -0.135773  | lowers risk |
| original_balance | 266000.0               |    0.0645554 | raises risk |
| ltv              | 76.71                  |    0.0339022 | raises risk |
| paydown_6m       | -0.0010162349921574165 |   -0.0328994 | lowers risk |
| dpd_mean_6m      | 42.5                   |   -0.0238909 | lowers risk |
| rate_spread      | 1.3594999999999997     |    0.0147751 | raises risk |
| current_balance  | 263381.72              |    0.0146361 | raises risk |


## Confusion summary

At the threshold tuned in Task 2.

| model       | outcome        |   records |     share |
|:------------|:---------------|----------:|----------:|
| delinquency | true positive  |      2547 | 0.039456  |
| delinquency | false positive |      1050 | 0.0162657 |
| delinquency | false negative |      4640 | 0.0718789 |
| delinquency | true negative  |     56316 | 0.872399  |
| default     | true positive  |      2543 | 0.0435579 |
| default     | false positive |      3100 | 0.0530986 |
| default     | false negative |      2520 | 0.043164  |
| default     | true negative  |     50219 | 0.86018   |
| prepayment  | true positive  |      3924 | 0.0672125 |
| prepayment  | false positive |     37771 | 0.646963  |
| prepayment  | false negative |      1304 | 0.0223357 |
| prepayment  | true negative  |     15383 | 0.263489  |


## Error rates by segment

Groups below the minimum size are dropped: a false positive rate computed on nine loans is noise with a decimal point, and putting it in a governance table invites someone to act on it.

| model       | segment           | group   |   records |   actual_positives |   actual_negatives |   flagged |   false_positives |   false_negatives |   true_positives |   selection_rate |   precision |    recall |   false_positive_rate |   false_negative_rate |
|:------------|:------------------|:--------|----------:|-------------------:|-------------------:|----------:|------------------:|------------------:|-----------------:|-----------------:|------------:|----------:|----------------------:|----------------------:|
| delinquency | credit_score_band | 620-659 |      7173 |               1744 |               5429 |       950 |               144 |               938 |              806 |        0.132441  |    0.848421 | 0.462156  |            0.0265242  |              0.537844 |
| delinquency | credit_score_band | 660-699 |     18417 |               2507 |              15910 |      1261 |               375 |              1621 |              886 |        0.0684693 |    0.702617 | 0.35341   |            0.0235701  |              0.64659  |
| delinquency | credit_score_band | <620    |      2249 |                650 |               1599 |       344 |                37 |               343 |              307 |        0.152957  |    0.892442 | 0.472308  |            0.0231395  |              0.527692 |
| delinquency | credit_score_band | 700-739 |     20820 |               1539 |              19281 |       725 |               329 |              1143 |              396 |        0.0348223 |    0.546207 | 0.25731   |            0.0170634  |              0.74269  |
| delinquency | credit_score_band | 740-799 |     14117 |                693 |              13424 |       298 |               151 |               546 |              147 |        0.0211093 |    0.493289 | 0.212121  |            0.0112485  |              0.787879 |
| delinquency | credit_score_band | 800+    |      1777 |                 54 |               1723 |        19 |                14 |                49 |                5 |        0.0106922 |    0.263158 | 0.0925926 |            0.00812536 |              0.907407 |
| delinquency | ltv_band          | 80-90%  |     13895 |               2294 |              11601 |      1197 |               274 |              1371 |              923 |        0.0861461 |    0.771094 | 0.402354  |            0.0236187  |              0.597646 |
| delinquency | ltv_band          | 90-97%  |      3037 |                738 |               2299 |       382 |                54 |               410 |              328 |        0.125782  |    0.858639 | 0.444444  |            0.0234885  |              0.555556 |
| delinquency | ltv_band          | 75-80%  |     12435 |               1426 |              11009 |       717 |               217 |               926 |              500 |        0.0576598 |    0.69735  | 0.350631  |            0.0197111  |              0.649369 |
| delinquency | ltv_band          | 60-75%  |     30370 |               2503 |              27867 |      1187 |               447 |              1763 |              740 |        0.0390846 |    0.62342  | 0.295645  |            0.0160405  |              0.704355 |
| delinquency | ltv_band          | <60%    |      4816 |                226 |               4590 |       114 |                58 |               170 |               56 |        0.0236711 |    0.491228 | 0.247788  |            0.0126362  |              0.752212 |
| delinquency | vintage_year      | 2018    |      2298 |                272 |               2026 |       146 |                45 |               171 |              101 |        0.0635335 |    0.691781 | 0.371324  |            0.0222113  |              0.628676 |
| delinquency | vintage_year      | 2019    |      9531 |                958 |               8573 |       537 |               169 |               590 |              368 |        0.0563425 |    0.685289 | 0.384134  |            0.0197131  |              0.615866 |
| delinquency | vintage_year      | 2021    |     14160 |               1575 |              12585 |       771 |               242 |              1046 |              529 |        0.0544492 |    0.686122 | 0.335873  |            0.0192292  |              0.664127 |
| delinquency | vintage_year      | 2020    |     11469 |               1193 |              10276 |       638 |               195 |               750 |              443 |        0.0556282 |    0.694357 | 0.371333  |            0.0189763  |              0.628667 |
| delinquency | vintage_year      | 2022    |     17090 |               1989 |              15101 |      1016 |               279 |              1252 |              737 |        0.05945   |    0.725394 | 0.370538  |            0.0184756  |              0.629462 |
| delinquency | vintage_year      | 2023    |     10005 |               1200 |               8805 |       489 |               120 |               831 |              369 |        0.0488756 |    0.754601 | 0.3075    |            0.0136286  |              0.6925   |
| delinquency | state             | MN      |      1254 |                183 |               1071 |        87 |                28 |               124 |               59 |        0.069378  |    0.678161 | 0.322404  |            0.0261438  |              0.677596 |
| delinquency | state             | AL      |       952 |                 91 |                861 |        58 |                22 |                55 |               36 |        0.0609244 |    0.62069  | 0.395604  |            0.0255517  |              0.604396 |
| delinquency | state             | MA      |      1581 |                184 |               1397 |       100 |                33 |               117 |               67 |        0.0632511 |    0.67     | 0.36413   |            0.023622   |              0.63587  |
| delinquency | state             | WA      |      1815 |                197 |               1618 |        89 |                37 |               145 |               52 |        0.0490358 |    0.58427  | 0.263959  |            0.0228677  |              0.736041 |
| delinquency | state             | CA      |      8064 |                950 |               7114 |       491 |               158 |               617 |              333 |        0.0608879 |    0.678208 | 0.350526  |            0.0222097  |              0.649474 |
| delinquency | state             | UT      |       569 |                 47 |                522 |        25 |                11 |                33 |               14 |        0.0439367 |    0.56     | 0.297872  |            0.0210728  |              0.702128 |
| delinquency | state             | AZ      |      1652 |                179 |               1473 |        87 |                31 |               123 |               56 |        0.0526634 |    0.643678 | 0.312849  |            0.0210455  |              0.687151 |
| delinquency | state             | TX      |      6519 |                759 |               5760 |       379 |               116 |               496 |              263 |        0.0581378 |    0.693931 | 0.346509  |            0.0201389  |              0.653491 |
| delinquency | state             | MO      |      1433 |                181 |               1252 |        81 |                25 |               125 |               56 |        0.0565248 |    0.691358 | 0.309392  |            0.0199681  |              0.690608 |
| delinquency | state             | GA      |      2369 |                269 |               2100 |       132 |                40 |               177 |               92 |        0.0557197 |    0.69697  | 0.342007  |            0.0190476  |              0.657993 |
| delinquency | state             | LA      |       849 |                108 |                741 |        52 |                14 |                70 |               38 |        0.0612485 |    0.730769 | 0.351852  |            0.0188934  |              0.648148 |
| delinquency | state             | IN      |      1280 |                157 |               1123 |        81 |                21 |                97 |               60 |        0.0632812 |    0.740741 | 0.382166  |            0.0186999  |              0.617834 |
| delinquency | state             | NJ      |      1849 |                217 |               1632 |       117 |                30 |               130 |               87 |        0.0632774 |    0.74359  | 0.400922  |            0.0183824  |              0.599078 |
| delinquency | state             | CT      |       622 |                 66 |                556 |        29 |                10 |                47 |               19 |        0.0466238 |    0.655172 | 0.287879  |            0.0179856  |              0.712121 |
| delinquency | state             | NY      |      3862 |                362 |               3500 |       168 |                62 |               256 |              106 |        0.0435008 |    0.630952 | 0.292818  |            0.0177143  |              0.707182 |
| delinquency | state             | KY      |       910 |                115 |                795 |        61 |                14 |                68 |               47 |        0.067033  |    0.770492 | 0.408696  |            0.0176101  |              0.591304 |
| delinquency | state             | OR      |      1057 |                 90 |                967 |        41 |                17 |                66 |               24 |        0.038789  |    0.585366 | 0.266667  |            0.0175801  |              0.733333 |
| delinquency | state             | IL      |      2629 |                274 |               2355 |       141 |                41 |               174 |              100 |        0.0536326 |    0.70922  | 0.364964  |            0.0174098  |              0.635036 |
| delinquency | state             | VA      |      1903 |                212 |               1691 |       106 |                29 |               135 |               77 |        0.0557015 |    0.726415 | 0.363208  |            0.0171496  |              0.636792 |
| delinquency | state             | PA      |      2687 |                351 |               2336 |       195 |                40 |               196 |              155 |        0.0725716 |    0.794872 | 0.441595  |            0.0171233  |              0.558405 |
| delinquency | state             | SC      |       905 |                 78 |                827 |        38 |                14 |                54 |               24 |        0.041989  |    0.631579 | 0.307692  |            0.0169287  |              0.692308 |
| delinquency | state             | FL      |      6649 |                755 |               5894 |       368 |                98 |               485 |              270 |        0.0553467 |    0.733696 | 0.357616  |            0.0166271  |              0.642384 |
| delinquency | state             | MI      |      2115 |                265 |               1850 |       145 |                30 |               150 |              115 |        0.0685579 |    0.793103 | 0.433962  |            0.0162162  |              0.566038 |
| delinquency | state             | TN      |      1485 |                148 |               1337 |        69 |                21 |               100 |               48 |        0.0464646 |    0.695652 | 0.324324  |            0.0157068  |              0.675676 |
| delinquency | state             | NC      |      2294 |                259 |               2035 |       129 |                30 |               160 |               99 |        0.0562337 |    0.767442 | 0.382239  |            0.014742   |              0.617761 |
| delinquency | state             | WI      |      1335 |                122 |               1213 |        59 |                16 |                79 |               43 |        0.0441948 |    0.728814 | 0.352459  |            0.0131904  |              0.647541 |
| delinquency | state             | MD      |      1275 |                125 |               1150 |        60 |                15 |                80 |               45 |        0.0470588 |    0.75     | 0.36      |            0.0130435  |              0.64     |
| delinquency | state             | OK      |       847 |                 91 |                756 |        46 |                 9 |                54 |               37 |        0.0543093 |    0.804348 | 0.406593  |            0.0119048  |              0.593407 |

_Showing 45 of 156 rows._


## What a false positive looks like

Mean feature value among false positives against true negatives, standardised by the spread. A large gap says the model is flagging loans that look like *this*, which is the actionable form of an error analysis.

| model       | feature                   |   false_positive_mean |   true_negative_mean |   standardised_gap_fp_vs_tn |   false_negative_mean |   true_positive_mean |   standardised_gap_fn_vs_tp |
|:------------|:--------------------------|----------------------:|---------------------:|----------------------------:|----------------------:|---------------------:|----------------------------:|
| delinquency | is_delinquent             |              1        |           0.00708502 |                    4.10408  |            0.00732759 |             1        |                   -4.10308  |
| delinquency | status_changed_this_month |              0.794286 |           0.0340578  |                    2.85286  |            0.0351293  |             0.805261 |                   -2.89003  |
| delinquency | days_past_due             |             49.7371   |           0.716138   |                    2.77343  |            0.704526   |            69.0667   |                   -3.86768  |
| delinquency | status_ordinal            |              2.1781   |           1.02051    |                    2.42018  |            1.01983    |             2.82097  |                   -3.76566  |
| delinquency | dpd_delta_6m              |             46.6209   |          -1.47912    |                    2.20004  |           -2.31112    |            66.7635   |                   -3.1594   |
| delinquency | dpd_delta_3m              |             44.9463   |          -1.6436     |                    2.19407  |           -1.85161    |            57.9162   |                   -2.81466  |
| delinquency | dpd_max_3m                |             53.219    |           4.31437    |                    2.00239  |            4.13319    |            71.3086   |                   -2.75049  |
| delinquency | dpd_mean_3m               |             25.9033   |           1.59752    |                    1.87912  |            1.53574    |            44.6478   |                   -3.33306  |
| delinquency | dpd_delta_1m              |             30.7433   |          -1.11756    |                    1.82636  |           -1.10099    |            27.8992   |                   -1.66238  |
| delinquency | dpd_max_6m                |             54.4457   |           8.51511    |                    1.58314  |            8.91767    |            72.342    |                   -2.18612  |
| delinquency | dpd_mean_6m               |             15.3453   |           1.8796     |                    1.4701   |            2.02107    |            26.5904   |                   -2.68233  |
| delinquency | dpd_mean_12m              |              9.94252  |           1.90251    |                    1.27311  |            2.21455    |            16.8169   |                   -2.31224  |
| default     | credit_score              |            641.348    |         715.496      |                   -1.54925  |          668.755      |           646.151    |                    0.472286 |
| default     | is_delinquent             |              0.292258 |           0.018957   |                    1.10886  |            0.0396825  |             0.721589 |                   -2.76668  |
| default     | ltv                       |             81.4657   |          73.0919     |                    0.871824 |           79.6576     |            81.9048   |                   -0.23396  |
| default     | days_past_due             |             16.8874   |           1.20727    |                    0.8652   |            1.92222    |            54.965    |                   -2.9268   |
| default     | status_ordinal            |              1.41516  |           1.03152    |                    0.781509 |            1.04802    |             2.48289  |                   -2.92298  |
| default     | status_changed_this_month |              0.252903 |           0.0435094  |                    0.775913 |            0.0571429  |             0.591427 |                   -1.9798   |
| default     | dpd_delta_3m              |             13.6325   |          -1.11419    |                    0.683525 |           -0.877306   |            44.2255   |                   -2.09057  |
| default     | dti                       |             39.783    |          34.8305     |                    0.66135  |           38.2199     |            39.2072   |                   -0.13184  |
| default     | dpd_delta_6m              |             13.7217   |          -0.942981   |                    0.65743  |           -0.94033    |            51.7691   |                   -2.36301  |
| default     | dpd_mean_3m               |             10.0847   |           1.81474    |                    0.621643 |            2.04874    |            37.7767   |                   -2.68563  |
| default     | dpd_max_3m                |             19.9881   |           4.81137    |                    0.611276 |            5.12381    |            58.0484   |                   -2.13166  |
| default     | dpd_delta_1m              |              8.52087  |          -0.735607   |                    0.524906 |           -0.0867197  |            18.6424   |                   -1.06207  |
| prepayment  | credit_score              |            714.793    |         683.534      |                    0.653144 |          699.092      |           719.469    |                   -0.425768 |
| prepayment  | dpd_mean_12m              |              1.68953  |           5.27424    |                   -0.554818 |            3.9224     |             1.65368  |                    0.351138 |
| prepayment  | dpd_mean_6m               |              1.80464  |           6.75941    |                   -0.527152 |            4.13264    |             1.82011  |                    0.246036 |
| prepayment  | dpd_mean_3m               |              2.01633  |           8.76713    |                   -0.507451 |            4.03106    |             1.87198  |                    0.162295 |
| prepayment  | dpd_max_12m               |             13.3029   |          30.4734     |                   -0.500446 |           24.2155     |            13.6697   |                    0.307363 |
| prepayment  | dpd_max_6m                |              7.8879   |          22.274      |                   -0.492241 |           15.9394     |             8.71662  |                    0.247139 |

_Showing 30 of 36 rows._


## Calibration

Expected calibration error is the population-weighted mean gap between predicted and observed, so a wild miss in a bin holding four records does not outweigh a small bias across the bulk of the book.

| model       |   expected_calibration_error |   mean_predicted |   observed_rate |         bias |
|:------------|-----------------------------:|-----------------:|----------------:|-------------:|
| delinquency |                   0.00382324 |        0.111389  |       0.111335  |  5.42444e-05 |
| default     |                   0.00512053 |        0.082415  |       0.0867219 | -0.0043069   |
| prepayment  |                   0.0129867  |        0.0832439 |       0.0895481 | -0.0063042   |


### Reliability by probability bin

| bin           |   records |   mean_predicted |   observed_rate |          gap | model       |
|:--------------|----------:|-----------------:|----------------:|-------------:|:------------|
| (-0.001, 0.1] |     45898 |        0.057157  |       0.0592836 | -0.00212661  | delinquency |
| (0.1, 0.2]    |     14447 |        0.132002  |       0.125286  |  0.00671625  | delinquency |
| (0.2, 0.3]    |       699 |        0.219413  |       0.195994  |  0.0234187   | delinquency |
| (0.3, 0.4]    |       171 |        0.387586  |       0.409357  | -0.0217703   | delinquency |
| (0.4, 0.5]    |       505 |        0.452358  |       0.429703  |  0.0226555   | delinquency |
| (0.5, 0.6]    |       444 |        0.530224  |       0.533784  | -0.00356002  | delinquency |
| (0.6, 0.7]    |       402 |        0.648811  |       0.651741  | -0.00293044  | delinquency |
| (0.7, 0.8]    |       553 |        0.743418  |       0.75226   | -0.00884268  | delinquency |
| (0.8, 0.9]    |       658 |        0.847004  |       0.866261  | -0.0192579   | delinquency |
| (0.9, 1.0]    |       776 |        0.96303   |       0.962629  |  0.000401369 | delinquency |
| (-0.001, 0.1] |     44643 |        0.0252517 |       0.0284703 | -0.00321866  | default     |
| (0.1, 0.2]    |      7741 |        0.14098   |       0.149981  | -0.00900063  | default     |
| (0.2, 0.3]    |      3113 |        0.255581  |       0.261484  | -0.00590299  | default     |
| (0.3, 0.4]    |      1053 |        0.328502  |       0.330484  | -0.00198276  | default     |
| (0.4, 0.5]    |       129 |        0.44323   |       0.542636  | -0.099406    | default     |
| (0.5, 0.6]    |         3 |        0.537769  |       0.333333  |  0.204435    | default     |
| (0.6, 0.7]    |       579 |        0.669873  |       0.639033  |  0.0308399   | default     |
| (0.7, 0.8]    |       138 |        0.719471  |       0.797101  | -0.0776305   | default     |
| (0.8, 0.9]    |       349 |        0.845755  |       0.896848  | -0.0510928   | default     |
| (0.9, 1.0]    |       634 |        0.962588  |       0.954259  |  0.00832977  | default     |
| (-0.001, 0.1] |     41527 |        0.0753939 |       0.0889542 | -0.0135603   | prepayment  |
| (0.1, 0.2]    |     16651 |        0.100429  |       0.0912858 |  0.00914293  | prepayment  |
| (0.2, 0.3]    |       204 |        0.278571  |       0.0686275 |  0.209944    | prepayment  |


### Confidence profile

A model that never leaves a narrow band is technically calibrated and operationally useless: nothing is ever decided.

| model       | band                                 |     share |
|:------------|:-------------------------------------|----------:|
| delinquency | confident, flagged                   | 0.0438864 |
| delinquency | uncertain (within 0.05 of threshold) | 0.0017505 |
| delinquency | confident, cleared                   | 0.516227  |
| delinquency | max predicted probability            | 1         |
| default     | confident, flagged                   | 0.0313624 |
| default     | uncertain (within 0.05 of threshold) | 0.0586653 |
| default     | confident, cleared                   | 0.402898  |
| default     | max predicted probability            | 1         |
| prepayment  | confident, flagged                   | 0         |
| prepayment  | uncertain (within 0.05 of threshold) | 0.980456  |
| prepayment  | confident, cleared                   | 0         |
| prepayment  | max predicted probability            | 0.278571  |


## Disparity screen

**This is a disparity screen, not a legal fairness test**, and the distinction
is the most important sentence in this section.

The panel contains **no protected attribute** -- no race, sex, age or national
origin -- so no disparate-treatment or disparate-impact analysis in the legal
sense is possible from this data. What exists is geography and servicer, which
are coarse proxies at best, and credit characteristics, which are not proxies
at all.

**A credit-band gap is the model working, not failing.** A model that flagged
sub-620 and 800+ borrowers at the same rate would be broken. The credit-band
table is a *monotonicity check* -- does the flag rate fall as credit quality
rises -- not a fairness result, and it is labelled that way in the `kind`
column so nobody reads it as one.

**Geography is where a real question lives.** State is not a legitimate risk
factor in the way credit score is, and in US mortgage lending it correlates
with protected classes. The metric ranked here is therefore **error-rate parity
conditional on outcome**: among borrowers who did *not* default, is one state
flagged more often than another? That question has no legitimate risk-based
answer, which is what makes a gap in it worth escalating.

The 0.80 ratio floor is borrowed from the US "four-fifths rule" as a screening
trigger for a human to look. It is not a verdict, and nothing here should be
represented to a regulator as a compliance test.


## Disparity summary

| model       | segment           | kind                       | metric              |   groups_compared | worst_group                  |   worst_value |   worst_group_records |   worst_group_events | best_group                    |   best_value |   ratio_best_to_worst | below_floor   |      p_value | enough_events_to_test   | significant   | interpretable   | escalate   |   overall_selection_rate |
|:------------|:------------------|:---------------------------|:--------------------|------------------:|:-----------------------------|--------------:|----------------------:|---------------------:|:------------------------------|-------------:|----------------------:|:--------------|-------------:|:------------------------|:--------------|:----------------|:-----------|-------------------------:|
| delinquency | state             | screen for disparity       | false_positive_rate |                30 | MN                           |     0.0261438 |                  1254 |                   28 | CO                            |   0.00945378 |            0.361607   | True          | 0.00516981   | True                    | True          | True            | True       |                0.0557217 |
| delinquency | state             | screen for disparity       | selection_rate      |                30 | PA                           |     0.0725716 |                  2687 |                  195 | OR                            |   0.038789   |            0.534493   | True          | 0.000128843  | True                    | True          | True            | True       |                0.0557217 |
| delinquency | state             | screen for disparity       | false_negative_rate |                30 | WA                           |     0.736041  |                  1815 |                  145 | PA                            |   0.558405   |            0.75866    | True          | 3.85975e-05  | True                    | True          | True            | True       |                0.0557217 |
| delinquency | credit_score_band | risk factor (gap expected) | selection_rate      |                 6 | <620                         |     0.152957  |                  2249 |                  344 | 800+                          |   0.0106922  |            0.0699032  | True          | 3.34799e-55  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | ltv_band          | risk factor (gap expected) | selection_rate      |                 5 | 90-97%                       |     0.125782  |                  3037 |                  382 | <60%                          |   0.0236711  |            0.188191   | True          | 2.37921e-73  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | credit_score_band | risk factor (gap expected) | false_positive_rate |                 6 | 620-659                      |     0.0265242 |                  7173 |                  144 | 800+                          |   0.00812536 |            0.306337   | True          | 5.98112e-06  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | ltv_band          | risk factor (gap expected) | false_positive_rate |                 5 | 80-90%                       |     0.0236187 |                 13895 |                  274 | <60%                          |   0.0126362  |            0.535008   | True          | 8.82569e-06  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | credit_score_band | risk factor (gap expected) | false_negative_rate |                 6 | 800+                         |     0.907407  |                  1777 |                   49 | <620                          |   0.527692   |            0.581538   | True          | 6.76472e-08  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | vintage_year      | risk factor (gap expected) | false_positive_rate |                 6 | 2018                         |     0.0222113 |                  2298 |                   45 | 2023                          |   0.0136286  |            0.613591   | True          | 0.00445814   | True                    | True          | True            | False      |                0.0557217 |
| delinquency | ltv_band          | risk factor (gap expected) | false_negative_rate |                 5 | <60%                         |     0.752212  |                  4816 |                  170 | 90-97%                        |   0.555556   |            0.738562   | True          | 1.26498e-07  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | vintage_year      | risk factor (gap expected) | selection_rate      |                 6 | 2018                         |     0.0635335 |                  2298 |                  146 | 2023                          |   0.0488756  |            0.769288   | True          | 0.00418293   | True                    | True          | True            | False      |                0.0557217 |
| delinquency | servicer_name     | screen for disparity       | selection_rate      |                 5 | Meridian Residential Capital |     0.0609669 |                 12597 |                  768 | Northgate Financial Servicing |   0.0517351  |            0.848576   | False         | 0.00147815   | True                    | True          | True            | False      |                0.0557217 |
| delinquency | vintage_year      | risk factor (gap expected) | false_negative_rate |                 6 | 2023                         |     0.6925    |                 10005 |                  831 | 2019                          |   0.615866   |            0.889338   | False         | 0.000191618  | True                    | True          | True            | False      |                0.0557217 |
| delinquency | servicer_name     | screen for disparity       | false_positive_rate |                 5 | Atlas Mortgage Services      |     0.0196229 |                 13276 |                  230 | Northgate Financial Servicing |   0.0159485  |            0.812752   | False         | 0.0349243    | True                    | False         | True            | False      |                0.0557217 |
| delinquency | servicer_name     | screen for disparity       | false_negative_rate |                 5 | Beacon Home Loans            |     0.665956  |                 12641 |                  937 | Meridian Residential Capital  |   0.626915   |            0.941377   | False         | 0.0278208    | True                    | False         | True            | False      |                0.0557217 |
| default     | state             | screen for disparity       | false_positive_rate |                30 | WI                           |     0.121659  |                  1184 |                  132 | OK                            |   0.011396   |            0.0936718  | True          | 2.4208e-17   | True                    | True          | True            | True       |                0.0966565 |
| default     | state             | screen for disparity       | selection_rate      |                30 | WI                           |     0.173986  |                  1184 |                  206 | UT                            |   0.0375235  |            0.215669   | True          | 1.0074e-14   | True                    | True          | True            | True       |                0.0966565 |
| default     | state             | screen for disparity       | false_negative_rate |                30 | CO                           |     0.728261  |                   996 |                   67 | WI                            |   0.252525   |            0.346751   | True          | 4.87196e-11  | True                    | True          | True            | True       |                0.0966565 |
| default     | servicer_name     | screen for disparity       | false_positive_rate |                 5 | Atlas Mortgage Services      |     0.0675315 |                 11970 |                  734 | Northgate Financial Servicing |   0.0494543  |            0.732314   | True          | 2.15118e-08  | True                    | True          | True            | True       |                0.0966565 |
| default     | servicer_name     | screen for disparity       | selection_rate      |                 5 | Atlas Mortgage Services      |     0.112114  |                 11970 |                 1342 | Cornerstone Loan Servicing    |   0.0840322  |            0.749527   | True          | 2.1807e-13   | True                    | True          | True            | True       |                0.0966565 |
| default     | servicer_name     | screen for disparity       | false_negative_rate |                 5 | Cornerstone Loan Servicing   |     0.563041  |                 12162 |                  585 | Atlas Mortgage Services       |   0.447775   |            0.795279   | True          | 9.81259e-08  | True                    | True          | True            | True       |                0.0966565 |
| default     | credit_score_band | risk factor (gap expected) | selection_rate      |                 6 | <620                         |     0.612141  |                  2158 |                 1321 | 800+                          |   0.0018797  |            0.0030707  | True          | 0            | True                    | True          | True            | False      |                0.0966565 |
| default     | credit_score_band | risk factor (gap expected) | false_positive_rate |                 6 | <620                         |     0.546638  |                  2158 |                  756 | 800+                          |   0.0018797  |            0.00343866 | True          | 7.87178e-254 | True                    | True          | True            | False      |                0.0966565 |
| default     | ltv_band          | risk factor (gap expected) | false_positive_rate |                 5 | 90-97%                       |     0.243681  |                  2870 |                  511 | <60%                          |   0.0101129  |            0.0415004  | True          | 3.31338e-211 | True                    | True          | True            | False      |                0.0966565 |
| default     | ltv_band          | risk factor (gap expected) | selection_rate      |                 5 | 90-97%                       |     0.326132  |                  2870 |                  936 | <60%                          |   0.0139893  |            0.0428945  | True          | 3.88755e-306 | True                    | True          | True            | False      |                0.0966565 |
| default     | vintage_year      | risk factor (gap expected) | false_positive_rate |                 6 | 2021                         |     0.080732  |                 14160 |                 1050 | 2018                          |   0.00735294 |            0.0910784  | True          | 0.00169859   | True                    | True          | True            | False      |                0.0966565 |
| default     | vintage_year      | risk factor (gap expected) | selection_rate      |                 6 | 2018                         |     0.336207  |                   232 |                   78 | 2023                          |   0.0713999  |            0.212369   | True          | 2.46762e-50  | True                    | True          | True            | False      |                0.0966565 |
| default     | vintage_year      | risk factor (gap expected) | false_negative_rate |                 6 | 2023                         |     0.638362  |                  9972 |                  639 | 2018                          |   0.197917   |            0.310038   | True          | 3.96267e-17  | True                    | True          | True            | False      |                0.0966565 |
| default     | credit_score_band | risk factor (gap expected) | false_negative_rate |                 5 | 700-739                      |     0.740291  |                 18569 |                  305 | <620                          |   0.270968   |            0.366029   | True          | 2.12889e-54  | True                    | True          | True            | False      |                0.0966565 |
| default     | ltv_band          | risk factor (gap expected) | false_negative_rate |                 5 | 75-80%                       |     0.584648  |                 11315 |                  556 | 80-90%                        |   0.420089   |            0.718534   | True          | 5.20505e-17  | True                    | True          | True            | False      |                0.0966565 |

_Showing 30 of 45 rows._


### Rates by group

| model       | segment           | kind                       | group                         |   records |   actual_positives |   actual_negatives |   flagged |   false_positives |   false_negatives |   true_positives |   selection_rate |   precision |      recall |   false_positive_rate |   false_negative_rate |
|:------------|:------------------|:---------------------------|:------------------------------|----------:|-------------------:|-------------------:|----------:|------------------:|------------------:|-----------------:|-----------------:|------------:|------------:|----------------------:|----------------------:|
| delinquency | credit_score_band | risk factor (gap expected) | 620-659                       |      7173 |               1744 |               5429 |       950 |               144 |               938 |              806 |       0.132441   |    0.848421 |   0.462156  |            0.0265242  |              0.537844 |
| delinquency | credit_score_band | risk factor (gap expected) | 660-699                       |     18417 |               2507 |              15910 |      1261 |               375 |              1621 |              886 |       0.0684693  |    0.702617 |   0.35341   |            0.0235701  |              0.64659  |
| delinquency | credit_score_band | risk factor (gap expected) | <620                          |      2249 |                650 |               1599 |       344 |                37 |               343 |              307 |       0.152957   |    0.892442 |   0.472308  |            0.0231395  |              0.527692 |
| delinquency | credit_score_band | risk factor (gap expected) | 700-739                       |     20820 |               1539 |              19281 |       725 |               329 |              1143 |              396 |       0.0348223  |    0.546207 |   0.25731   |            0.0170634  |              0.74269  |
| delinquency | credit_score_band | risk factor (gap expected) | 740-799                       |     14117 |                693 |              13424 |       298 |               151 |               546 |              147 |       0.0211093  |    0.493289 |   0.212121  |            0.0112485  |              0.787879 |
| delinquency | credit_score_band | risk factor (gap expected) | 800+                          |      1777 |                 54 |               1723 |        19 |                14 |                49 |                5 |       0.0106922  |    0.263158 |   0.0925926 |            0.00812536 |              0.907407 |
| delinquency | ltv_band          | risk factor (gap expected) | 80-90%                        |     13895 |               2294 |              11601 |      1197 |               274 |              1371 |              923 |       0.0861461  |    0.771094 |   0.402354  |            0.0236187  |              0.597646 |
| delinquency | ltv_band          | risk factor (gap expected) | 90-97%                        |      3037 |                738 |               2299 |       382 |                54 |               410 |              328 |       0.125782   |    0.858639 |   0.444444  |            0.0234885  |              0.555556 |
| delinquency | ltv_band          | risk factor (gap expected) | 75-80%                        |     12435 |               1426 |              11009 |       717 |               217 |               926 |              500 |       0.0576598  |    0.69735  |   0.350631  |            0.0197111  |              0.649369 |
| delinquency | ltv_band          | risk factor (gap expected) | 60-75%                        |     30370 |               2503 |              27867 |      1187 |               447 |              1763 |              740 |       0.0390846  |    0.62342  |   0.295645  |            0.0160405  |              0.704355 |
| delinquency | ltv_band          | risk factor (gap expected) | <60%                          |      4816 |                226 |               4590 |       114 |                58 |               170 |               56 |       0.0236711  |    0.491228 |   0.247788  |            0.0126362  |              0.752212 |
| delinquency | vintage_year      | risk factor (gap expected) | 2018                          |      2298 |                272 |               2026 |       146 |                45 |               171 |              101 |       0.0635335  |    0.691781 |   0.371324  |            0.0222113  |              0.628676 |
| delinquency | vintage_year      | risk factor (gap expected) | 2019                          |      9531 |                958 |               8573 |       537 |               169 |               590 |              368 |       0.0563425  |    0.685289 |   0.384134  |            0.0197131  |              0.615866 |
| delinquency | vintage_year      | risk factor (gap expected) | 2021                          |     14160 |               1575 |              12585 |       771 |               242 |              1046 |              529 |       0.0544492  |    0.686122 |   0.335873  |            0.0192292  |              0.664127 |
| delinquency | vintage_year      | risk factor (gap expected) | 2020                          |     11469 |               1193 |              10276 |       638 |               195 |               750 |              443 |       0.0556282  |    0.694357 |   0.371333  |            0.0189763  |              0.628667 |
| delinquency | vintage_year      | risk factor (gap expected) | 2022                          |     17090 |               1989 |              15101 |      1016 |               279 |              1252 |              737 |       0.05945    |    0.725394 |   0.370538  |            0.0184756  |              0.629462 |
| delinquency | vintage_year      | risk factor (gap expected) | 2023                          |     10005 |               1200 |               8805 |       489 |               120 |               831 |              369 |       0.0488756  |    0.754601 |   0.3075    |            0.0136286  |              0.6925   |
| delinquency | state             | screen for disparity       | MN                            |      1254 |                183 |               1071 |        87 |                28 |               124 |               59 |       0.069378   |    0.678161 |   0.322404  |            0.0261438  |              0.677596 |
| delinquency | state             | screen for disparity       | AL                            |       952 |                 91 |                861 |        58 |                22 |                55 |               36 |       0.0609244  |    0.62069  |   0.395604  |            0.0255517  |              0.604396 |
| delinquency | state             | screen for disparity       | MA                            |      1581 |                184 |               1397 |       100 |                33 |               117 |               67 |       0.0632511  |    0.67     |   0.36413   |            0.023622   |              0.63587  |
| delinquency | state             | screen for disparity       | WA                            |      1815 |                197 |               1618 |        89 |                37 |               145 |               52 |       0.0490358  |    0.58427  |   0.263959  |            0.0228677  |              0.736041 |
| delinquency | state             | screen for disparity       | CA                            |      8064 |                950 |               7114 |       491 |               158 |               617 |              333 |       0.0608879  |    0.678208 |   0.350526  |            0.0222097  |              0.649474 |
| delinquency | state             | screen for disparity       | UT                            |       569 |                 47 |                522 |        25 |                11 |                33 |               14 |       0.0439367  |    0.56     |   0.297872  |            0.0210728  |              0.702128 |
| delinquency | state             | screen for disparity       | AZ                            |      1652 |                179 |               1473 |        87 |                31 |               123 |               56 |       0.0526634  |    0.643678 |   0.312849  |            0.0210455  |              0.687151 |
| delinquency | state             | screen for disparity       | TX                            |      6519 |                759 |               5760 |       379 |               116 |               496 |              263 |       0.0581378  |    0.693931 |   0.346509  |            0.0201389  |              0.653491 |
| delinquency | state             | screen for disparity       | MO                            |      1433 |                181 |               1252 |        81 |                25 |               125 |               56 |       0.0565248  |    0.691358 |   0.309392  |            0.0199681  |              0.690608 |
| delinquency | state             | screen for disparity       | GA                            |      2369 |                269 |               2100 |       132 |                40 |               177 |               92 |       0.0557197  |    0.69697  |   0.342007  |            0.0190476  |              0.657993 |
| delinquency | state             | screen for disparity       | LA                            |       849 |                108 |                741 |        52 |                14 |                70 |               38 |       0.0612485  |    0.730769 |   0.351852  |            0.0188934  |              0.648148 |
| delinquency | state             | screen for disparity       | IN                            |      1280 |                157 |               1123 |        81 |                21 |                97 |               60 |       0.0632812  |    0.740741 |   0.382166  |            0.0186999  |              0.617834 |
| delinquency | state             | screen for disparity       | NJ                            |      1849 |                217 |               1632 |       117 |                30 |               130 |               87 |       0.0632774  |    0.74359  |   0.400922  |            0.0183824  |              0.599078 |
| delinquency | state             | screen for disparity       | CT                            |       622 |                 66 |                556 |        29 |                10 |                47 |               19 |       0.0466238  |    0.655172 |   0.287879  |            0.0179856  |              0.712121 |
| delinquency | state             | screen for disparity       | NY                            |      3862 |                362 |               3500 |       168 |                62 |               256 |              106 |       0.0435008  |    0.630952 |   0.292818  |            0.0177143  |              0.707182 |
| delinquency | state             | screen for disparity       | KY                            |       910 |                115 |                795 |        61 |                14 |                68 |               47 |       0.067033   |    0.770492 |   0.408696  |            0.0176101  |              0.591304 |
| delinquency | state             | screen for disparity       | OR                            |      1057 |                 90 |                967 |        41 |                17 |                66 |               24 |       0.038789   |    0.585366 |   0.266667  |            0.0175801  |              0.733333 |
| delinquency | state             | screen for disparity       | IL                            |      2629 |                274 |               2355 |       141 |                41 |               174 |              100 |       0.0536326  |    0.70922  |   0.364964  |            0.0174098  |              0.635036 |
| delinquency | state             | screen for disparity       | VA                            |      1903 |                212 |               1691 |       106 |                29 |               135 |               77 |       0.0557015  |    0.726415 |   0.363208  |            0.0171496  |              0.636792 |
| delinquency | state             | screen for disparity       | PA                            |      2687 |                351 |               2336 |       195 |                40 |               196 |              155 |       0.0725716  |    0.794872 |   0.441595  |            0.0171233  |              0.558405 |
| delinquency | state             | screen for disparity       | SC                            |       905 |                 78 |                827 |        38 |                14 |                54 |               24 |       0.041989   |    0.631579 |   0.307692  |            0.0169287  |              0.692308 |
| delinquency | state             | screen for disparity       | FL                            |      6649 |                755 |               5894 |       368 |                98 |               485 |              270 |       0.0553467  |    0.733696 |   0.357616  |            0.0166271  |              0.642384 |
| delinquency | state             | screen for disparity       | MI                            |      2115 |                265 |               1850 |       145 |                30 |               150 |              115 |       0.0685579  |    0.793103 |   0.433962  |            0.0162162  |              0.566038 |
| delinquency | state             | screen for disparity       | TN                            |      1485 |                148 |               1337 |        69 |                21 |               100 |               48 |       0.0464646  |    0.695652 |   0.324324  |            0.0157068  |              0.675676 |
| delinquency | state             | screen for disparity       | NC                            |      2294 |                259 |               2035 |       129 |                30 |               160 |               99 |       0.0562337  |    0.767442 |   0.382239  |            0.014742   |              0.617761 |
| delinquency | state             | screen for disparity       | WI                            |      1335 |                122 |               1213 |        59 |                16 |                79 |               43 |       0.0441948  |    0.728814 |   0.352459  |            0.0131904  |              0.647541 |
| delinquency | state             | screen for disparity       | MD                            |      1275 |                125 |               1150 |        60 |                15 |                80 |               45 |       0.0470588  |    0.75     |   0.36      |            0.0130435  |              0.64     |
| delinquency | state             | screen for disparity       | OK                            |       847 |                 91 |                756 |        46 |                 9 |                54 |               37 |       0.0543093  |    0.804348 |   0.406593  |            0.0119048  |              0.593407 |
| delinquency | state             | screen for disparity       | OH                            |      2744 |                256 |               2488 |       119 |                29 |               166 |               90 |       0.0433673  |    0.756303 |   0.351562  |            0.0116559  |              0.648438 |
| delinquency | state             | screen for disparity       | CO                            |      1048 |                 96 |                952 |        44 |                 9 |                61 |               35 |       0.0419847  |    0.795455 |   0.364583  |            0.00945378 |              0.635417 |
| delinquency | servicer_name     | screen for disparity       | Atlas Mortgage Services       |     13276 |               1555 |              11721 |       790 |               230 |               995 |              560 |       0.0595059  |    0.708861 |   0.360129  |            0.0196229  |              0.639871 |
| delinquency | servicer_name     | screen for disparity       | Beacon Home Loans             |     12641 |               1407 |              11234 |       690 |               220 |               937 |              470 |       0.0545843  |    0.681159 |   0.334044  |            0.0195834  |              0.665956 |
| delinquency | servicer_name     | screen for disparity       | Meridian Residential Capital  |     12597 |               1501 |              11096 |       768 |               208 |               941 |              560 |       0.0609669  |    0.729167 |   0.373085  |            0.0187455  |              0.626915 |
| delinquency | servicer_name     | screen for disparity       | Cornerstone Loan Servicing    |     13417 |               1451 |              11966 |       696 |               211 |               966 |              485 |       0.0518745  |    0.696839 |   0.334252  |            0.0176333  |              0.665748 |
| delinquency | servicer_name     | screen for disparity       | Northgate Financial Servicing |     12622 |               1273 |              11349 |       653 |               181 |               801 |              472 |       0.0517351  |    0.722818 |   0.370778  |            0.0159485  |              0.629222 |
| default     | credit_score_band | risk factor (gap expected) | <620                          |      2158 |                775 |               1383 |      1321 |               756 |               210 |              565 |       0.612141   |    0.427706 |   0.729032  |            0.546638   |              0.270968 |
| default     | credit_score_band | risk factor (gap expected) | 620-659                       |      6847 |               1980 |               4867 |      2814 |              1640 |               806 |             1174 |       0.410983   |    0.4172   |   0.592929  |            0.336963   |              0.407071 |
| default     | credit_score_band | risk factor (gap expected) | 660-699                       |     16786 |               1790 |              14996 |      1193 |               537 |              1134 |              656 |       0.0710711  |    0.549874 |   0.36648   |            0.0358095  |              0.63352  |
| default     | credit_score_band | risk factor (gap expected) | 700-739                       |     18569 |                412 |              18157 |       228 |               121 |               305 |              107 |       0.0122785  |    0.469298 |   0.259709  |            0.0066641  |              0.740291 |
| default     | credit_score_band | risk factor (gap expected) | 740-799                       |     12426 |                106 |              12320 |        84 |                43 |                65 |               41 |       0.00676002 |    0.488095 |   0.386792  |            0.00349026 |              0.613208 |
| default     | credit_score_band | risk factor (gap expected) | 800+                          |      1596 |                  0 |               1596 |         3 |                 3 |                 0 |                0 |       0.0018797  |    0        | nan         |            0.0018797  |            nan        |
| default     | ltv_band          | risk factor (gap expected) | 90-97%                        |      2870 |                773 |               2097 |       936 |               511 |               348 |              425 |       0.326132   |    0.45406  |   0.549806  |            0.243681   |              0.450194 |
| default     | ltv_band          | risk factor (gap expected) | 80-90%                        |     12993 |               2021 |              10972 |      2574 |              1402 |               849 |             1172 |       0.198107   |    0.455322 |   0.579911  |            0.12778    |              0.420089 |

_Showing 60 of 156 rows._


### Monotonicity check

Not a fairness question but a correctness one: if the model flags 740-799 borrowers more often than 620-659 borrowers, something is inverted.

| model       | segment           | checked   | monotone_decreasing   | first_group   |   first_rate | last_group   |   last_rate |
|:------------|:------------------|:----------|:----------------------|:--------------|-------------:|:-------------|------------:|
| delinquency | credit_score_band | True      | True                  | <620          |     0.152957 | 800+         |   0.0106922 |
| default     | credit_score_band | True      | True                  | <620          |     0.612141 | 800+         |   0.0018797 |
| prepayment  | credit_score_band | True      | False                 | <620          |     0.273865 | 800+         |   0.741228  |


## Figures

**Local explanation: delinquency -- confident true positive**

![Local explanation: delinquency -- confident true positive](explainability_report/waterfall_delinquency_confident_true_positive.png)

**Local explanation: delinquency -- confident false positive**

![Local explanation: delinquency -- confident false positive](explainability_report/waterfall_delinquency_confident_false_positive.png)

**Local explanation: delinquency -- missed event (false negative)**

![Local explanation: delinquency -- missed event (false negative)](explainability_report/waterfall_delinquency_missed_event_false_negative.png)

**Local explanation: delinquency -- borderline**

![Local explanation: delinquency -- borderline](explainability_report/waterfall_delinquency_borderline.png)

**What drives the delinquency model**

![What drives the delinquency model](explainability_report/shap_beeswarm_delinquency.png)

**Local explanation: default -- confident true positive**

![Local explanation: default -- confident true positive](explainability_report/waterfall_default_confident_true_positive.png)

**Local explanation: default -- confident false positive**

![Local explanation: default -- confident false positive](explainability_report/waterfall_default_confident_false_positive.png)

**Local explanation: default -- missed event (false negative)**

![Local explanation: default -- missed event (false negative)](explainability_report/waterfall_default_missed_event_false_negative.png)

**Local explanation: default -- borderline**

![Local explanation: default -- borderline](explainability_report/waterfall_default_borderline.png)

**What drives the default model**

![What drives the default model](explainability_report/shap_beeswarm_default.png)

**Local explanation: prepayment -- missed event (false negative)**

![Local explanation: prepayment -- missed event (false negative)](explainability_report/waterfall_prepayment_missed_event_false_negative.png)

**What drives the prepayment model**

![What drives the prepayment model](explainability_report/shap_beeswarm_prepayment.png)

**Global feature importance**

![Global feature importance](explainability_report/shap_global_importance.png)

**Reliability**

![Reliability](explainability_report/reliability.png)

**Error rates by credit score band (default model)**

![Error rates by credit score band (default model)](explainability_report/error_rates_credit_score_band.png)

**Error rates by vintage year (default model)**

![Error rates by vintage year (default model)](explainability_report/error_rates_vintage_year.png)

**Error rates by servicer name (default model)**

![Error rates by servicer name (default model)](explainability_report/error_rates_servicer_name.png)

