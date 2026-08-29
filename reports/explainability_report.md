# Explainability & Responsible AI Report (Task 6)

_Generated 2026-08-29 23:53:46_

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

| model       | feature                  |   mean_abs_shap |   mean_signed_shap |   max_abs_shap |     share |
|:------------|:-------------------------|----------------:|-------------------:|---------------:|----------:|
| delinquency | credit_score             |       0.262414  |       -0.0381042   |       1.00289  | 0.148436  |
| delinquency | months_since_delinquency |       0.199246  |        0.00460414  |       2.25201  | 0.112704  |
| delinquency | credit_score_band        |       0.188334  |       -0.0119804   |       0.4114   | 0.106532  |
| delinquency | ltv                      |       0.123288  |        0.00489735  |       0.690251 | 0.0697383 |
| delinquency | state                    |       0.106132  |        0.00347861  |       0.500949 | 0.0600338 |
| delinquency | days_past_due            |       0.102194  |        0.00276873  |       1.39274  | 0.0578063 |
| delinquency | interest_rate            |       0.0541601 |        0.0107315   |       0.284914 | 0.0306359 |
| delinquency | rate_spread              |       0.0505995 |        0.0258525   |       0.321862 | 0.0286218 |
| delinquency | paydown_6m               |       0.0445347 |       -0.0158675   |       0.32645  | 0.0251912 |
| delinquency | paydown_3m               |       0.0375085 |       -0.024185    |       0.558705 | 0.0212168 |
| delinquency | status_changes_to_date   |       0.0363332 |       -0.0221019   |       0.430344 | 0.020552  |
| delinquency | status_ordinal           |       0.0316422 |       -0.00286717  |       0.378447 | 0.0178985 |
| delinquency | dti                      |       0.0294187 |        0.00434508  |       0.279265 | 0.0166408 |
| delinquency | balance_vs_expected      |       0.0273664 |       -0.0111517   |       0.326554 | 0.0154799 |
| delinquency | loan_age_months          |       0.0268336 |       -0.0114617   |       0.223718 | 0.0151785 |
| default     | credit_score             |       0.972418  |       -0.116705    |       2.95629  | 0.284084  |
| default     | credit_score_band        |       0.358275  |       -0.034082    |       0.757762 | 0.104667  |
| default     | ltv                      |       0.286363  |       -0.0108484   |       1.20918  | 0.0836589 |
| default     | state                    |       0.254019  |        0.02077     |       1.43105  | 0.0742099 |
| default     | days_past_due            |       0.250935  |        0.00203123  |       3.69729  | 0.0733087 |
| default     | ltv_band                 |       0.199809  |        0.0208146   |       0.795301 | 0.0583726 |
| default     | interest_rate            |       0.130122  |       -0.0369003   |       0.736508 | 0.0380141 |
| default     | loan_age_months          |       0.0949037 |        0.0664322   |       0.329413 | 0.0277254 |
| default     | dti                      |       0.0902938 |        0.0191931   |       0.657799 | 0.0263786 |
| default     | rate_spread              |       0.0659378 |        0.0246394   |       0.352302 | 0.0192632 |
| default     | status_ordinal           |       0.064636  |        0.000576814 |       0.878386 | 0.0188829 |
| default     | current_balance          |       0.0498132 |        0.00335243  |       0.446044 | 0.0145526 |
| default     | original_balance         |       0.0438766 |        0.0106702   |       0.723514 | 0.0128182 |
| default     | paydown_6m               |       0.0392627 |       -0.018716    |       0.29021  | 0.0114703 |
| default     | paydown_3m               |       0.0392305 |       -0.0275669   |       0.228961 | 0.0114609 |
| prepayment  | credit_score             |       0.211817  |        0.00419017  |       1.10453  | 0.197623  |
| prepayment  | state                    |       0.177615  |       -0.00334689  |       0.890867 | 0.165713  |
| prepayment  | interest_rate            |       0.104217  |       -0.0499974   |       0.498481 | 0.0972329 |
| prepayment  | ltv                      |       0.0754389 |        0.0101757   |       0.666483 | 0.0703837 |
| prepayment  | dti                      |       0.0566959 |        0.000173256 |       0.612406 | 0.0528967 |
| prepayment  | original_balance         |       0.0536411 |        0.00641684  |       0.425983 | 0.0500465 |
| prepayment  | rate_spread              |       0.0382602 |       -0.00569389  |       0.308566 | 0.0356964 |
| prepayment  | dpd_max_12m              |       0.0376696 |       -0.00143952  |       0.489973 | 0.0351453 |
| prepayment  | current_balance          |       0.0322702 |       -0.00228229  |       0.491707 | 0.0301078 |
| prepayment  | balance_pctile_in_month  |       0.0248507 |        0.00312653  |       0.184336 | 0.0231854 |
| prepayment  | dpd_mean_12m             |       0.0222676 |        0.00173497  |       0.445084 | 0.0207755 |
| prepayment  | loan_purpose             |       0.0218957 |        0.00121147  |       0.175934 | 0.0204284 |
| prepayment  | paydown_6m               |       0.0197798 |       -0.00774239  |       0.225606 | 0.0184544 |
| prepayment  | servicer_name            |       0.0186219 |       -0.000302394 |       0.120957 | 0.017374  |
| prepayment  | property_type            |       0.0176111 |        0.000729934 |       0.324183 | 0.016431  |


## Loans selected for local explanation

Deliberately not a highlight reel: a confident hit, a confident false positive, a missed event and a borderline case. Showing only the confident hit would demonstrate the model on the records where nothing was in doubt.

| model       | case                          |   position | loan_id      | reporting_month     |   predicted_probability |   actual_outcome |
|:------------|:------------------------------|-----------:|:-------------|:--------------------|------------------------:|-----------------:|
| delinquency | confident true positive       |       9235 | GY5ARW8PMV7P | 2023-01-01 00:00:00 |               1         |                1 |
| delinquency | confident false positive      |       6668 | CD0OSLUPND2V | 2023-08-01 00:00:00 |               0.980952  |                0 |
| delinquency | missed event (false negative) |       8762 | G5V6OJNQD5QO | 2023-01-01 00:00:00 |               0         |                1 |
| delinquency | borderline                    |        137 | 08B77I816HSD | 2023-09-01 00:00:00 |               0.483108  |                0 |
| default     | confident true positive       |       1042 | 22U6DIG1LM4Y | 2023-01-01 00:00:00 |               1         |                1 |
| default     | confident false positive      |      14122 | PMI2G717VDTC | 2023-06-01 00:00:00 |               0.972727  |                0 |
| default     | missed event (false negative) |       5415 | 9X5K386P97VI | 2023-06-01 00:00:00 |               0         |                1 |
| default     | borderline                    |        960 | 1W5M9PEAL1H5 | 2023-07-01 00:00:00 |               0.5       |                1 |
| prepayment  | missed event (false negative) |       8774 | G8QEJP544TIQ | 2023-04-01 00:00:00 |               0.0176471 |                1 |


### Local explanation: delinquency -- confident true positive

| feature                  | feature_value      |   shap_value | direction   |
|:-------------------------|:-------------------|-------------:|:------------|
| months_since_delinquency | 0.0                |    1.89204   | raises risk |
| days_past_due            | 94.0               |    1.3023    | raises risk |
| dpd_mean_6m              | 70.66666666666667  |   -0.381422  | lowers risk |
| status_ordinal           | 4.0                |    0.270134  | raises risk |
| is_delinquent            | 1.0                |    0.208607  | raises risk |
| credit_score             | 660.0              |    0.171684  | raises risk |
| dpd_lag_1m               | 98.0               |    0.160029  | raises risk |
| ltv                      | 97.35              |    0.119817  | raises risk |
| dpd_mean_12m             | 35.333333333333336 |    0.104541  | raises risk |
| dpd_mean_3m              | 100.0              |    0.0925613 | raises risk |
| dpd_delta_1m             | -4.0               |    0.0880827 | raises risk |
| credit_score_band        | 660-699            |    0.0866681 | raises risk |


### Local explanation: delinquency -- confident false positive

| feature                  | feature_value      |   shap_value | direction   |
|:-------------------------|:-------------------|-------------:|:------------|
| months_since_delinquency | 0.0                |    1.96289   | raises risk |
| days_past_due            | 76.0               |    1.30681   | raises risk |
| dpd_mean_6m              | 63.166666666666664 |   -0.407833  | lowers risk |
| status_ordinal           | 3.0                |    0.34834   | raises risk |
| credit_score             | 638.0              |    0.258343  | raises risk |
| is_delinquent            | 1.0                |    0.212056  | raises risk |
| state                    | MN                 |   -0.138636  | lowers risk |
| dpd_lag_1m               | 84.0               |    0.1354    | raises risk |
| credit_score_band        | 620-659            |    0.0887822 | raises risk |
| dpd_delta_1m             | -8.0               |    0.0874016 | raises risk |
| status_changes_to_date   | 8.0                |   -0.0770223 | lowers risk |
| dpd_mean_12m             | 34.083333333333336 |    0.0690444 | raises risk |


### Local explanation: delinquency -- missed event (false negative)

| feature                   | feature_value         |   shap_value | direction   |
|:--------------------------|:----------------------|-------------:|:------------|
| dpd_mean_6m               | 58.333333333333336    |   -0.608709  | lowers risk |
| state                     | OR                    |   -0.324735  | lowers risk |
| balance_ratio             | 0.8124085816876122    |   -0.247552  | lowers risk |
| status_changes_to_date    | 9.0                   |   -0.24499   | lowers risk |
| delinquent_months_to_date | 9.0                   |   -0.203788  | lowers risk |
| paydown_6m                | -0.005487109179072913 |   -0.1533    | lowers risk |
| months_since_delinquency  | nan                   |   -0.133564  | lowers risk |
| balance_vs_expected       | 1.0534428037453785    |   -0.126664  | lowers risk |
| modified_ever             | 1.0                   |    0.126601  | raises risk |
| dpd_mean_12m              | 56.916666666666664    |    0.0810525 | raises risk |
| credit_score_band         | 660-699               |   -0.0805379 | lowers risk |
| days_past_due             | 0.0                   |   -0.0761335 | lowers risk |


### Local explanation: delinquency -- borderline

| feature                  | feature_value          |   shap_value | direction   |
|:-------------------------|:-----------------------|-------------:|:------------|
| months_since_delinquency | 0.0                    |    1.60456   | raises risk |
| days_past_due            | 34.0                   |    0.790131  | raises risk |
| status_ordinal           | 2.0                    |    0.212849  | raises risk |
| is_delinquent            | 1.0                    |    0.19173   | raises risk |
| ltv                      | 68.93                  |   -0.134441  | lowers risk |
| credit_score_band        | 660-699                |    0.101154  | raises risk |
| status_changes_to_date   | 5.0                    |   -0.0933791 | lowers risk |
| dti                      | 54.28                  |   -0.0933323 | lowers risk |
| dpd_delta_1m             | 34.0                   |    0.0882424 | raises risk |
| paydown_1m               | -0.0017225891231479329 |   -0.063559  | lowers risk |
| dpd_lag_1m               | 0.0                    |   -0.0621178 | lowers risk |
| current_status           | 30-DPD                 |    0.0549636 | raises risk |


### Local explanation: default -- confident true positive

| feature           | feature_value      |   shap_value | direction   |
|:------------------|:-------------------|-------------:|:------------|
| days_past_due     | 116.0              |    2.41594   | raises risk |
| credit_score      | 635.0              |    1.24561   | raises risk |
| status_ordinal    | 4.0                |    0.506994  | raises risk |
| credit_score_band | 620-659            |    0.349283  | raises risk |
| ltv               | 86.09              |    0.163251  | raises risk |
| dpd_lag_1m        | 115.0              |    0.132599  | raises risk |
| is_delinquent     | 1.0                |    0.113798  | raises risk |
| dpd_delta_1m      | 1.0                |    0.109107  | raises risk |
| balance_ratio     | 0.8351985852713177 |   -0.10405   | lowers risk |
| loan_age_months   | 47.0               |    0.0983803 | raises risk |
| ltv_band          | 80-90%             |    0.0899423 | raises risk |
| state             | WA                 |    0.0870348 | raises risk |


### Local explanation: default -- confident false positive

| feature           | feature_value   |   shap_value | direction   |
|:------------------|:----------------|-------------:|:------------|
| days_past_due     | 108.0           |    2.42086   | raises risk |
| credit_score      | 653.0           |    1.05894   | raises risk |
| status_ordinal    | 4.0             |    0.493286  | raises risk |
| ltv               | 87.5            |    0.37154   | raises risk |
| credit_score_band | 620-659         |    0.36158   | raises risk |
| dpd_lag_1m        | 71.0            |    0.144822  | raises risk |
| is_delinquent     | 1.0             |    0.118716  | raises risk |
| interest_rate     | 6.401           |   -0.118098  | lowers risk |
| state             | OR              |   -0.115884  | lowers risk |
| ltv_band          | 80-90%          |    0.104801  | raises risk |
| dpd_delta_1m      | 37.0            |    0.0780039 | raises risk |
| paydown_1m        | 0.0             |    0.0615417 | raises risk |


### Local explanation: default -- missed event (false negative)

| feature           | feature_value      |   shap_value | direction   |
|:------------------|:-------------------|-------------:|:------------|
| credit_score      | 707.0              |   -0.888921  | lowers risk |
| state             | AL                 |   -0.660116  | lowers risk |
| credit_score_band | 700-739            |   -0.309004  | lowers risk |
| ltv               | 69.63              |   -0.201922  | lowers risk |
| ltv_band          | 60-75%             |   -0.183883  | lowers risk |
| days_past_due     | 0.0                |   -0.14091   | lowers risk |
| rate_spread       | 2.6125             |    0.111395  | raises risk |
| interest_rate     | 7.172              |   -0.0977821 | lowers risk |
| dti               | 32.51              |   -0.0690953 | lowers risk |
| status_ordinal    | 1.0                |   -0.0356052 | lowers risk |
| balance_ratio     | 0.9992077205882353 |    0.0295514 | raises risk |
| loan_age_months   | 1.0                |   -0.0266325 | lowers risk |


### Local explanation: default -- borderline

| feature           | feature_value          |   shap_value | direction   |
|:------------------|:-----------------------|-------------:|:------------|
| credit_score      | 646.0                  |    1.62672   | raises risk |
| days_past_due     | 50.0                   |    1.56995   | raises risk |
| credit_score_band | 620-659                |    0.467025  | raises risk |
| status_ordinal    | 2.0                    |    0.443746  | raises risk |
| ltv               | 67.74                  |   -0.19501   | lowers risk |
| is_delinquent     | 1.0                    |    0.116752  | raises risk |
| dpd_delta_3m      | 50.0                   |    0.0861854 | raises risk |
| occupancy_type    | Investment Property    |   -0.083179  | lowers risk |
| paydown_6m        | -0.00580701619798607   |   -0.0653252 | lowers risk |
| dti               | 42.23                  |    0.061525  | raises risk |
| ltv_band          | 60-75%                 |   -0.0601072 | lowers risk |
| paydown_1m        | -0.0009852800739298395 |    0.057636  | raises risk |


### Local explanation: prepayment -- missed event (false negative)

| feature          | feature_value          |   shap_value | direction   |
|:-----------------|:-----------------------|-------------:|:------------|
| credit_score     | 644.0                  |   -0.924123  | lowers risk |
| dpd_max_12m      | 116.0                  |   -0.314062  | lowers risk |
| interest_rate    | 6.736                  |   -0.17333   | lowers risk |
| dpd_mean_12m     | 9.666666666666666      |   -0.0917235 | lowers risk |
| dpd_max_6m       | 116.0                  |   -0.0902053 | lowers risk |
| days_past_due    | 116.0                  |   -0.055517  | lowers risk |
| dti              | 31.24                  |    0.0523439 | raises risk |
| original_balance | 452000.0               |    0.0317182 | raises risk |
| loan_purpose     | Rate/Term Refinance    |    0.0276354 | raises risk |
| dpd_mean_6m      | 19.333333333333332     |   -0.0238909 | lowers risk |
| paydown_3m       | -0.002815663513674127  |    0.023258  | raises risk |
| paydown_1m       | -0.0009455946870864373 |    0.0167201 | raises risk |


## Confusion summary

At the threshold tuned in Task 2.

| model       | outcome        |   records |     share |
|:------------|:---------------|----------:|----------:|
| delinquency | true positive  |      2581 | 0.0399826 |
| delinquency | false positive |      1187 | 0.018388  |
| delinquency | false negative |      4606 | 0.0713522 |
| delinquency | true negative  |     56179 | 0.870277  |
| default     | true positive  |      2360 | 0.0404234 |
| default     | false positive |      2661 | 0.0455791 |
| default     | false negative |      2703 | 0.0462985 |
| default     | true negative  |     50658 | 0.867699  |
| prepayment  | true positive  |      2992 | 0.0512487 |
| prepayment  | false positive |     28378 | 0.486074  |
| prepayment  | false negative |      2236 | 0.0382995 |
| prepayment  | true negative  |     24776 | 0.424377  |


## Error rates by segment

Groups below the minimum size are dropped: a false positive rate computed on nine loans is noise with a decimal point, and putting it in a governance table invites someone to act on it.

| model       | segment           | group   |   records |   actual_positives |   actual_negatives |   flagged |   false_positives |   false_negatives |   true_positives |   selection_rate |   precision |    recall |   false_positive_rate |   false_negative_rate |
|:------------|:------------------|:--------|----------:|-------------------:|-------------------:|----------:|------------------:|------------------:|-----------------:|-----------------:|------------:|----------:|----------------------:|----------------------:|
| delinquency | credit_score_band | 620-659 |      7173 |               1744 |               5429 |      1035 |               220 |               929 |              815 |        0.144291  |    0.78744  | 0.467317  |            0.0405231  |              0.532683 |
| delinquency | credit_score_band | <620    |      2249 |                650 |               1599 |       387 |                62 |               325 |              325 |        0.172076  |    0.839793 | 0.5       |            0.0387742  |              0.5      |
| delinquency | credit_score_band | 660-699 |     18417 |               2507 |              15910 |      1278 |               390 |              1619 |              888 |        0.0693924 |    0.694836 | 0.354208  |            0.0245129  |              0.645792 |
| delinquency | credit_score_band | 700-739 |     20820 |               1539 |              19281 |       725 |               329 |              1143 |              396 |        0.0348223 |    0.546207 | 0.25731   |            0.0170634  |              0.74269  |
| delinquency | credit_score_band | 740-799 |     14117 |                693 |              13424 |       321 |               169 |               541 |              152 |        0.0227385 |    0.47352  | 0.219336  |            0.0125894  |              0.780664 |
| delinquency | credit_score_band | 800+    |      1777 |                 54 |               1723 |        22 |                17 |                49 |                5 |        0.0123804 |    0.227273 | 0.0925926 |            0.00986651 |              0.907407 |
| delinquency | ltv_band          | 90-97%  |      3037 |                738 |               2299 |       469 |               128 |               397 |              341 |        0.154429  |    0.727079 | 0.46206   |            0.0556764  |              0.53794  |
| delinquency | ltv_band          | 80-90%  |     13895 |               2294 |              11601 |      1245 |               307 |              1356 |              938 |        0.0896006 |    0.753414 | 0.408893  |            0.0264632  |              0.591107 |
| delinquency | ltv_band          | 75-80%  |     12435 |               1426 |              11009 |       724 |               224 |               926 |              500 |        0.0582228 |    0.690608 | 0.350631  |            0.020347   |              0.649369 |
| delinquency | ltv_band          | 60-75%  |     30370 |               2503 |              27867 |      1211 |               467 |              1759 |              744 |        0.0398749 |    0.614368 | 0.297243  |            0.0167582  |              0.702757 |
| delinquency | ltv_band          | <60%    |      4816 |                226 |               4590 |       119 |                61 |               168 |               58 |        0.0247093 |    0.487395 | 0.256637  |            0.0132898  |              0.743363 |
| delinquency | vintage_year      | 2018    |      2298 |                272 |               2026 |       149 |                48 |               171 |              101 |        0.064839  |    0.677852 | 0.371324  |            0.023692   |              0.628676 |
| delinquency | vintage_year      | 2022    |     17090 |               1989 |              15101 |      1106 |               349 |              1232 |              757 |        0.0647162 |    0.684448 | 0.380593  |            0.0231111  |              0.619407 |
| delinquency | vintage_year      | 2021    |     14160 |               1575 |              12585 |       791 |               260 |              1044 |              531 |        0.0558616 |    0.671302 | 0.337143  |            0.0206595  |              0.662857 |
| delinquency | vintage_year      | 2019    |      9531 |                958 |               8573 |       544 |               174 |               588 |              370 |        0.0570769 |    0.680147 | 0.386221  |            0.0202963  |              0.613779 |
| delinquency | vintage_year      | 2020    |     11469 |               1193 |              10276 |       644 |               200 |               749 |              444 |        0.0561514 |    0.689441 | 0.372171  |            0.0194628  |              0.627829 |
| delinquency | vintage_year      | 2023    |     10005 |               1200 |               8805 |       534 |               156 |               822 |              378 |        0.0533733 |    0.707865 | 0.315     |            0.0177172  |              0.685    |
| delinquency | state             | MI      |      2115 |                265 |               1850 |       168 |                52 |               149 |              116 |        0.0794326 |    0.690476 | 0.437736  |            0.0281081  |              0.562264 |
| delinquency | state             | MN      |      1254 |                183 |               1071 |        88 |                29 |               124 |               59 |        0.0701754 |    0.670455 | 0.322404  |            0.0270775  |              0.677596 |
| delinquency | state             | GA      |      2369 |                269 |               2100 |       151 |                56 |               174 |               95 |        0.06374   |    0.629139 | 0.35316   |            0.0266667  |              0.64684  |
| delinquency | state             | MA      |      1581 |                184 |               1397 |       103 |                36 |               117 |               67 |        0.0651486 |    0.650485 | 0.36413   |            0.0257695  |              0.63587  |
| delinquency | state             | AL      |       952 |                 91 |                861 |        58 |                22 |                55 |               36 |        0.0609244 |    0.62069  | 0.395604  |            0.0255517  |              0.604396 |
| delinquency | state             | CA      |      8064 |                950 |               7114 |       516 |               177 |               611 |              339 |        0.0639881 |    0.656977 | 0.356842  |            0.0248805  |              0.643158 |
| delinquency | state             | WA      |      1815 |                197 |               1618 |        91 |                39 |               145 |               52 |        0.0501377 |    0.571429 | 0.263959  |            0.0241038  |              0.736041 |
| delinquency | state             | TX      |      6519 |                759 |               5760 |       400 |               137 |               496 |              263 |        0.0613591 |    0.6575   | 0.346509  |            0.0237847  |              0.653491 |
| delinquency | state             | AZ      |      1652 |                179 |               1473 |        92 |                33 |               120 |               59 |        0.0556901 |    0.641304 | 0.329609  |            0.0224033  |              0.670391 |
| delinquency | state             | LA      |       849 |                108 |                741 |        55 |                16 |                69 |               39 |        0.0647821 |    0.709091 | 0.361111  |            0.0215924  |              0.638889 |
| delinquency | state             | UT      |       569 |                 47 |                522 |        25 |                11 |                33 |               14 |        0.0439367 |    0.56     | 0.297872  |            0.0210728  |              0.702128 |
| delinquency | state             | MO      |      1433 |                181 |               1252 |        82 |                26 |               125 |               56 |        0.0572226 |    0.682927 | 0.309392  |            0.0207668  |              0.690608 |
| delinquency | state             | IN      |      1280 |                157 |               1123 |        83 |                23 |                97 |               60 |        0.0648438 |    0.722892 | 0.382166  |            0.0204809  |              0.617834 |
| delinquency | state             | OH      |      2744 |                256 |               2488 |       138 |                48 |               166 |               90 |        0.0502915 |    0.652174 | 0.351562  |            0.0192926  |              0.648438 |
| delinquency | state             | IL      |      2629 |                274 |               2355 |       146 |                45 |               173 |              101 |        0.0555344 |    0.691781 | 0.368613  |            0.0191083  |              0.631387 |
| delinquency | state             | NY      |      3862 |                362 |               3500 |       173 |                65 |               254 |              108 |        0.0447954 |    0.624277 | 0.298343  |            0.0185714  |              0.701657 |
| delinquency | state             | FL      |      6649 |                755 |               5894 |       384 |               109 |               480 |              275 |        0.057753  |    0.716146 | 0.364238  |            0.0184934  |              0.635762 |
| delinquency | state             | NJ      |      1849 |                217 |               1632 |       118 |                30 |               129 |               88 |        0.0638183 |    0.745763 | 0.40553   |            0.0183824  |              0.59447  |
| delinquency | state             | CT      |       622 |                 66 |                556 |        29 |                10 |                47 |               19 |        0.0466238 |    0.655172 | 0.287879  |            0.0179856  |              0.712121 |
| delinquency | state             | VA      |      1903 |                212 |               1691 |       107 |                30 |               135 |               77 |        0.056227  |    0.719626 | 0.363208  |            0.017741   |              0.636792 |
| delinquency | state             | KY      |       910 |                115 |                795 |        62 |                14 |                67 |               48 |        0.0681319 |    0.774194 | 0.417391  |            0.0176101  |              0.582609 |
| delinquency | state             | OR      |      1057 |                 90 |                967 |        41 |                17 |                66 |               24 |        0.038789  |    0.585366 | 0.266667  |            0.0175801  |              0.733333 |
| delinquency | state             | PA      |      2687 |                351 |               2336 |       196 |                41 |               196 |              155 |        0.0729438 |    0.790816 | 0.441595  |            0.0175514  |              0.558405 |
| delinquency | state             | TN      |      1485 |                148 |               1337 |        75 |                23 |                96 |               52 |        0.0505051 |    0.693333 | 0.351351  |            0.0172027  |              0.648649 |
| delinquency | state             | SC      |       905 |                 78 |                827 |        38 |                14 |                54 |               24 |        0.041989  |    0.631579 | 0.307692  |            0.0169287  |              0.692308 |
| delinquency | state             | NC      |      2294 |                259 |               2035 |       131 |                31 |               159 |              100 |        0.0571055 |    0.763359 | 0.3861    |            0.0152334  |              0.6139   |
| delinquency | state             | MD      |      1275 |                125 |               1150 |        66 |                17 |                76 |               49 |        0.0517647 |    0.742424 | 0.392     |            0.0147826  |              0.608    |
| delinquency | state             | WI      |      1335 |                122 |               1213 |        60 |                17 |                79 |               43 |        0.0449438 |    0.716667 | 0.352459  |            0.0140148  |              0.647541 |

_Showing 45 of 156 rows._


## What a false positive looks like

Mean feature value among false positives against true negatives, standardised by the spread. A large gap says the model is flagging loans that look like *this*, which is the actionable form of an error analysis.

| model       | feature                   |   false_positive_mean |   true_negative_mean |   standardised_gap_fp_vs_tn |   false_negative_mean |   true_positive_mean |   standardised_gap_fn_vs_tp |
|:------------|:--------------------------|----------------------:|---------------------:|----------------------------:|----------------------:|---------------------:|----------------------------:|
| delinquency | is_delinquent             |              0.914912 |           0.00646149 |                    3.75496  |            0.00564481 |             0.989926 |                  -4.06839   |
| delinquency | status_changed_this_month |              0.734625 |           0.0334645  |                    2.63121  |            0.0334347  |             0.79814  |                  -2.86967   |
| delinquency | days_past_due             |             46.1609   |           0.672155   |                    2.57358  |            0.589666   |            68.3712   |                  -3.83483   |
| delinquency | status_ordinal            |              2.09773  |           1.01938    |                    2.2545   |            1.01693    |             2.8024   |                  -3.7329    |
| delinquency | dpd_delta_3m              |             42.2575   |          -1.69318    |                    2.06979  |           -1.96052    |            57.4738   |                  -2.79896   |
| delinquency | dpd_delta_6m              |             43.5      |          -1.51894    |                    2.05912  |           -2.36102    |            66.1168   |                  -3.1321    |
| delinquency | dpd_max_3m                |             49.4288   |           4.27519    |                    1.84881  |            4.02149    |            70.623    |                  -2.72699   |
| delinquency | dpd_delta_1m              |             29.3123   |          -1.16335    |                    1.74696  |           -1.21119    |            27.7053   |                  -1.65758   |
| delinquency | dpd_mean_3m               |             23.7101   |           1.58459    |                    1.71055  |            1.49624    |            44.1504   |                  -3.29766   |
| delinquency | dpd_max_6m                |             50.5442   |           8.48554    |                    1.44968  |            8.84129    |            71.6428   |                  -2.16465   |
| delinquency | dpd_mean_6m               |             14.0433   |           1.87428    |                    1.32855  |            2.00357    |            26.2979   |                  -2.65232   |
| delinquency | dpd_mean_12m              |              9.16259  |           1.89939    |                    1.1501   |            2.19427    |            16.6608   |                  -2.29072   |
| default     | credit_score              |            643.758    |         714.726      |                   -1.48283  |          665.584      |           648.031    |                   0.366752  |
| default     | is_delinquent             |              0.326193 |           0.0195428  |                    1.24417  |            0.0388457  |             0.775424 |                  -2.9885    |
| default     | days_past_due             |             18.9113   |           1.23684    |                    0.975243 |            1.84166    |            59.1703   |                  -3.16329   |
| default     | status_ordinal            |              1.46336  |           1.03231    |                    0.878079 |            1.04624    |             2.59619  |                  -3.15738   |
| default     | status_changed_this_month |              0.280722 |           0.0438628  |                    0.877685 |            0.0558639  |             0.634322 |                  -2.14349   |
| default     | ltv                       |             81.1037   |          73.1835     |                    0.824604 |           79.9922     |            81.6959   |                  -0.177381  |
| default     | dpd_delta_3m              |             15.3653   |          -1.09186    |                    0.762811 |           -1.096      |            47.4902   |                  -2.25203   |
| default     | dpd_delta_6m              |             15.2724   |          -0.925491   |                    0.726165 |           -1.30265    |            55.1282   |                  -2.52985   |
| default     | dpd_mean_3m               |             11.1994   |           1.82785    |                    0.704447 |            2.04581    |            40.5505   |                  -2.89436   |
| default     | dpd_max_3m                |             21.9658   |           4.839      |                    0.689821 |            5.16685    |            62.103    |                  -2.29323   |
| default     | dti                       |             39.8479   |          34.87       |                    0.664743 |           38.5561     |            38.8987   |                  -0.0457571 |
| default     | dpd_delta_1m              |              9.58571  |          -0.714936   |                    0.584118 |           -0.154748   |            20.0941   |                  -1.14825   |
| prepayment  | credit_score              |            716.685    |         693.218      |                    0.490316 |          705.725      |           720.86     |                  -0.316216  |
| prepayment  | dpd_mean_12m              |              1.54898  |           4.0762     |                   -0.391148 |            3.21192    |             1.47794  |                   0.268376  |
| prepayment  | dpd_max_12m               |             12.3643   |          25.0388     |                   -0.369406 |           21.0121     |            12.7787   |                   0.239966  |
| prepayment  | dpd_mean_6m               |              1.6436   |           5.06542    |                   -0.364056 |            3.49334    |             1.57753  |                   0.203827  |
| prepayment  | dpd_mean_3m               |              1.80703  |           6.44752    |                   -0.34882  |            3.33028    |             1.72315  |                   0.120806  |
| prepayment  | dpd_max_6m                |              7.31102  |          17.4807     |                   -0.347972 |           14.3135     |             7.68182  |                   0.226913  |

_Showing 30 of 36 rows._


## Calibration

Expected calibration error is the population-weighted mean gap between predicted and observed, so a wild miss in a bin holding four records does not outweigh a small bias across the bulk of the book.

| model       |   expected_calibration_error |   mean_predicted |   observed_rate |         bias |
|:------------|-----------------------------:|-----------------:|----------------:|-------------:|
| delinquency |                   0.00400925 |        0.111667  |       0.111335  |  0.000332136 |
| default     |                   0.0070369  |        0.0808743 |       0.0867219 | -0.00584759  |
| prepayment  |                   0.00848467 |        0.0838606 |       0.0895481 | -0.00568759  |


### Reliability by probability bin

| bin           |   records |   mean_predicted |   observed_rate |         gap | model       |
|:--------------|----------:|-----------------:|----------------:|------------:|:------------|
| (-0.001, 0.1] |     46209 |        0.0578409 |       0.0597935 | -0.00195265 | delinquency |
| (0.1, 0.2]    |     14389 |        0.133752  |       0.125999  |  0.00775268 | delinquency |
| (0.2, 0.3]    |       426 |        0.225844  |       0.20892   |  0.0169233  | delinquency |
| (0.3, 0.4]    |         1 |        0.332733  |       0         |  0.332733   | delinquency |
| (0.4, 0.5]    |       922 |        0.447871  |       0.43167   |  0.0162008  | delinquency |
| (0.5, 0.6]    |        51 |        0.591638  |       0.607843  | -0.0162052  | delinquency |
| (0.6, 0.7]    |       638 |        0.636508  |       0.626959  |  0.00954884 | delinquency |
| (0.7, 0.8]    |       554 |        0.751496  |       0.772563  | -0.0210672  | delinquency |
| (0.8, 0.9]    |       489 |        0.852833  |       0.869121  | -0.0162881  | delinquency |
| (0.9, 1.0]    |       874 |        0.951955  |       0.961098  | -0.00914363 | delinquency |
| (-0.001, 0.1] |     46593 |        0.0281993 |       0.0318717 | -0.00367246 | default     |
| (0.1, 0.2]    |      5902 |        0.145177  |       0.172484  | -0.0273071  | default     |
| (0.2, 0.3]    |      3870 |        0.267528  |       0.261757  |  0.00577081 | default     |
| (0.3, 0.4]    |       232 |        0.357945  |       0.456897  | -0.0989519  | default     |
| (0.4, 0.5]    |        12 |        0.5       |       0.666667  | -0.166667   | default     |
| (0.5, 0.6]    |       285 |        0.574968  |       0.550877  |  0.0240913  | default     |
| (0.6, 0.7]    |        81 |        0.680134  |       0.691358  | -0.0112241  | default     |
| (0.7, 0.8]    |       543 |        0.731748  |       0.745856  | -0.0141087  | default     |
| (0.8, 0.9]    |       280 |        0.866763  |       0.903571  | -0.0368089  | default     |
| (0.9, 1.0]    |       584 |        0.971777  |       0.962329  |  0.00944867 | default     |
| (-0.001, 0.1] |     55545 |        0.082209  |       0.089657  | -0.00744806 | prepayment  |
| (0.1, 0.2]    |      2837 |        0.116197  |       0.0874163 |  0.0287803  | prepayment  |


### Confidence profile

A model that never leaves a narrow band is technically calibrated and operationally useless: nothing is ever decided.

| model       | band                                 |     share |
|:------------|:-------------------------------------|----------:|
| delinquency | confident, flagged                   | 0.048193  |
| delinquency | uncertain (within 0.05 of threshold) | 0.012176  |
| delinquency | confident, cleared                   | 0.0841634 |
| delinquency | max predicted probability            | 1         |
| default     | confident, flagged                   | 0.0305745 |
| default     | uncertain (within 0.05 of threshold) | 0.0663047 |
| default     | confident, cleared                   | 0.685297  |
| default     | max predicted probability            | 1         |
| prepayment  | confident, flagged                   | 0         |
| prepayment  | uncertain (within 0.05 of threshold) | 0.984036  |
| prepayment  | confident, cleared                   | 0         |
| prepayment  | max predicted probability            | 0.190476  |


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
| delinquency | state             | screen for disparity       | false_positive_rate |                30 | MI                           |     0.0281081 |                  2115 |                   52 | CO                            |   0.00945378 |            0.336336   | True          | 0.00135165   | True                    | True          | True            | True       |                0.0583706 |
| delinquency | state             | screen for disparity       | selection_rate      |                30 | MI                           |     0.0794326 |                  2115 |                  168 | OR                            |   0.038789   |            0.488326   | True          | 1.36616e-05  | True                    | True          | True            | True       |                0.0583706 |
| delinquency | servicer_name     | screen for disparity       | false_positive_rate |                 5 | Beacon Home Loans            |     0.0242122 |                 12641 |                  272 | Northgate Financial Servicing |   0.0169178  |            0.69873    | True          | 0.000111753  | True                    | True          | True            | True       |                0.0583706 |
| delinquency | state             | screen for disparity       | false_negative_rate |                30 | WA                           |     0.736041  |                  1815 |                  145 | PA                            |   0.558405   |            0.75866    | True          | 3.85975e-05  | True                    | True          | True            | True       |                0.0583706 |
| delinquency | credit_score_band | risk factor (gap expected) | selection_rate      |                 6 | <620                         |     0.172076  |                  2249 |                  387 | 800+                          |   0.0123804  |            0.0719472  | True          | 2.80316e-62  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | ltv_band          | risk factor (gap expected) | selection_rate      |                 5 | 90-97%                       |     0.154429  |                  3037 |                  469 | <60%                          |   0.0247093  |            0.160005   | True          | 2.1206e-100  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | ltv_band          | risk factor (gap expected) | false_positive_rate |                 5 | 90-97%                       |     0.0556764 |                  3037 |                  128 | <60%                          |   0.0132898  |            0.238697   | True          | 3.12218e-24  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | credit_score_band | risk factor (gap expected) | false_positive_rate |                 6 | 620-659                      |     0.0405231 |                  7173 |                  220 | 800+                          |   0.00986651 |            0.243479   | True          | 5.86621e-10  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | credit_score_band | risk factor (gap expected) | false_negative_rate |                 6 | 800+                         |     0.907407  |                  1777 |                   49 | <620                          |   0.5        |            0.55102    | True          | 8.18076e-09  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | ltv_band          | risk factor (gap expected) | false_negative_rate |                 5 | <60%                         |     0.743363  |                  4816 |                  168 | 90-97%                        |   0.53794    |            0.723658   | True          | 4.11068e-08  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | servicer_name     | screen for disparity       | selection_rate      |                 5 | Meridian Residential Capital |     0.063666  |                 12597 |                  802 | Northgate Financial Servicing |   0.0529235  |            0.831268   | False         | 0.000271897  | True                    | True          | True            | False      |                0.0583706 |
| delinquency | vintage_year      | risk factor (gap expected) | false_negative_rate |                 6 | 2023                         |     0.685     |                 10005 |                  822 | 2019                          |   0.613779   |            0.896027   | False         | 0.00055191   | True                    | True          | True            | False      |                0.0583706 |
| delinquency | vintage_year      | risk factor (gap expected) | false_positive_rate |                 6 | 2018                         |     0.023692  |                  2298 |                   48 | 2023                          |   0.0177172  |            0.747814   | True          | 0.0744727    | True                    | False         | True            | False      |                0.0583706 |
| delinquency | vintage_year      | risk factor (gap expected) | selection_rate      |                 6 | 2018                         |     0.064839  |                  2298 |                  149 | 2023                          |   0.0533733  |            0.823167   | False         | 0.0304192    | True                    | False         | True            | False      |                0.0583706 |
| delinquency | servicer_name     | screen for disparity       | false_negative_rate |                 5 | Beacon Home Loans            |     0.66027   |                 12641 |                  929 | Meridian Residential Capital  |   0.623584   |            0.944438   | False         | 0.03927      | True                    | False         | True            | False      |                0.0583706 |
| default     | state             | screen for disparity       | false_positive_rate |                30 | WI                           |     0.0995392 |                  1184 |                  108 | OK                            |   0.00997151 |            0.100177   | True          | 4.85317e-14  | True                    | True          | True            | True       |                0.0860025 |
| default     | state             | screen for disparity       | selection_rate      |                30 | WI                           |     0.149493  |                  1184 |                  177 | SC                            |   0.0379267  |            0.253702   | True          | 2.16512e-15  | True                    | True          | True            | True       |                0.0860025 |
| default     | state             | screen for disparity       | false_negative_rate |                30 | CT                           |     0.75      |                   545 |                   33 | MA                            |   0.25       |            0.333333   | True          | 5.45012e-09  | True                    | True          | True            | True       |                0.0860025 |
| default     | servicer_name     | screen for disparity       | false_positive_rate |                 5 | Atlas Mortgage Services      |     0.0588831 |                 11970 |                  640 | Northgate Financial Servicing |   0.0429827  |            0.729967   | True          | 1.46641e-07  | True                    | True          | True            | True       |                0.0860025 |
| default     | servicer_name     | screen for disparity       | selection_rate      |                 5 | Atlas Mortgage Services      |     0.0999165 |                 11970 |                 1196 | Cornerstone Loan Servicing    |   0.0744943  |            0.745566   | True          | 2.51972e-12  | True                    | True          | True            | True       |                0.0860025 |
| default     | credit_score_band | risk factor (gap expected) | selection_rate      |                 6 | <620                         |     0.500463  |                  2158 |                 1080 | 800+                          |   0.0018797  |            0.00375592 | True          | 1.27878e-243 | True                    | True          | True            | False      |                0.0860025 |
| default     | credit_score_band | risk factor (gap expected) | false_positive_rate |                 6 | <620                         |     0.433116  |                  2158 |                  599 | 800+                          |   0.0018797  |            0.00433994 | True          | 7.49911e-188 | True                    | True          | True            | False      |                0.0860025 |
| default     | ltv_band          | risk factor (gap expected) | selection_rate      |                 5 | 90-97%                       |     0.279791  |                  2870 |                  803 | <60%                          |   0.013523   |            0.0483324  | True          | 1.50287e-252 | True                    | True          | True            | False      |                0.0860025 |
| default     | ltv_band          | risk factor (gap expected) | false_positive_rate |                 5 | 90-97%                       |     0.194087  |                  2870 |                  407 | <60%                          |   0.00964252 |            0.0496815  | True          | 1.90764e-160 | True                    | True          | True            | False      |                0.0860025 |
| default     | vintage_year      | risk factor (gap expected) | false_positive_rate |                 6 | 2021                         |     0.0751192 |                 14160 |                  977 | 2018                          |   0.00735294 |            0.0978837  | True          | 0.00273943   | True                    | True          | True            | False      |                0.0860025 |
| default     | vintage_year      | risk factor (gap expected) | selection_rate      |                 6 | 2018                         |     0.331897  |                   232 |                   77 | 2023                          |   0.0516446  |            0.155605   | True          | 7.74758e-73  | True                    | True          | True            | False      |                0.0860025 |
| default     | vintage_year      | risk factor (gap expected) | false_negative_rate |                 6 | 2023                         |     0.699301  |                  9972 |                  700 | 2018                          |   0.208333   |            0.297917   | True          | 3.8336e-22   | True                    | True          | True            | False      |                0.0860025 |
| default     | credit_score_band | risk factor (gap expected) | false_negative_rate |                 5 | 700-739                      |     0.740291  |                 18569 |                  305 | <620                          |   0.379355   |            0.51244    | True          | 2.42315e-32  | True                    | True          | True            | False      |                0.0860025 |
| default     | ltv_band          | risk factor (gap expected) | false_negative_rate |                 5 | 75-80%                       |     0.599369  |                 11315 |                  570 | 80-90%                        |   0.483919   |            0.80738    | False         | 4.17889e-09  | True                    | True          | True            | False      |                0.0860025 |
| default     | servicer_name     | screen for disparity       | false_negative_rate |                 5 | Cornerstone Loan Servicing   |     0.596728  |                 12162 |                  620 | Atlas Mortgage Services       |   0.495005   |            0.829532   | False         | 2.3308e-06   | True                    | True          | True            | False      |                0.0860025 |

_Showing 30 of 45 rows._


### Rates by group

| model       | segment           | kind                       | group                         |   records |   actual_positives |   actual_negatives |   flagged |   false_positives |   false_negatives |   true_positives |   selection_rate |   precision |      recall |   false_positive_rate |   false_negative_rate |
|:------------|:------------------|:---------------------------|:------------------------------|----------:|-------------------:|-------------------:|----------:|------------------:|------------------:|-----------------:|-----------------:|------------:|------------:|----------------------:|----------------------:|
| delinquency | credit_score_band | risk factor (gap expected) | 620-659                       |      7173 |               1744 |               5429 |      1035 |               220 |               929 |              815 |       0.144291   |    0.78744  |   0.467317  |            0.0405231  |              0.532683 |
| delinquency | credit_score_band | risk factor (gap expected) | <620                          |      2249 |                650 |               1599 |       387 |                62 |               325 |              325 |       0.172076   |    0.839793 |   0.5       |            0.0387742  |              0.5      |
| delinquency | credit_score_band | risk factor (gap expected) | 660-699                       |     18417 |               2507 |              15910 |      1278 |               390 |              1619 |              888 |       0.0693924  |    0.694836 |   0.354208  |            0.0245129  |              0.645792 |
| delinquency | credit_score_band | risk factor (gap expected) | 700-739                       |     20820 |               1539 |              19281 |       725 |               329 |              1143 |              396 |       0.0348223  |    0.546207 |   0.25731   |            0.0170634  |              0.74269  |
| delinquency | credit_score_band | risk factor (gap expected) | 740-799                       |     14117 |                693 |              13424 |       321 |               169 |               541 |              152 |       0.0227385  |    0.47352  |   0.219336  |            0.0125894  |              0.780664 |
| delinquency | credit_score_band | risk factor (gap expected) | 800+                          |      1777 |                 54 |               1723 |        22 |                17 |                49 |                5 |       0.0123804  |    0.227273 |   0.0925926 |            0.00986651 |              0.907407 |
| delinquency | ltv_band          | risk factor (gap expected) | 90-97%                        |      3037 |                738 |               2299 |       469 |               128 |               397 |              341 |       0.154429   |    0.727079 |   0.46206   |            0.0556764  |              0.53794  |
| delinquency | ltv_band          | risk factor (gap expected) | 80-90%                        |     13895 |               2294 |              11601 |      1245 |               307 |              1356 |              938 |       0.0896006  |    0.753414 |   0.408893  |            0.0264632  |              0.591107 |
| delinquency | ltv_band          | risk factor (gap expected) | 75-80%                        |     12435 |               1426 |              11009 |       724 |               224 |               926 |              500 |       0.0582228  |    0.690608 |   0.350631  |            0.020347   |              0.649369 |
| delinquency | ltv_band          | risk factor (gap expected) | 60-75%                        |     30370 |               2503 |              27867 |      1211 |               467 |              1759 |              744 |       0.0398749  |    0.614368 |   0.297243  |            0.0167582  |              0.702757 |
| delinquency | ltv_band          | risk factor (gap expected) | <60%                          |      4816 |                226 |               4590 |       119 |                61 |               168 |               58 |       0.0247093  |    0.487395 |   0.256637  |            0.0132898  |              0.743363 |
| delinquency | vintage_year      | risk factor (gap expected) | 2018                          |      2298 |                272 |               2026 |       149 |                48 |               171 |              101 |       0.064839   |    0.677852 |   0.371324  |            0.023692   |              0.628676 |
| delinquency | vintage_year      | risk factor (gap expected) | 2022                          |     17090 |               1989 |              15101 |      1106 |               349 |              1232 |              757 |       0.0647162  |    0.684448 |   0.380593  |            0.0231111  |              0.619407 |
| delinquency | vintage_year      | risk factor (gap expected) | 2021                          |     14160 |               1575 |              12585 |       791 |               260 |              1044 |              531 |       0.0558616  |    0.671302 |   0.337143  |            0.0206595  |              0.662857 |
| delinquency | vintage_year      | risk factor (gap expected) | 2019                          |      9531 |                958 |               8573 |       544 |               174 |               588 |              370 |       0.0570769  |    0.680147 |   0.386221  |            0.0202963  |              0.613779 |
| delinquency | vintage_year      | risk factor (gap expected) | 2020                          |     11469 |               1193 |              10276 |       644 |               200 |               749 |              444 |       0.0561514  |    0.689441 |   0.372171  |            0.0194628  |              0.627829 |
| delinquency | vintage_year      | risk factor (gap expected) | 2023                          |     10005 |               1200 |               8805 |       534 |               156 |               822 |              378 |       0.0533733  |    0.707865 |   0.315     |            0.0177172  |              0.685    |
| delinquency | state             | screen for disparity       | MI                            |      2115 |                265 |               1850 |       168 |                52 |               149 |              116 |       0.0794326  |    0.690476 |   0.437736  |            0.0281081  |              0.562264 |
| delinquency | state             | screen for disparity       | MN                            |      1254 |                183 |               1071 |        88 |                29 |               124 |               59 |       0.0701754  |    0.670455 |   0.322404  |            0.0270775  |              0.677596 |
| delinquency | state             | screen for disparity       | GA                            |      2369 |                269 |               2100 |       151 |                56 |               174 |               95 |       0.06374    |    0.629139 |   0.35316   |            0.0266667  |              0.64684  |
| delinquency | state             | screen for disparity       | MA                            |      1581 |                184 |               1397 |       103 |                36 |               117 |               67 |       0.0651486  |    0.650485 |   0.36413   |            0.0257695  |              0.63587  |
| delinquency | state             | screen for disparity       | AL                            |       952 |                 91 |                861 |        58 |                22 |                55 |               36 |       0.0609244  |    0.62069  |   0.395604  |            0.0255517  |              0.604396 |
| delinquency | state             | screen for disparity       | CA                            |      8064 |                950 |               7114 |       516 |               177 |               611 |              339 |       0.0639881  |    0.656977 |   0.356842  |            0.0248805  |              0.643158 |
| delinquency | state             | screen for disparity       | WA                            |      1815 |                197 |               1618 |        91 |                39 |               145 |               52 |       0.0501377  |    0.571429 |   0.263959  |            0.0241038  |              0.736041 |
| delinquency | state             | screen for disparity       | TX                            |      6519 |                759 |               5760 |       400 |               137 |               496 |              263 |       0.0613591  |    0.6575   |   0.346509  |            0.0237847  |              0.653491 |
| delinquency | state             | screen for disparity       | AZ                            |      1652 |                179 |               1473 |        92 |                33 |               120 |               59 |       0.0556901  |    0.641304 |   0.329609  |            0.0224033  |              0.670391 |
| delinquency | state             | screen for disparity       | LA                            |       849 |                108 |                741 |        55 |                16 |                69 |               39 |       0.0647821  |    0.709091 |   0.361111  |            0.0215924  |              0.638889 |
| delinquency | state             | screen for disparity       | UT                            |       569 |                 47 |                522 |        25 |                11 |                33 |               14 |       0.0439367  |    0.56     |   0.297872  |            0.0210728  |              0.702128 |
| delinquency | state             | screen for disparity       | MO                            |      1433 |                181 |               1252 |        82 |                26 |               125 |               56 |       0.0572226  |    0.682927 |   0.309392  |            0.0207668  |              0.690608 |
| delinquency | state             | screen for disparity       | IN                            |      1280 |                157 |               1123 |        83 |                23 |                97 |               60 |       0.0648438  |    0.722892 |   0.382166  |            0.0204809  |              0.617834 |
| delinquency | state             | screen for disparity       | OH                            |      2744 |                256 |               2488 |       138 |                48 |               166 |               90 |       0.0502915  |    0.652174 |   0.351562  |            0.0192926  |              0.648438 |
| delinquency | state             | screen for disparity       | IL                            |      2629 |                274 |               2355 |       146 |                45 |               173 |              101 |       0.0555344  |    0.691781 |   0.368613  |            0.0191083  |              0.631387 |
| delinquency | state             | screen for disparity       | NY                            |      3862 |                362 |               3500 |       173 |                65 |               254 |              108 |       0.0447954  |    0.624277 |   0.298343  |            0.0185714  |              0.701657 |
| delinquency | state             | screen for disparity       | FL                            |      6649 |                755 |               5894 |       384 |               109 |               480 |              275 |       0.057753   |    0.716146 |   0.364238  |            0.0184934  |              0.635762 |
| delinquency | state             | screen for disparity       | NJ                            |      1849 |                217 |               1632 |       118 |                30 |               129 |               88 |       0.0638183  |    0.745763 |   0.40553   |            0.0183824  |              0.59447  |
| delinquency | state             | screen for disparity       | CT                            |       622 |                 66 |                556 |        29 |                10 |                47 |               19 |       0.0466238  |    0.655172 |   0.287879  |            0.0179856  |              0.712121 |
| delinquency | state             | screen for disparity       | VA                            |      1903 |                212 |               1691 |       107 |                30 |               135 |               77 |       0.056227   |    0.719626 |   0.363208  |            0.017741   |              0.636792 |
| delinquency | state             | screen for disparity       | KY                            |       910 |                115 |                795 |        62 |                14 |                67 |               48 |       0.0681319  |    0.774194 |   0.417391  |            0.0176101  |              0.582609 |
| delinquency | state             | screen for disparity       | OR                            |      1057 |                 90 |                967 |        41 |                17 |                66 |               24 |       0.038789   |    0.585366 |   0.266667  |            0.0175801  |              0.733333 |
| delinquency | state             | screen for disparity       | PA                            |      2687 |                351 |               2336 |       196 |                41 |               196 |              155 |       0.0729438  |    0.790816 |   0.441595  |            0.0175514  |              0.558405 |
| delinquency | state             | screen for disparity       | TN                            |      1485 |                148 |               1337 |        75 |                23 |                96 |               52 |       0.0505051  |    0.693333 |   0.351351  |            0.0172027  |              0.648649 |
| delinquency | state             | screen for disparity       | SC                            |       905 |                 78 |                827 |        38 |                14 |                54 |               24 |       0.041989   |    0.631579 |   0.307692  |            0.0169287  |              0.692308 |
| delinquency | state             | screen for disparity       | NC                            |      2294 |                259 |               2035 |       131 |                31 |               159 |              100 |       0.0571055  |    0.763359 |   0.3861    |            0.0152334  |              0.6139   |
| delinquency | state             | screen for disparity       | MD                            |      1275 |                125 |               1150 |        66 |                17 |                76 |               49 |       0.0517647  |    0.742424 |   0.392     |            0.0147826  |              0.608    |
| delinquency | state             | screen for disparity       | WI                            |      1335 |                122 |               1213 |        60 |                17 |                79 |               43 |       0.0449438  |    0.716667 |   0.352459  |            0.0140148  |              0.647541 |
| delinquency | state             | screen for disparity       | OK                            |       847 |                 91 |                756 |        47 |                10 |                54 |               37 |       0.05549    |    0.787234 |   0.406593  |            0.0132275  |              0.593407 |
| delinquency | state             | screen for disparity       | CO                            |      1048 |                 96 |                952 |        45 |                 9 |                60 |               36 |       0.0429389  |    0.8      |   0.375     |            0.00945378 |              0.625    |
| delinquency | servicer_name     | screen for disparity       | Beacon Home Loans             |     12641 |               1407 |              11234 |       750 |               272 |               929 |              478 |       0.0593307  |    0.637333 |   0.33973   |            0.0242122  |              0.66027  |
| delinquency | servicer_name     | screen for disparity       | Meridian Residential Capital  |     12597 |               1501 |              11096 |       802 |               237 |               936 |              565 |       0.063666   |    0.704489 |   0.376416  |            0.021359   |              0.623584 |
| delinquency | servicer_name     | screen for disparity       | Atlas Mortgage Services       |     13276 |               1555 |              11721 |       816 |               248 |               987 |              568 |       0.0614643  |    0.696078 |   0.365273  |            0.0211586  |              0.634727 |
| delinquency | servicer_name     | screen for disparity       | Cornerstone Loan Servicing    |     13417 |               1451 |              11966 |       732 |               238 |               957 |              494 |       0.0545577  |    0.674863 |   0.340455  |            0.0198897  |              0.659545 |
| delinquency | servicer_name     | screen for disparity       | Northgate Financial Servicing |     12622 |               1273 |              11349 |       668 |               192 |               797 |              476 |       0.0529235  |    0.712575 |   0.37392   |            0.0169178  |              0.62608  |
| default     | credit_score_band | risk factor (gap expected) | <620                          |      2158 |                775 |               1383 |      1080 |               599 |               294 |              481 |       0.500463   |    0.44537  |   0.620645  |            0.433116   |              0.379355 |
| default     | credit_score_band | risk factor (gap expected) | 620-659                       |      6847 |               1980 |               4867 |      2480 |              1399 |               899 |             1081 |       0.362202   |    0.435887 |   0.54596   |            0.287446   |              0.45404  |
| default     | credit_score_band | risk factor (gap expected) | 660-699                       |     16786 |               1790 |              14996 |      1155 |               506 |              1141 |              649 |       0.0688073  |    0.561905 |   0.36257   |            0.0337423  |              0.63743  |
| default     | credit_score_band | risk factor (gap expected) | 700-739                       |     18569 |                412 |              18157 |       220 |               113 |               305 |              107 |       0.0118477  |    0.486364 |   0.259709  |            0.0062235  |              0.740291 |
| default     | credit_score_band | risk factor (gap expected) | 740-799                       |     12426 |                106 |              12320 |        83 |                41 |                64 |               42 |       0.00667954 |    0.506024 |   0.396226  |            0.00332792 |              0.603774 |
| default     | credit_score_band | risk factor (gap expected) | 800+                          |      1596 |                  0 |               1596 |         3 |                 3 |                 0 |                0 |       0.0018797  |    0        | nan         |            0.0018797  |            nan        |
| default     | ltv_band          | risk factor (gap expected) | 90-97%                        |      2870 |                773 |               2097 |       803 |               407 |               377 |              396 |       0.279791   |    0.493151 |   0.51229   |            0.194087   |              0.48771  |
| default     | ltv_band          | risk factor (gap expected) | 80-90%                        |     12993 |               2021 |              10972 |      2195 |              1152 |               978 |             1043 |       0.168937   |    0.475171 |   0.516081  |            0.104995   |              0.483919 |

_Showing 60 of 156 rows._


### Monotonicity check

Not a fairness question but a correctness one: if the model flags 740-799 borrowers more often than 620-659 borrowers, something is inverted.

| model       | segment           | checked   | monotone_decreasing   | first_group   |   first_rate | last_group   |   last_rate |
|:------------|:------------------|:----------|:----------------------|:--------------|-------------:|:-------------|------------:|
| delinquency | credit_score_band | True      | True                  | <620          |     0.172076 | 800+         |   0.0123804 |
| default     | credit_score_band | True      | True                  | <620          |     0.500463 | 800+         |   0.0018797 |
| prepayment  | credit_score_band | True      | False                 | <620          |     0.149212 | 800+         |   0.515038  |


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

