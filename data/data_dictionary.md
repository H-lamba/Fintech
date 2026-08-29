# Data Dictionary

Loan Performance Intelligence Engine -- synthetic benchmark suite.

## Panel fields

| Field | Definition |
| :--- | :--- |
| loan_id | Unique 12-character alphanumeric loan identifier. |
| month_index | Zero-based months-on-book counter within the loan's observed history. |
| reporting_month | Calendar month the performance record describes. |
| origination_month | Month the loan was originated. |
| loan_age_months | Months elapsed between origination and the reporting month. |
| remaining_term_months | Contractual months remaining at the reporting month. |
| original_balance | Unpaid principal balance at origination, USD. |
| current_balance | Unpaid principal balance at the reporting month, USD. Zero once terminal. |
| interest_rate | Annual fixed note rate, percent. Vintage base rate plus a credit-risk premium. |
| credit_score | Borrower credit score at origination (500-850). |
| credit_score_band | Credit score bucket: <620, 620-659, 660-699, 700-739, 740-799, 800+. |
| ltv | Loan-to-value ratio at origination, percent. |
| ltv_band | LTV bucket: <60%, 60-75%, 75-80%, 80-90%, 90-97%. |
| dti | Debt-to-income ratio at origination, percent. |
| dti_band | DTI bucket: <30%, 30-36%, 36-43%, >43%. |
| state | US state of the mortgaged property. |
| loan_purpose | Home Purchase, Rate/Term Refinance, or Cash-Out Refinance. |
| property_type | Single-Family Detached, Condominium, or PUD. |
| occupancy_type | Primary Residence, Investment Property, or Second Home. |
| servicer_name | Institution servicing the loan. |
| original_term_months | Contractual term at origination (180, 240, or 360 months). |
| vintage_year | Calendar year of origination. |
| current_status | Performance state: Current, 30-DPD, 60-DPD, 90-DPD, Default, Prepaid. |
| days_past_due | Days delinquent at the reporting month, consistent with current_status. |
| modification_flag | True if the loan has received a loss-mitigation modification. |
| prepayment_flag | 1 if the loan prepaid in full in this period. |
| default_flag | 1 if the loan is in default in this period. |
| loss_severity_band | Realised loss severity (Low/Medium/High). Populated only on Default rows; null by design elsewhere. |
| last_updated_at | Timestamp the servicing record was last refreshed. |
| source_system | System of record that produced the row. |
| document_status | Completeness of the loan's document file: Complete, Pending, or Missing. |
| next_3m_delinquency_flag | TARGET. 1 if the loan reaches 30+ DPD in any of months t+1 to t+3. Null where the window is censored. |
| next_6m_delinquency_flag | TARGET. 1 if the loan reaches 60+ DPD in any of months t+1 to t+6. Null where the window is censored. |
| next_12m_default_flag | TARGET. 1 if the loan defaults within months t+1 to t+12. Null where the window is censored. |
| next_12m_prepayment_flag | TARGET. 1 if the loan prepays within months t+1 to t+12. Null where the window is censored. |
| next_state | TARGET. Exact performance state at month t+1. Null where t+1 is unobserved. |
| exception_required | TARGET. True if the record carries an injected data-quality defect. |
| exception_type | TARGET. Defect taxonomy: Balance Discrepancy, Time Travel, Impossible State Transition, Zombie Loan, or None. |

## Censoring convention

Forward-looking targets are null where the outcome window extends past the observation cutoff **and** the loan had not already reached an absorbing state. Loans that defaulted or prepaid inside the window retain their labels, because an absorbing state resolves the outcome with certainty -- censoring those rows would bias the observed event rate downward.

## Absorbing states

`Default` and `Prepaid` are terminal. A loan emits exactly one row in its absorbing state and none afterwards. Any subsequent active row is, by construction, a `Zombie Loan` exception.

## Structural nulls

`loss_severity_band` is populated only for `Default` rows. Its absence elsewhere is a business rule, not a data-quality defect, and should be excluded from missingness penalties.
