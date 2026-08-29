# AI Development Log

Required deliverable (Task 8, Agentic Coding Evidence — 5 points).

Kept incrementally, per session, rather than reconstructed at the end.

---

## Tools used

| Tool | Model | Used for |
| :--- | :--- | :--- |
| Claude (Cowork) | claude-opus-5 | Architecture, pipeline code, profiling logic, report generation |
| xAI Grok | grok-4 | Runtime reviewer copilot inside the product (Task 7), not development |

---

## Session 1 — Phase 0: repository and environment

**Date:** 2026-08-26

**Goal:** Reproducible skeleton, pinned dependencies, secret handling.

**Accepted:**
- Dependency set organised by the task each library serves, so an unused library
  is visibly unused rather than silently carried.
- `.gitignore` covering `.env`, `.venv/`, raw data CSVs, `mlruns/`.
- Single-seed reproducibility via `src/config.RANDOM_SEED`.

**Rejected / corrected:**
- *AI initially proposed Gemini for the copilot.* Changed to xAI Grok on the
  team's decision. The client was rewritten against the OpenAI-compatible
  endpoint rather than a vendor SDK, so the provider is now a two-line `.env`
  change instead of a code change — a better outcome than the original.
- *AI's first folder scaffold produced directories with a literal trailing comma*
  (`data,`, `reports,`) from a malformed shell brace-expansion. Caught by
  listing the tree before building on it. Lesson: verify the filesystem state
  after a scaffold command instead of trusting the exit code.

**Human review:** Every file read before commit. Directory structure verified
against the plan document.

---

## Session 2 — Phase 1: data intelligence and profiling

**Date:** 2026-08-26

**Goal:** All seven Task 1 requirements, reproducible, with a graded report.

**Representative prompts:**
1. "Build the Phase 1 profiling package: distributions, missingness, outliers and
   date validity, a rule engine that consumes validation_rules.json plus custom
   domain rules, correlations and categorical associations, train/test drift, and
   record + batch data-quality scores."
2. "The organiser's data pack hasn't arrived. Generate a synthetic stand-in
   matching the published schema with deliberately injected defects, so the
   profiler can be verified against known ground truth."

**Accepted:**
- Rule engine with an `applicable()` guard per rule, so a missing column
  degrades to a skipped rule rather than a crash. This matters because the real
  schema is not yet known.
- Bias-corrected Cramer's V rather than raw chi-square — plain chi-square
  inflates on high-cardinality fields like `state`, which would have produced a
  false "everything depends on everything" finding.
- PSI with quantile bins taken from the *train* distribution, plus a KS test as a
  second opinion.
- Synthetic data generator that records its injected defects to
  `data/_injected_defects.json`, giving a ground-truth set to validate detection
  against.

**Rejected / corrected:**
- *AI's first data-quality score returned 0.23% of records defect-free* — an
  obviously useless number. Root cause: `loss_severity_band` is null for every
  non-defaulted loan by design, and the scorer counted each of those as a missing
  value. Fixed by adding `detect_conditional_columns`, which identifies columns
  whose missingness is near-total in one segment and near-zero in another, and
  excludes them from the penalty. Score moved to a defensible 92.1/100 with 72%
  of records clean. **This was the most valuable catch of the session — the
  original number would have been reported as a finding and been wrong.**
- *AI's drift table ranked `loan_id` as the single most-drifted column* with
  PSI 6.6. An identifier has no distribution to compare. Added an exclusion for
  the ID column and any near-unique key. Without this, the real drift findings
  (`interest_rate`, `credit_score_band`) were pushed below the fold.
- *AI initially wrote the report builder to hard-depend on the `markdown` package.*
  Changed to fall back to preformatted HTML if the import fails — a reporting
  dependency should never be able to fail a data pipeline.

**Verification performed:** Ran the profiler against the synthetic pack and
compared detected rule violations to the injected ground truth. Detection rates
matched the injected counts scaled by the train share (~80%) across all six
injected rule defects — e.g. 124 injected `balance_exceeds_original`, 93 detected
in train. Date violations, source conflicts, staleness, and duplicates confirmed
present in their respective tables.

**Human review:** Every module read line by line. The quality-score formula and
the weight table were reviewed specifically, since they are judgement calls rather
than derived quantities and will need re-tuning against the real defect mix.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI,
100% reviewed by a human, with two substantive logic corrections (conditional
missingness, ID exclusion from drift) originating from human review of the output
rather than from the AI.

---

## Session 3 — data sourcing: synthetic vs. public loan-performance data

**Date:** 2026-08-26

**Trigger:** Human challenged the decision to generate synthetic data when the
problem statement supplies links to real public datasets.

**Investigation (AI, verified against primary sources):**
- **HMDA** (FFIEC/CFPB) is freely downloadable with no registration, but the LAR
  field documentation confirms it is *application and origination data only* —
  no payment status, no delinquency, no monthly records. A forward-looking
  target such as `next_3m_delinquency_flag` cannot be constructed from it, so it
  cannot drive Tasks 2–5. Wrong shape, not merely inconvenient.
- **Fannie Mae / Freddie Mac** single-family loan performance data is exactly the
  right shape, but requires account registration and acceptance of a license
  agreement. Freddie's terms prohibit "further distributing or making available
  the ... Dataset or any derived products to any third party."
- **Consequence for this project:** §13 of the problem statement lists "uses
  public data in violation of source terms" as a disqualification condition.
  Committing downloaded CSVs to the public GitHub repo — itself a required
  deliverable — would therefore be disqualifying. `.gitignore` already excludes
  `data/*.csv`; this must stay true.

**Decision:** Not either/or. Real data is the better basis for calibrating rule
thresholds and quoting honest model performance. Synthetic data retains one
property real data structurally cannot offer: **known ground truth about its own
defects.** When the profiler reports 93 balance violations against 124 injected,
detection is *verified*; against real data, an anomaly count is unfalsifiable.
Synthetic therefore stays as the test fixture; real data becomes the calibration
and evaluation set once the human completes the licensed download.

**Rejected / corrected:**
- *AI built the synthetic generator without surfacing the alternative.* The
  reasoning was sound but never stated, so the human had to prompt for it. A
  design decision presented without its discarded alternatives is not reviewable.
  Correction: state alternatives considered, not just the path taken.

**Human review:** Human raised the challenge, and retains the licensing decision —
the AI explicitly declined to register or accept dataset terms on the human's
behalf.

---

## Session 4 — Phase 0 (data): five-phase synthetic generation suite

**Date:** 2026-08-26

**Goal:** Replace the throwaway generator with a benchmark-grade suite: 50,000
loans, ~1.7M panel rows, under two minutes.

**Representative prompt:** A five-phase architectural blueprint specifying a
Gaussian copula static attribute space, a risk-scaled discrete-time Markov chain,
vectorised forward-looking targets with censoring, labelled anomaly injection,
and time-aware partitioning with supporting artifacts.

**Accepted:**
- **True Gaussian copula** (MVN → normal CDF → truncated-normal PPF per margin)
  rather than sample-and-clip. Clipping piles mass on the bounds and attenuates
  the very correlation being specified. Achieved −0.4530 / −0.4462 against a
  −0.45 target.
- **Vectorised-across-loans Markov engine.** The chain is sequential in time but
  parallel across loans, so the Python loop runs 60 times (months) rather than
  50,000 times (loans). Whole-cohort simulation: 6.7s.
- **Closed-form amortisation** rather than month-by-month accumulation: exact,
  no floating-point drift, and vectorises. Payment counter increments only on
  `Current` months, so delinquent loans correctly stop amortising.
- **Block-geometry target shifts.** Forward windows computed as array shifts with
  run-length-derived validity masks, avoiding any groupby-apply over 50,000
  groups.

**Rejected / corrected:**
- *The blueprint requested "multi-core batching".* Declined for the simulation,
  with measurement rather than assertion: simulation is 6.7s of a 47.8s run,
  while CSV serialisation is 31.9s. A process pool would spend more on pickling
  state arrays than it recovers. Threads were applied where they actually pay —
  the four independent CSV writes, where pandas releases the GIL. Adding
  multiprocessing to the simulation would have been performance theatre.
- *Deviated from the literal censoring specification.* The naive rule nulls every
  target within H months of the window end, including loans that already reached
  an absorbing state. But absorption resolves the outcome with certainty, so
  those labels are known; censoring them discards resolved outcomes near the
  cutoff and biases the observed default rate downward. Implemented rule censors
  only when the window is incomplete **and** the loan was still active.
  Documented in `data_dictionary.md`.
- *Flagged rather than silently retuned:* the specified transition matrix yields
  **19.8% lifetime default** against 1–3% for real GSE prime collateral. The
  matrix behaves exactly as specified (90-DPD → Default at 0.45/month against a
  0.10 cure rate). Implemented verbatim and raised to the human as a judgement
  call, with the two-line change identified. Silently "fixing" a human-specified
  parameter would have been the worse error.
- *Three implementation bugs caught in verification:* `PeriodIndex` has no `.dt`
  accessor; an `int16` column rejected an `int64` anomaly assignment; and the
  profiler crashed taking quantiles of a boolean column (`is_numeric_dtype`
  returns True for bools, `quantile()` throws). All three surfaced only by
  running the code, not by reading it.
- *A false alarm worth recording.* Verification appeared to show 44 rows
  following an absorbing state without a `Zombie Loan` label. Investigated before
  changing anything: they were test-partition rows whose labels had been
  correctly stripped by design. **Nearly "fixed" correct behaviour.** Reproducing
  a suspected bug in isolation must precede editing.

**Verification performed:**
- Invariants on non-anomalous rows all zero: balance-exceeds-original, negative
  balance, reporting-before-origination, prepaid-with-balance, default-with-DPD<90.
- `loss_severity_band` populated on exactly the `Default` rows, null elsewhere.
- All 130 post-absorbing rows in train labelled `Zombie Loan`.
- Generated `validation_rules.json` confirmed consumable by the existing
  profiling rule engine: `BALANCE_CEILING` catches Balance Discrepancy,
  `TEMPORAL_ORDERING` catches Time Travel.
- End-to-end timing at full scale: 47.8s for 1,722,799 rows.

**Design property worth noting for the demo:** two of the four injected anomaly
types (Balance Discrepancy, Time Travel) are catchable by row-level deterministic
rules; the other two (Impossible State Transition, Zombie Loan) require
sequence-aware detection and are deliberately invisible to rules. This gives the
Task 4 ML component something rules genuinely cannot do, rather than duplicating
the rule engine.

**Approximate AI-generated code share this session:** ~95% of lines drafted by
AI, 100% human-reviewed. Two substantive design decisions (declining
multiprocessing, flagging the default rate) originated from AI analysis and were
escalated to the human rather than resolved unilaterally.

---

## Session 5 — Phases 2 and 3: feature engineering and loan performance prediction (Task 2)

**Date:** 2026-08-26

**Goal:** The Task 2 pipeline end to end — time-aware split, five targets,
baseline vs. improved model, imbalance handling, calibration, full metric table.

**Representative prompts:**
1. "Write a production-grade modular pipeline for Task 2: strictly time-aware
   splitting on `reporting_month` with random row-level splitting explicitly
   forbidden, all five targets, a baseline model and an improved LightGBM model,
   a robust class-imbalance strategy, probability calibration with Brier before
   and after, and one comparison results table."
2. "Split the code logically for ingestion, preprocessing, training and
   evaluation."

**Accepted:**
- **Horizon purging on top of the time split.** Splitting on `reporting_month`
  alone is not sufficient: a row labelled `next_12m_default_flag` at 2021-06
  describes outcomes through 2022-06, which is the validation window. The last
  *horizon* months of each window are dropped per target, and
  `reports/task2_split_audit.csv` records exactly which months were purged.
- **`audit_split` raises rather than warns.** A random split is invisible in the
  metrics — it just makes them better — so the audit is an assertion, and
  `tests/test_leakage_controls.py` feeds it a deliberately shuffled split to
  confirm it fails.
- **Rolling features computed on a gap-free monthly grid.** The panel has
  injected missing months; a position-based `rolling(3)` on gapped rows silently
  reaches further back than three calendar months. The grid makes every window
  mean what its name says.
- **Reweighting over SMOTE for imbalance,** because a synthetic minority row
  carries an interpolated rolling history ("months since last delinquency",
  "paydown over 6 months") belonging to no loan that ever existed.
- **Backend fallback chain** LightGBM → XGBoost → sklearn `HistGradientBoosting`.
  Both boosters need an OpenMP runtime that was in fact missing on this machine
  (`libomp`), which is precisely the failure a judge's laptop can hit.

**Rejected / corrected:**
- *LightGBM early-stopped at one tree.* The default `binary_logloss` was being
  tracked alongside `average_precision`, and log-loss on a `scale_pos_weight`
  model degrades from the first iteration by construction, so early stopping
  fired immediately. Fixed by setting `metric` explicitly and
  `first_metric_only=True`; the improved model went from 1 tree to 74.
- *Multiclass `log_loss` came back at 8.72 and OvR AUC as NaN* while the same
  probabilities scored a healthy Brier. Root cause: scikit-learn binds
  probability *columns* to labels in **sorted** order inside `log_loss` and
  `roc_auc_score`, regardless of the order the `labels` argument is given in.
  This pipeline keeps classes in severity order for readability, so every class
  was being scored against the wrong column. Log-loss dropped to 0.2326 once the
  columns were re-sorted for those two metrics. A regression test now asserts the
  multiclass metrics are invariant to class ordering.
- *The initial split boundaries silently emptied the validation window* for the
  two 12-month targets — the window was exactly 12 months wide, so purging
  removed all of it, and the failure surfaced as a `SimpleImputer` error deep in
  sklearn. Widened the window to 18 months and added an explicit guard that names
  the cause.
- *The AI's first draft aligned `groupby().rolling()` output by position.* That
  ordering is not contractual; replaced with an explicit reindex.

**Verification performed:** 11 regression tests in `tests/test_leakage_controls.py`,
including a truncation-invariance test — rebuilding features on a panel cut off
after month *t* must reproduce month *t*'s rolling features exactly, which is the
operational definition of "backward-looking". Full pipeline run end to end in ~90s.

**Findings reported rather than tuned away:**
- Prepayment is close to unpredictable from this pack: ROC-AUC 0.52, PR-AUC 0.09
  against a 0.09 base rate, precision never reaching 50% at any threshold. Checked
  against the generator: its prepayment hazard depends only on credit band, and an
  oracle using the true band hazard scores AUC 0.55 on the same window. The signal
  is not there, and the table says so.
- The improved model's ROC-AUC on `next_3m_delinquency_flag` is marginally *worse*
  than the baseline's (0.753 vs 0.760) while PR-AUC is materially better. Both
  numbers are reported.

**Human review:** Every module read line by line. The three corrections above
(early stopping, multiclass label ordering, empty validation window) came from
reading pipeline *output* that looked wrong, not from reading the code.

**Approximate AI-generated code share this session:** ~95% of lines drafted by AI,
100% human-reviewed.

---

## Session 6 — Phase 4: time-to-event / survival modelling (Task 3)

**Date:** 2026-08-28

**Goal:** Competing-risk survival modelling for default and prepayment, with
right-censoring handled explicitly, a baseline to beat, and segmented event curves.

**Representative prompts:**
1. "Write a production-grade survival pipeline using `lifelines`: convert the panel to
   duration/event format for two competing risks, handle right-censoring for loans still
   active at the cutoff, compare a naive constant-hazard baseline against an advanced
   model, and produce event curves segmented by credit band and vintage."
2. "Include a written explanation of how right-censoring was treated — it is a strict
   grading requirement."

**Accepted:**
- **Censoring as the design centre, not an afterthought.** All 6,072 censored loans stay
  in the sample and contribute exposure to every risk set they survive through. Dropping
  them would leave only resolved loans, and 1,848 of 3,813 resolved loans defaulted — a
  "default rate" of 48% against the 36-month Aalen-Johansen estimate of 21%. The four
  cases (administrative censoring, loss to follow-up, competing event, left truncation)
  are separated in the code and counted in the report.
- **Prepayment as a competing risk, not as censoring.** Cause-specific hazards for
  estimation, Aalen-Johansen for cumulative incidence, both reported. The naive
  `1 - KM` overstates 36-month default incidence by 2.9pp, and that number is printed
  in the report rather than described.
- **Split by vintage, not by reporting month.** A duration model cannot represent half a
  loan, so the Task 2 reporting-month split does not transfer; splitting on origination
  keeps histories intact and still puts the holdout strictly forward in time.
- **A hand-rolled Aalen-Johansen estimator.** `lifelines.AalenJohansenFitter` resolves
  tied event times by random jitter, and monthly durations are nothing but ties. The
  discrete-time formula is exact and deterministic; a test cross-checks it against
  lifelines on untied data, which is what makes the substitution safe rather than merely
  convenient.

**Rejected / corrected:**
- *The AI's first pass read each loan's outcome from `groupby().last()`.* Correct-looking
  and wrong: 664 loans in this pack emit a `Current` row **after** their absorbing
  `Default`/`Prepaid` row — the Phase 1 Zombie Loan defect. Reading the last row
  reclassifies those 664 resolved loans as still active, moving them from the event count
  into the censored count and biasing every curve downward. Changed to read the *earliest
  absorbing row by loan age* and discard everything after it. Caught by comparing the
  loan-level event counts (1,511 default) against the panel's absorbing-row counts
  (1,848) and asking why they differed.
- *The AI reported 1,528 left-truncated loans.* They are not left-truncated. Every loan in
  the pack has an age-0 row; those 1,528 have a *calendar*-first row at a later age
  because their age-0 row carries a corrupted `reporting_month` — the Phase 1 Time Travel
  defect. Reading delayed entry off calendar order manufactures 1,528 truncated loans out
  of a data defect. Durations are now taken on the loan-age axis throughout, the
  truncation machinery is kept (a real servicing extract does start mid-life) and the
  calendar-vs-age disagreement is counted in the censoring report.
- *The first segmented curves ran flat to month 59 for every vintage.* The 2023 cohort has
  15 months of follow-up; a flat cumulative-incidence tail past that reads as "these loans
  stopped defaulting" when it means "we stopped watching". Every curve is now truncated at
  the month its own risk set falls below a threshold, and each panel is labelled with its
  follow-up length. This turned the vintage figure from a misleading chart into the
  clearest illustration of censoring in the report.
- *Multiclass-style overlay for the six credit bands was dropped.* Six curves times two
  causes is twelve lines on one axis, and a validated single-hue ordinal ramp cannot hold
  six distinguishable steps at the required lightness gaps. Faceted into small multiples
  instead, which keeps one colour per event type across every figure in the report.

**Verification performed:** 16 regression tests in `tests/test_survival_censoring.py`,
including the competing-risk identity (`CIF_default + CIF_prepaid + S = 1` at every t),
the direction of the naive-KM bias, the cross-check against lifelines, and an assertion
that dropping censored loans *raises* the fitted hazard — so a future change that quietly
filters them out fails a test rather than a report nobody re-derives. Every figure was
rendered and inspected, not just generated.

**Findings reported rather than tuned away:**
- Cox beats the constant-hazard baseline decisively on default (IBS 0.044 vs 0.065,
  C = 0.822) and barely at all on prepayment (IBS 0.070 vs 0.071, C = 0.558) — the same
  split Task 2 found, from a completely different modelling family.
- The Cox model over-predicts default incidence on the newest vintages at horizons past
  ~25 months, where the constant-hazard baseline tracks the level better. Both curves are
  plotted against the observed one rather than only the flattering comparison.
- The Schoenfeld tests flag proportional-hazards violations; they are reported in the
  model card rather than corrected, because the assumption is what the Cox apparatus rests
  on and quietly breaking it is a standard way survival results go wrong.

**Human review:** Every module read line by line. All three corrections above came from
reading pipeline *output* — a count that did not reconcile, a statistic that was too
convenient, a curve that flattened — not from reading the code.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI, 100%
human-reviewed, with the three substantive corrections originating from review of the
output.

---

## Session 7 — hardening Phases 0-2: charts, feature dictionary, one-command reproduce

**Date:** 2026-08-28

**Goal:** Close the gaps a self-assessment found in the phases already marked done —
a profiling report with no figures, an undocumented feature matrix, and a repository
that could not be rebuilt with one command.

**Representative prompts:**
1. "Which phases are complete, and how do they score out of 10?"
2. "Make them push to 9+ for phases 0, 1 and 2."

**Accepted:**
- **One chart palette for the whole project** (`src/viz.py`). The survival figures had
  their own tokens; those were folded into a shared module so the Data Intelligence
  Report and the survival report read as one document. Categorical hues assigned in a
  fixed order and never cycled, single-hue sequential, and the status colours reserved
  for state with a text label always attached.
- **Five figures in the Data Intelligence Report**, which previously had none: a
  missingness bar plus a by-state heatmap, numeric distributions with the Tukey fences
  drawn on, train/test drift, the data-quality score distribution over time, and rule
  violation counts.
- **A generated feature dictionary.** Built from the feature matrix that was actually
  constructed rather than maintained by hand, so it cannot drift from what the models
  train on. Each of the 59 entries carries an **information window** (`as-at t`,
  `t-k..t`, `month t`) — the leakage claim, stated per feature, in the same document a
  judge reads the data description from. An undocumented feature is emitted as
  `unclassified` with a warning rather than silently omitted.
- **`make all`**, which runs profile → predict → survival → profile → test in 2m12s.
  The doubled profiling step is not redundant: profiling produces the data-quality
  scores the feature matrix consumes, and prediction emits the feature dictionary the
  profiling report folds in. Running it at both ends is what resolves that reference.
- **CI on every push**, running the regression suite plus a 400-loan smoke run of all
  three pipelines. Unit tests on synthetic frames cannot catch a CLI regression or an
  import error; the smoke run can.

**Rejected / corrected:**
- *The first drift chart was a linear bar chart.* PSI on this pack spans 0.000 to 4.2 —
  four orders of magnitude — so thirteen of eighteen fields collapsed to zero-length
  stubs and the chart conveyed only that two fields were large. A log axis on bars is
  not the fix, because a bar has to run from zero to be honest. Changed the *mark*
  instead: a dot plot on a log axis, where position carries the value. Its two threshold
  labels also overlapped into unreadable overlap and were separated vertically.
- *The drift figure silently never rendered.* The call was inserted after the
  `else:` branch of `if not test.empty`, so it only ran when there was no test file —
  and `drift_tbl` would not have existed there. Nothing raised; the chart was simply
  absent. Caught by listing the output directory rather than by reading the log.
- *The heatmap cells were indistinguishable at the light end.* Near-zero is *supposed*
  to recede toward the surface in a sequential ramp, but that lost the grid entirely.
  Added hairline surface-coloured separators between cells.
- *Six credit bands as six categorical hues, revisited.* The validator was run on the
  candidate six-step single-hue ordinal ramp and it failed the adjacent-lightness gate —
  the usable band between "clears the surface" and "black" cannot hold six
  distinguishable steps. Kept the small-multiples faceting rather than shipping a
  palette that fails its own check.
- *Aspirational dependencies removed.* `optuna`, `mlflow`, `seaborn`, `jupyter`,
  `missingno` and `imbalanced-learn` were in `requirements.txt` and imported nowhere.
  Removed, with a comment recording *why* each was dropped — `imbalanced-learn` in
  particular was a deliberate rejection (SMOTE on a temporal panel), and a reader
  should be able to see that it was considered rather than forgotten.

**Findings the new charts surfaced:**
- The data-quality score sits at ~38 for every month of 2017 and jumps to ~92 from
  2018 onward. Investigated rather than described: **all 188 rows dated 2017 predate the
  earliest origination in the entire book (2018-01)**. They are corrupted timestamps
  describing months in which no loan existed — the same Time Travel defect that made
  1,528 loans look left-truncated in Phase 4. Added
  `outliers.impossible_reporting_periods` as a new domain check so this is a named
  finding in the report rather than an unexplained step in a chart.
- The missingness heatmap makes the structural-missingness argument in one look:
  `loss_severity_band` is 100% missing in every state except `Default`, where it is 0%.
  That was previously a paragraph asking the reader to take it on trust.

**Verification performed:** `make all` end to end (2m12s), 27 tests passing, every
figure rendered and inspected individually. The DI report now carries 10 sections
including figures and the feature dictionary.

**Human review:** The self-assessment that started this session was itself the review
step — the gaps it named (no charts, no feature dictionary, nothing committed) were
found by auditing deliverables against the rubric, not by re-reading code.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI,
100% human-reviewed.

---

## Session 8 — Phase 5: anomaly & exception detection (Task 4)

**Date:** 2026-08-29

**Goal:** A hybrid detector combining deterministic rules with unsupervised ML, supervised
prediction of `exception_required` and `exception_type`, per-record drivers, and a curated
reviewer queue of at least 20 examples.

**Representative prompts:**
1. "Combine deterministic rule violations from `validation_rules.json` with an unsupervised
   anomaly score; predict exception probability and type; extract top drivers; curate at
   least 20 reviewer-ready examples to `reports/anomaly_examples.csv`."

**Accepted:**
- **Sequence-aware detectors as a *deterministic* layer.** Phase 1's row-level engine caught
  100% of Balance Discrepancy and Time Travel and ~10% of Impossible State Transition and
  Zombie Loan, because an expression evaluated against one row cannot see last month's
  status or know the loan was already terminal. `post_absorbing_activity` and
  `illegal_status_transition` take overall recall from 52% to 99.7%. They are rules, just
  rules that need history, so they are reported as deterministic rather than as an ML win.
- **Noisy-OR score combination.** `hybrid = 1 - (1 - rule)(1 - ml)`, with `rule_score`
  itself a noisy-OR over severity weights. A fired high-severity rule sets a floor the
  model cannot argue down. Under a plain sum, three `missing_document_status` flags would
  outrank one balance exceeding origination.
- **Stratified curation.** Top-25 by score would have returned twenty near-identical
  Balance Discrepancy rows. The queue guarantees a block per predicted type plus five
  reserved slots for high-scoring records with *no* rule violation — the only rows capable
  of teaching the rule set something.
- **LightGBM native `pred_contrib` for drivers** rather than the `shap` package. TreeSHAP
  runs inside the booster in one pass: exact attribution, no background sampling, no
  memory blow-up. Running an exact attribution against the Isolation Forest was rejected
  for a different reason than cost — its output is an isolation depth, not a probability,
  so an exact attribution of it is an exact attribution of something a reviewer cannot
  interpret. Robust deviation in sigma units is used for that path instead.

**Rejected / corrected:**
- *Four of the organiser's own rules were silently never evaluated.* Phase 1 reported
  `SEQUENTIAL_DELINQUENCY`, `ABSORBING_STATE_FINALITY`, `DEFAULT_DELINQUENCY_CONSISTENCY`
  and `MUTUALLY_EXCLUSIVE_TERMINATION` as "not applicable to this data". They were not:
  the expression parser extracted required columns with a bare identifier regex, so
  `'90-DPD'` contributed `DPD` and `and`/`not` contributed themselves. No real frame could
  satisfy the requirement, so `applicable()` returned False and the rules were skipped.
  Fixed by stripping string literals and operator keywords. **On this pack the four rules
  find zero violations, so detection is unchanged — the fix removes a silent blind spot,
  not a miss.**
- *My own ablation leaked the signal it claimed to remove.* The "no sequence information"
  head still received `rule_score`, which is a noisy-OR over *every* signal including the
  sequence detectors. The ablation therefore proved nothing. Fixed by computing a separate
  row-level aggregate; a test now asserts the property.
- *The detector comparison chart was the wrong form.* A precision/recall scatter put four
  detectors on the same point — at a fixed queue size they flag the same records and catch
  nearly everything — and their labels overlapped into mush. What actually separates them
  is ranking, so it became precision-within-the-top-k as the queue lengthens, which is
  also the question a queue owner asks.
- *The reviewer queue lost a label on every clean record.* The clean class was the literal
  string `"None"`, which pandas reads back from CSV as NaN. Renamed to `"No exception"`,
  with a round-trip regression test.
- *Driver strings reported deviations of 71 million MAD.* `balance_vs_scheduled` is exactly
  1.0 for most records, so its MAD is zero and the z-score exploded. Added an IQR fallback
  scale and a display cap. A formatting accident was being presented to a human as a
  finding.
- *The driver chart mislabelled `rule_score` as "learned".* Prefix matching classified the
  deterministic aggregate as learned record state and the sequence *context* features as
  learned too, so the chart claimed the model leaned on inference where it leaned on
  rules. Replaced with an explicit taxonomy: rules 41.9%, record state 29.7%, sequence
  context 24.5%, sequence detector flags 3.9%.

**Findings reported rather than tuned away:**
- **The task is near-solved because the defects were injected, not because the model is
  good.** The supervised head reaches 0.999 precision at 0.999 recall and perfect per-class
  F1. Every defect class carries a near-deterministic fingerprint. The report leads with
  this caveat rather than burying it, because a reader who takes the headline at face value
  will overestimate what transfers to a real servicer feed.
- **The learned layer did find something the rules did not.** A model given no sequence
  information still reaches 0.979 PR-AUC. Investigating its dominant feature showed why:
  Zombie Loan rows carry a stale balance (mean `balance_vs_scheduled` 0.77 vs 0.99 clean),
  and Impossible State Transition rows sit at *exactly* 1.000 where genuine 90-DPD records
  sit at 1.005+, because a real three-months-delinquent loan has accrued arrears and the
  injected row was written with a performing loan's balance. Neither fingerprint was
  anticipated by any hand-written rule.
- **Every one of the Isolation Forest's top unsupported flags is benign.** All five are
  ordinary terminations — `Default` or `Prepaid` at zero balance. An unsupervised model run
  alone would head the reviewer queue with correct loan closures. This is the concrete case
  for the hybrid, and it is the same lesson stated a priori in the task brief; it is
  reported here with the five specific records that demonstrate it.

**Verification performed:** 20 new regression tests (47 across the project), including the
ladder-versus-jumper test that stops the transition detector from simply flagging every
90-DPD row, the noisy-OR floor property, and the ablation-integrity assertions. Every
figure rendered and inspected.

**Human review:** Every module read line by line. Four of the six corrections above came
from reading output that looked too good — a rule marked inapplicable, an ablation that
scored as well as the full model, a driver string with an absurd number.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI, 100%
human-reviewed.

---

## Session 9 — Phase 6: scenario & stress simulation (Task 5)

**Date:** 2026-08-29

**Goal:** Project the Phase 3 models under the three macro scenarios, segment the impact,
and attribute the movement to features.

**Representative prompts:**
1. "Ingest `macro_scenarios.csv`; apply the macro scalars to the feature set or to baseline
   hazard rates; project portfolio delinquency, default and prepayment per scenario;
   segment by vintage, credit band, state and servicer; extract the top 3-5 drivers."

**Accepted:**
- **The scenario file as the only source of assumptions.** A missing column stops the run
  rather than falling back to a default. Nothing in the package invents an unemployment
  path, a rate shock or an elasticity.
- **Three transmission channels, ranked by how much they assume.** House prices to LTV is
  arithmetic (`100 / hpi_index`). Market rate to refinance incentive is arithmetic. The
  third -- labour market to credit quality -- has no stated elasticity, so it is **solved
  for** rather than assumed: find the portfolio-wide credit-score shift that makes the
  model reproduce the file's own stated default multiplier. The file stays authoritative,
  the model supplies the transmission, and the answer lands in a unit a credit officer
  recognises: a 1.77x multiplier is this book losing 93 points of FICO.
- **Bands rebuilt from the value underneath them.** Shifting `ltv` while leaving `ltv_band`
  at its original level hands the model a record that contradicts itself, and the model
  scores it without complaint. Every stressed feature is clipped to a plausible range and
  the clip count reported.
- **Driver attribution from the model's own contributions**, not from the scenario's name.
  The change in each feature's mean contribution between baseline and stressed portfolios
  is its share of the change in the rate, so the written narrative cannot drift from what
  the model did.

**Rejected / corrected:**
- *A single projection method would have hidden the headline finding.* The first design
  reported only the feature-stress projection. It turned out the credit channel
  **saturates**: past month 24, no credit-score shift reproduces the stated multiplier --
  even moving the whole book to the floor of the observable score range tops out at 2.07x
  against a stated 2.60x. A naive calibration clamps at its search bound, reports the
  clamp, and the projection quietly under-states the scenario. Added the stated-multiplier
  projection as a second first-class method, a saturation table naming the shortfall, and
  a dotted overlay on the chart so the gap is drawn rather than described.
- *The delinquency panel was showing a "stated multiplier" the file never states.* The
  first `multiplier_for` fell back to the default multiplier for any measure that was not
  prepayment, which put a number in the report and credited it to a source that never gave
  it. Now returns `None` and the column is dropped for measures the file does not cover.
- *The calibration solved all 48 horizons to report five.* Each solve is a root-find
  costing several full-portfolio scorings. Restricted to the requested horizons and cached
  the baseline scoring per horizon.
- *The credit-band axis was in scrambled order* -- `<620, 800+, 740-799, ...` -- because
  the segment values were sorted alphabetically. An ordinal axis out of order is a chart
  that has to be decoded rather than read.
- **I hand-wrote the README results table from numbers on screen, and two of them were
  from the 500-loan smoke run rather than the full one.** Caught by re-reading the CSV
  before committing. The table and the three prose figures are now generated from
  `reports/scenario_report.csv` so they cannot drift again.

**Findings reported rather than tuned away:**
- The scenario file's severity exceeds what the model can express through credit score
  alone. Reported as a saturation table with the shortfall in the file's own units, and
  the feature-stress figure is labelled a floor rather than a forecast.
- **Prepayment barely responds to the rate path**: the file states 2.9x by month 48, the
  feature-stress projection gives 1.06x. Consistent with Task 2 (ROC-AUC 0.52) and Task 3
  (Cox C = 0.558) -- a model with no prepayment signal cannot acquire one under stress.
  Three phases, three methods, the same conclusion.
- The stress concentrates in the middle credit bands. The `<620` band barely moves under
  Adverse-Credit because it is already near its risk ceiling; the movement lands on
  620-659 and 660-699.

**Verification performed:** 13 new regression tests (60 across the project), including a
calibration test with a known closed-form answer and a test that an unreachable multiplier
is reported rather than silently clamped. Every figure rendered and inspected. Full
`make all` run end to end.

**Human review:** Every module read line by line. The saturation finding came from reading
a calibration log line that said `-250.0 pts [not attainable]` and asking what a -250 point
FICO shift would even mean.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI, 100%
human-reviewed.

---

## Session 10 — Phases 7 and 8: explainability, governance, and the LLM copilot

**Date:** 2026-08-29

**Goal:** SHAP-based global and local explanations, error analysis, calibration and a
fairness screen (Task 6); a grounded LLM reviewer copilot with a mandatory audit trail and
deliberate failure testing (Task 7). Plus the model card that section 11 requires.

**Representative prompts:**
1. "SHAP global and local explanations for the Phase 3 models, false positive/negative
   analysis by segment, calibration curves, and a bias check across state or credit band."
2. "A grounded LLM layer that assembles the loan record, data dictionary, triggered rules
   and ML outputs into a prompt; log every call to `llm_prompt_log.jsonl`; append a
   'recommendation, not a decision' disclaimer; and deliberately trigger a failure."

**Accepted:**
- **SHAP explains the booster, and the report says so.** `TreeExplainer` decomposes the base
  model's log-odds, not the calibrated probability that is deployed. The calibrator is
  monotone so it cannot reorder contributions, but the decomposition does not sum to the
  deployed probability, and claiming otherwise would assert something the arithmetic does
  not support. Error analysis, reliability and disparity all run on the calibrated
  probability at the tuned threshold instead.
- **A disparity screen with three guards.** Ratio floor, a two-proportion significance test,
  and an exemption for legitimate risk factors. Without the significance test the screen
  escalated 19 of 45 segment-metric pairs, including California against New York on roughly
  twenty events.
- **Guardrails as controls, not instructions.** A prompt asking a model not to invent numbers
  checks nothing. Prediction language, decision language and numeric grounding are checked on
  every response, and a failed note is withheld in full rather than repaired.
- **Provider auto-detection from the key prefix.** Not cosmetic: see below.
- **Offline mode as the default for the copilot.** A script that spends the account's credits
  and sends data to a third party should not do so because someone forgot a flag.

**Rejected / corrected:**
- *`vintage_year` was being escalated as a fairness finding.* It is almost pure loan age
  within one reporting window -- mean age runs from 54 months for the 2018 cohort to 3.5 for
  2023 -- and seasoning is a legitimate driver of default hazard. The 2018 cohort is also
  survivorship: 232 loans left, 41% of which actually default. Reclassified as a risk factor,
  with the confound measured rather than asserted.
- *The prepayment model's disparity findings were the loudest in the report.* They are an
  artefact: that model flags 53.7% of the book at its tuned threshold, which makes every
  group gap enormous and meaningless. Findings are now suppressed as uninterpretable above a
  selection-rate ceiling, with the reason carried in the table.
- *A schema collision found by my own test.* `error_rates_by_segment` inserted a column named
  `segment` alongside the segment's own column, which broke outright when a segment was
  itself called "segment". The frame now always returns `segment` (the name) and `group` (the
  level), so its schema no longer depends on the data.
- *Waterfall filenames collided.* "confident true positive" and "confident false positive"
  both slugged to `confident`, silently dropping one of the four demo cases.
- *The audit log was missing the guardrail verdict entirely.* Logging happened the instant
  the call returned -- before the check ran -- so the most important governance field was
  empty in every record. Fixed with a second append-only record joined on `call_id`, which
  keeps the property that makes the log evidence: the call record is written before the check,
  so a crash between the two cannot erase the fact that the call happened.
- *The first live run failed on every call: `ModuleNotFoundError: openai`.* The package is in
  `requirements.txt`; the virtualenv had drifted from it.
- *The second live run failed on every call: "Incorrect API key provided".* The key in `.env`
  is a **Groq** key (`gsk_` prefix) and the client was hardcoded to xAI's endpoint --
  `.env.example` even said "Get an xAI API key at console.groq.com". The error reads as a bad
  key rather than a wrong endpoint, which is exactly the wrong diagnosis to hand someone.
  Replaced with detection from the key prefix, and an explicit `.env` setting still wins.

**Findings reported rather than tuned away:**
- **The model passed all six adversarial probes**, including a false-premise attack ("given
  that this borrower recently lost their job...", a fact in no field of the panel) and a
  role-reassignment attack. Manufacturing a model failure to fill the section would have been
  worse than useless.
- **So the honest failure examples are failures of my own control.** The grounding check
  blocked correct notes three times: Unicode hyphens shredded `2023-06-01` into "06" and "01";
  pandas Timestamps in the record grounded nothing, so any note quoting a date it was *given*
  could be blocked; and "3-5 sentences" -- the pipeline's own prompt text -- parsed as the
  number `-5`. Each is recorded in the report with its fix. A grounding check with false
  positives is a serious governance defect rather than a footnote: reviewers learn to ignore
  the flag, and the one real hallucination goes out with the rest.
- Nine disparity findings survive the screen, all geographic or servicer-level. The default
  model's false-positive rate in WI is 10.0% against 1.0% in OK (p ~ 5e-14). Those are in the
  model card.

**Verification performed:** 35 new regression tests (95 across the project). The grounding
fixes were verified by replaying all 30 recorded live outputs through the corrected check
before spending further API credits, rather than by re-running live each time. Every figure
rendered and inspected.

**Human review:** Every module read line by line. Four of the seven corrections came from
reading output that looked wrong in a specific way -- a note withheld for quoting its own
loan's origination date, an audit record with an empty verdict field, a disparity screen that
flagged more than it cleared.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI, 100%
human-reviewed.

---

## Session 11 — Phase 9: packaging, submission and reproducibility

**Date:** 2026-08-29

**Goal:** Score the unlabelled panel, write a `submission.csv` conforming to the template,
provide a single-command entrypoint, and generate the model card section 11 requires.

**Representative prompts:**
1. "Run the full inference pipeline on the unlabelled test set, format strictly to
   `submission_template.csv`, provide a `main.py` entrypoint and a comprehensive
   `requirements.txt`, and draft a model card with a pre-filled leakage-controls section."

**Accepted:**
- **The template is read at run time and treated as the contract**, not hardcoded. A change
  the organiser makes to it then surfaces as a validation failure on the next run instead of
  as a silently wrong submission -- the failure mode that costs a whole entry.
- **Validation before the write, and a structural failure refuses it.** A file with the
  wrong columns is not a partially-good submission; writing it anyway just moves the
  discovery later.
- **Rows aligned by `(loan_id, reporting_month)`, never by position.** A submission in a
  different order than the template looks correct on inspection and is completely wrong to a
  scorer that joins on index.
- **A model card generated from the measured tables**, so its limitations section cannot
  quietly diverge from what the pipeline found -- the failure mode that makes most model
  cards worthless: they describe the model the author meant to build.

**Rejected / corrected:**
- *`exception_required` was firing on 13.8% of the book* against a 2.6% base rate, because
  the first version used the raw "any rule fired" flag -- almost all of it one low-severity
  document check. Replaced with the supervised head's judgement, fitted on the labelled
  panel. Rate fell to 2.01%.
- *Then it fired on 0.00% -- one row in 78,409.* The exception head's probabilities stay
  compressed below 0.52 because the sequence detectors are near-perfect indicators, average
  precision saturates within ten boosting rounds, and early stopping correctly halts at
  `best_iteration = 10`. The model ranked fine; the hardcoded 0.5 threshold was wrong. The
  threshold is now tuned on a held-back window and travels with the predictions. **Two
  opposite failures from the same column in one session, and the second only became visible
  because the first was fixed.**
- *`exception_type` and `top_drivers` came back as NaN for every clean row.* The literal
  string `"None"` and the empty string both pass an in-memory null check and read back from
  CSV as NaN -- the same bug that bit the Phase 5 reviewer queue, recurring because the
  validation checked the in-memory frame. Validation now round-trips through a CSV buffer
  and checks what a consumer would actually see.
- *Every cleared row carried the action "No rule fired; the model flagged this on its
  pattern alone. Review manually and write the rule that would have caught it."* That text
  is correct for a high-scoring rule-clean row and nonsense for the 98% of the book that is
  simply fine. The action now depends on whether anything was flagged.

**On the no-refit requirement:** the prompt anticipated a scaler or imputer being refitted on
the test set. That path does not exist here -- each `FittedModel` carries the fitted sklearn
pipeline and the train-fitted `CategoryHarmoniser`, and `predict_proba` calls `transform`
only. Rather than claim a correction that did not happen, I wrote the test that proves it:
`test_scoring_does_not_refit_the_preprocessing` scores a deliberately shifted distribution
and asserts every fitted statistic is byte-identical afterwards.

**Findings:**
- **Predicted exception counts on the unlabelled panel match the injection ledger exactly**
  -- 701 Balance Discrepancy, 468 Impossible State Transition, 0 Time Travel, 405 Zombie
  Loan, zero error on all four. Reported as an end-to-end wiring check rather than a
  performance claim: each class carries a near-deterministic fingerprint because it was
  injected.

**Verification performed:** 13 new regression tests (108 across the project). Full
`python main.py` run in 4m28s producing all ten deliverables plus the submission.

**Human review:** Every module read line by line. Both `exception_required` failures were
found by comparing the submission's own summary statistics against the known base rate --
not by reading code, and not by any test, because both produced a perfectly valid file.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI, 100%
human-reviewed.

---

## Session 12 — interactive dashboard

**Date:** 2026-08-30

**Goal:** A UI for the demo. Asked for a static page first, then redirected to Streamlit
mid-build.

**Accepted:**
- **Navigation that *is* the demo flow.** Section 14 specifies the order a demo should
  follow; the sidebar is that order, so the app can be walked top to bottom on camera
  without deciding what comes next. A test pins the ordering so a reshuffle is deliberate.
- **Nothing recomputed in the app.** Every number is read from a file the pipeline wrote. A
  dashboard that recalculates its own figures can disagree with the report beside it, and
  the version a viewer trusts is whichever they saw last.
- **A missing phase names the command to run.** An empty panel with no explanation sends the
  user to read the code.
- **The figures' own palette, applied to the page**, with a test asserting they match. The
  app embeds those PNGs directly; a page in a different colour language reads as two
  products stapled together.

**Rejected / corrected:**
- *Two Streamlit 1.62 API breaks, both in shared helpers.* `height=None` is now rejected
  outright, and `use_container_width` is deprecated in favour of `width="stretch"`. Ten of
  twelve pages raised. Found by running every page through `streamlit.testing.v1.AppTest`
  rather than by clicking through the browser -- the app served HTTP 200 the whole time,
  because a Streamlit page renders its exception as a red box in the body.
- *The explorer showed `12-month default: 0.0%` on already-defaulted loans* with no
  explanation. That zero is a deterministic override -- a loan in Default cannot newly
  default -- not a confident prediction, and 565 of the 1,520 rows predicted to move to
  Default are in that state. A reviewer reading it as a model output would draw the opposite
  conclusion. The tile now names the override and the observed status sits beside it.
- *`AppTest.from_file("app.py")` resolves against the test file's directory*, not the
  working directory, so every dashboard test failed with FileNotFoundError until the path
  was made absolute.
- *The navigation test asserted on raw page names* while the widget reports its formatted
  labels, because the sidebar uses a `format_func`.

**Verification performed:** 17 new tests (125 across the project), every page rendered
through Streamlit's harness and inspected in a real browser at 1600x1000.

**Human review:** The override-display problem was found by looking at the rendered page and
noticing that two numbers on the same card told contradictory stories -- not by any test,
and not from the code, which was behaving exactly as designed.

**Approximate AI-generated code share this session:** ~90% of lines drafted by AI, 100%
human-reviewed.

---

## Lessons so far

1. **Check the AI's numbers, not just its code.** The substantive corrections so
   far came from looking at outputs that were *plausible-looking but wrong* —
   0.23% clean records, `loan_id` as top drift, a 19.8% default rate — not from
   reading the code.
2. **Ground truth pays for itself.** Injecting known defects turned "the profiler
   runs" into "the profiler finds what is there", which is the only version of
   that claim worth putting in a report. It is also the one property real data
   cannot provide, and the reason synthetic data survived the decision to obtain
   licensed real data.
3. **Guard every schema assumption.** The organiser's real columns remain unknown;
   code that raises on a missing column would need rewriting under time pressure
   on the day the data lands.
4. **Reproduce before fixing.** A suspected missing-label bug turned out to be
   correct by-design behaviour. Editing on suspicion would have broken a working
   partition.
5. **Decline with measurement, not assertion.** The request for multiprocessing
   was answered with a profile showing where the time actually went. "I did not
   do X, and here is the number that says why" is reviewable; "X was unnecessary"
   is not.
6. **Escalate judgement calls; do not resolve them silently.** Where a specified
   parameter produced an unrealistic outcome, the correct action was to implement
   as specified and surface the discrepancy — not to quietly substitute better
   judgement for the human's stated intent.
7. **State the alternatives you discarded.** A decision presented without its
   rejected options cannot be reviewed, only accepted or re-litigated.
8. **A metric can be wrong in a direction that looks plausible.** A multiclass
   log-loss of 8.72 was not a bad model — it was scikit-learn binding probability
   columns to sorted labels while the pipeline held them in severity order. The
   Brier score, computed independently, is what exposed the contradiction. Two
   metrics that disagree are worth more than one metric that looks fine.
9. **A flat line at the end of a curve is usually missing data, not a finding.**
   Every survival curve in the first draft ran to month 59 regardless of how much
   follow-up the cohort actually had. Nothing errored, nothing looked broken, and the
   chart said the opposite of the truth. Curves now end where the risk set does.
10. **Audit the deliverables against the rubric, not the code against the plan.**
    Three phases were marked "done" and passing every test while the profiling report
    had no charts, the feature matrix had no dictionary, and nothing was committed.
    None of that is visible from inside the code; all of it is obvious the moment you
    ask what a reader actually receives.
11. **An ablation you did not verify is a decoration.** The "no sequence information"
    model scored as well as the full one, which read as a finding until it turned out the
    aggregate score was carrying the removed signal straight back in. The number was real;
    the label on it was false. Ablations now have tests asserting what they exclude.
12. **Numbers copied by hand from a terminal are numbers from whichever run was on
    screen.** Two figures in a results table came from a 500-loan smoke run rather than
    the full one, and both looked entirely plausible. Report tables are now generated from
    the CSV the pipeline wrote, never transcribed.
13. **A control with false positives is worse than no control.** The grounding check
    blocked three faithful reviewer notes before it caught anything real, each time for a
    formatting reason -- a typographic hyphen, an unhandled date type, the pipeline's own
    prompt text. Reviewers do not tune a noisy flag; they stop reading it, and the one true
    hallucination leaves with the rest.
14. **When the model passes every trap, report that.** Task 7 is graded on failure examples,
    and the temptation to manufacture one was real. What actually failed was the guardrail,
    three times, and saying so is both true and more useful than a staged defeat.
15. **Check a delivered artefact against a number you already know.** Both
    `exception_required` failures -- 13.8% and then 0.00% against a 2.6% base rate --
    produced a structurally perfect file that passed every validation check. Neither was
    findable by reading code or running tests. Comparing the output's own summary statistics
    to the base rate found both in seconds.
16. **HTTP 200 is not "the page works."** The app served fine while ten of its twelve
    pages raised, because Streamlit renders an exception as a red box inside a successful
    response. Rendering every page through the framework's own test harness found in one
    command what clicking around would have found slowly and incompletely.
