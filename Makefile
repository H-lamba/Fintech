# Loan Performance Intelligence Engine -- reproducible pipeline.
#
#   make setup     create the venv and install dependencies
#   make data      generate the synthetic benchmark data pack
#   make all       run every phase end to end, from raw CSVs to reports
#   make test      run the regression suite
#
# Every target is idempotent: re-running regenerates its outputs from the CSVs
# in data/ and nothing else. There are no manual steps between them.

PY := .venv/bin/python
LOANS ?= 10000
SAMPLE ?=

.PHONY: all setup data profile predict survival anomaly scenario test clean-reports help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install pinned dependencies
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data: ## Generate the synthetic benchmark data pack into data/
	$(PY) scripts/generate_synthetic_suite.py --loans $(LOANS)

profile: ## Phase 1 -- data intelligence report (Task 1)
	$(PY) scripts/run_profiling.py $(if $(SAMPLE),--sample $(SAMPLE),)

predict: ## Phases 2-3 -- feature engineering + loan performance prediction (Task 2)
	$(PY) scripts/run_prediction.py $(if $(SAMPLE),--sample $(SAMPLE),) --score-test

survival: ## Phase 4 -- survival / competing-risk modelling (Task 3)
	$(PY) scripts/run_survival.py $(if $(SAMPLE),--sample $(SAMPLE),)

anomaly: ## Phase 5 -- anomaly & exception detection (Task 4)
	$(PY) scripts/run_anomaly.py $(if $(SAMPLE),--sample $(SAMPLE),)

scenario: ## Phase 6 -- scenario & stress simulation (Task 5); needs `make predict` first
	$(PY) scripts/run_scenario.py $(if $(SAMPLE),--sample $(SAMPLE),)

test: ## Run the regression suite
	$(PY) -m pytest tests/ -q

# `predict` runs before the second `profile` so the feature dictionary it emits
# is folded into the Data Intelligence Report. The first pass is what produces
# the data-quality scores the feature matrix consumes -- the two reports are
# mutually referential by design, and this is the order that resolves it.
all: profile predict survival anomaly scenario profile test ## Run every phase end to end
	@echo ""
	@echo "Pipeline complete. Deliverables:"
	@echo "  reports/data_intelligence_report.md   (Task 1)"
	@echo "  reports/task2_model_results.md        (Task 2)"
	@echo "  reports/survival_report.md            (Task 3)"
	@echo "  reports/anomaly_report.md             (Task 4)"
	@echo "  reports/anomaly_examples.csv          (Task 4, reviewer queue)"
	@echo "  reports/scenario_report.md            (Task 5)"
	@echo "  reports/feature_dictionary.md"

clean-reports: ## Delete every generated report (data/ is untouched)
	rm -rf reports/ models/
