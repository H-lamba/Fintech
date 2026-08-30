# Setup & running

> **Docs:** [Setup](setup.md) · [Architecture](architecture.md) · [Pipeline](pipeline.md) · [Module reference](api.md) · [Data](data.md) · [Outputs](outputs.md) · [Dashboard](dashboard.md) · [Results](results.md) · [Design decisions](design-decisions.md) · [Testing](testing.md)
>
> [← Back to the README](../README.md)

---

Python **3.11 or newer** is required. Everything else is pinned in `requirements.txt`.

<details open>
<summary><b>macOS</b></summary>

```bash
# 1. Prerequisites (LightGBM and XGBoost link against OpenMP)
brew install libomp

# 2. Clone and enter
git clone <your-fork-url> Fintech && cd Fintech

# 3. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: the Phase 8 copilot. Without a key it runs offline stubs.
cp .env.example .env        # then paste your key into it

# 5. Run everything
python scripts/generate_synthetic_suite.py --loans 10000
python main.py

# 6. Explore
streamlit run app.py
```
</details>

<details open>
<summary><b>Linux (Debian / Ubuntu)</b></summary>

```bash
# 1. Prerequisites
sudo apt-get update && sudo apt-get install -y python3-venv libgomp1

# 2. Clone and enter
git clone <your-fork-url> Fintech && cd Fintech

# 3. Environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Optional: the Phase 8 copilot
cp .env.example .env

# 5. Run everything
python scripts/generate_synthetic_suite.py --loans 10000
python main.py

# 6. Explore
streamlit run app.py
```
</details>

<details open>
<summary><b>Windows (PowerShell)</b></summary>

No OpenMP install is needed — the LightGBM and XGBoost wheels ship it on Windows.

```powershell
# 1. Clone and enter
git clone <your-fork-url> Fintech; cd Fintech

# 2. Environment  (note: Scripts\, not bin/)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Optional: the Phase 8 copilot
Copy-Item .env.example .env

# 4. Run everything
python scripts\generate_synthetic_suite.py --loans 10000
python main.py

# 5. Explore
streamlit run app.py
```

If `Activate.ps1` is blocked, run PowerShell once as
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use
`.\.venv\Scripts\activate.bat` from `cmd.exe`.

**`make` is not available on Windows** unless you have installed it. Every `make` target
below has a plain-Python equivalent — use `python main.py` and the `python scripts\*.py`
commands directly.
</details>

### The `make` shortcuts (macOS / Linux)

```bash
make setup                     # venv + pinned dependencies
make data                      # generate the synthetic benchmark pack
make all                       # every phase -> submission.csv
make test                      # the regression suite
make ui                        # launch the dashboard

make profile                   # Phase 1  -- data intelligence report (Task 1)
make predict                   # Phases 2-3 -- features + prediction (Task 2)
make survival                  # Phase 4  -- survival / competing risks (Task 3)
make anomaly                   # Phase 5  -- anomaly & exception detection (Task 4)
make scenario                  # Phase 6  -- scenario & stress simulation (Task 5)
make explain                   # Phase 7  -- explainability & model card (Task 6)
make copilot                   # Phase 8  -- LLM reviewer copilot, offline (Task 7)
make copilot-live              # Phase 8  -- LIVE; spends API credits

make all SAMPLE=2000           # every phase on 2,000 loans, for fast iteration
make data LOANS=10000          # smaller data pack
```

### Entry points

```bash
python main.py                 # every phase, ending in submission/submission.csv
python main.py --live-copilot  # same, but Phase 8 calls the real LLM provider
python main.py --sample 2000   # fast end-to-end pass on 2,000 loans
python main.py --submission    # inference and submission only (needs trained models)
python main.py --model-card    # regenerate the model card from existing reports
```

Or call the scripts directly for their full option sets (`--help` on any of them):

```bash
python scripts/run_profiling.py --sample 250000
python scripts/run_prediction.py --backend xgboost --score-test
python scripts/run_survival.py --no-left-truncation
```

### Troubleshooting

| Symptom | Cause and fix |
| :--- | :--- |
| `OSError: libgomp.so.1: cannot open shared object file` | LightGBM needs OpenMP. `sudo apt-get install libgomp1` (Linux) or `brew install libomp` (macOS). |
| `UnicodeEncodeError: 'charmap' codec` on Windows | An older checkout. Every file write in this repo pins `encoding="utf-8"`; pull the latest. |
| `No reviewer queue at reports/anomaly_examples.csv` | Phase 8 needs Phase 5. Run `python main.py`, or `python scripts/run_anomaly.py` first. |
| Phase 8 says `Mode: OFFLINE` | No key in `.env`. That is the safe default — it runs deterministic stubs and marks them as such. |
| `make: command not found` (Windows) | Expected. Use the `python` commands above. |
| The dashboard shows "has not been run yet" | That phase has no output on disk. The page names the command to run. |

---

