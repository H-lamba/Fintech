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
git clone https://github.com/H-lamba/Fintech.git && cd Fintech

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
git clone https://github.com/H-lamba/Fintech.git && cd Fintech

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
git clone https://github.com/H-lamba/Fintech.git; cd Fintech

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

---

## Configuring `.env` (optional — for the Phase 8 copilot)

**You can skip this entirely.** Without a key the copilot runs in **offline mode**: it
exercises the full context assembly, guardrail and audit-logging path with deterministic
stubs, each marked `provider: offline-stub` in the log and never presented as a real model
response. Every other phase is unaffected — the LLM is downstream of all of them.

Set a key only if you want the copilot to make **real** calls.

### 1. Create the file

```bash
cp .env.example .env          # macOS / Linux
```
```powershell
Copy-Item .env.example .env   # Windows
```

### 2. Add a key

Open `.env` and replace the placeholder on the `LLM_API_KEY` line:

```ini
LLM_API_KEY=gsk_your_actual_key_here
```

**Any OpenAI-compatible provider works, and you do not configure which one.** The client
detects the provider from the key's *prefix*, so the key is usually the only line you need
to touch:

| Key prefix | Provider | Default model | Get a key |
| :--- | :--- | :--- | :--- |
| `gsk_…` | Groq | `openai/gpt-oss-120b` | [console.groq.com/keys](https://console.groq.com/keys) |
| `xai-…` | xAI | `grok-4` | [console.x.ai](https://console.x.ai) |
| `sk-…` | OpenAI | `gpt-4o-mini` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

Prefix detection is not cosmetic. A Groq key pointed at xAI's endpoint returns
*"Incorrect API key provided"*, which reads as a bad key rather than a wrong endpoint, and
cost a debugging cycle before this was added.

`XAI_API_KEY`, `GROQ_API_KEY` and `OPENAI_API_KEY` are all accepted as aliases for
`LLM_API_KEY`. **The variable's name does not choose the provider** — the key's prefix
does — so a Groq key stored under `XAI_API_KEY` still routes to Groq.

### 3. Optional overrides

Uncomment these only to override what the key prefix implies:

```ini
# LLM_BASE_URL=https://api.groq.com/openai/v1
# LLM_MODEL=openai/gpt-oss-120b
LLM_LOG_PATH=reports/llm_prompt_log.jsonl   # where the audit trail is written
```

### 4. Verify it worked

```bash
python scripts/run_copilot.py --live --notes 1
```

Expect a first line reading:

```
Mode: LIVE against https://api.groq.com/openai/v1 (openai/gpt-oss-120b)
  key from .env:LLM_API_KEY -> provider detected: groq
```

If it says `Mode: OFFLINE (no API calls)`, the key was not picked up — check the line has
no quotes, no spaces around `=`, and is not still the placeholder.

To run the whole pipeline with a live copilot:

```bash
python main.py --live-copilot
```

**Offline stays the default for `python main.py` on purpose.** A default entry point must
not send loan data to a third party or spend an account's credits because someone ran it
without reading the flags.

### Security

- `.env` is **gitignored and has never been committed** — keep it that way.
- Commit `.env.example` (the template), never `.env` (the key).
- The dashboard shows only *live / offline* status; it never renders the key or the model
  name, and a test asserts it.
- A live run sends the selected loan's record and the models' outputs to the provider. The
  data here is synthetic, but check your provider's retention policy before pointing this
  at anything real.

---

## Deploying the dashboard to Streamlit Community Cloud

The app is deployable **because it computes nothing**. Every page reads a file the pipeline
already wrote, and `reports/`, `submission/` and `data/` are committed — so the hosted app
serves the same numbers as the repo without ever running a phase.

### 1. Push everything first

Cloud builds from the branch, not your working tree.

```bash
git add -A && git commit -m "Docs, dashboard and deploy prep" && git push
```

### 2. Create the app

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub.**
3. Fill in:
   - **Repository:** `H-lamba/Fintech`
   - **Branch:** whichever you pushed (`UI` unless you merge to `main` first)
   - **Main file path:** `app.py`
4. Open **Advanced settings** and set **Python version to 3.11**. This matters: the pinned
   `scikit-survival` and `lightgbm` versions have prebuilt Linux wheels for 3.11, and on a
   newer interpreter pip may try to compile them and fail the build.

### 3. Add the API key as a secret

**This does not happen automatically.** `.env` is gitignored, so it will not exist on
Cloud and the copilot will serve offline stubs until you paste the key in.

In **Advanced settings → Secrets** (or later under **⋮ → Settings → Secrets**), paste:

```toml
LLM_API_KEY = "gsk_your_actual_key_here"
```

**Keep it at the top level — do not put it under a `[section]` header.** Streamlit exports
top-level secrets as environment variables but not nested ones, and
`src/copilot/llm_client.py` reads the environment. `app.py` bridges `st.secrets` into
`os.environ` at startup as a second guarantee, and that bridge only reads top-level keys
too.

`XAI_API_KEY`, `GROQ_API_KEY` and `OPENAI_API_KEY` work as aliases here exactly as they do
in `.env`. No code change is needed either way.

**Without a key the app still works** — the copilot falls back to deterministic stubs, the
badge on the Loan explorer reads *offline stub* instead of *live*, and a caveat on the page
says so. Nothing else on the dashboard is affected.

To confirm it took after deploying: open **Loan explorer** and check the badge beside
*Analyse with AI* reads **live**.

> **Never commit a local `.streamlit/secrets.toml`.** `.streamlit/config.toml` *is* tracked,
> so the folder is not ignored as a whole — the secrets file is ignored by name.

### 4. Deploy

**This app is deployed at [fintech-cqm5kbt2xwkx3krp7gacjb.streamlit.app](https://fintech-cqm5kbt2xwkx3krp7gacjb.streamlit.app).**

First build takes several minutes — the repo is ~190 MB and `requirements.txt` installs
the full pipeline stack, not just the dashboard's five libraries.

### Known constraints

| Issue | Detail |
| :--- | :--- |
| **Build time** | `requirements.txt` is the *pipeline's* dependency set. The dashboard only needs `streamlit`, `pandas`, `numpy`, `matplotlib` and `markdown`; the rest install for nothing. If the build times out or fails on `scikit-survival` / `pyod`, that is the thing to trim. |
| **Repo size** | ~190 MB tracked, mostly `data/loan_monthly_performance_train.csv` at 68 MB. Within limits, but it slows every rebuild. The dashboard never reads the train panel — only the test panel and `reports/`. |
| **Resources** | Community Cloud gives ~1 GB RAM. The app loads `submission.csv` (16 MB) and the test panel (18 MB) lazily and caches them; that fits, but do not add a page that loads the train panel. |
| **Ephemeral disk** | `publish_report()` writes self-contained report copies into `static/` at runtime. That works on Cloud and simply regenerates after a restart. |
| **Sleeping** | Free apps sleep after inactivity and cold-start in ~30 s. Wake it *before* a demo or a judging session. |

### If the build fails

The failure will almost certainly be pip, not the app. Read the build log and check which
package broke:

- **`scikit-survival` or `pyod`** — the usual suspects. Confirm Python 3.11 is selected.
- **`lightgbm` / `xgboost` import errors** — would need `libgomp1` in a `packages.txt`, but
  the dashboard never imports them, so this should not arise.

The fastest fix is a dashboard-only requirements file, at the cost of the repo no longer
having one dependency list. Only do that if the full set genuinely will not build.

---

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

