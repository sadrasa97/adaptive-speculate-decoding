# Adaptive Speculative Decoding for CPU-Constrained GGUF Models

AdaptiveSD is an experimental framework for evaluating adaptive speculative decoding under CPU and memory-bandwidth constraints.

It includes:
- A runtime monitor for TPS, latency, acceptance, CPU, and stability metrics.
- An adaptive draft-depth controller with safety guards and oscillation suppression.
- Multiple policy modes (AdaptiveSD variants plus baseline approximations).
- KV-cache coordination with draft quantization and rollback accounting.
- Benchmarking utilities, comparison tables, and publication-ready plots.

Repository: https://github.com/sadrasa97/adaptive-speculate-decoding/

## Why this project

Most speculative decoding write-ups focus on throughput gains. This project additionally emphasizes efficiency and stability under constrained CPU environments:
- Wasted compute fraction (WCF).
- Inter-token latency variability (ITL CV).
- Tail latency behavior.
- AR-fallback contamination tracking.

This makes it easier to report realistic behavior when speedup is sub-unity on a given backend.

## Project structure

- main entrypoint: `main.py`
- monitoring and metrics: `monitor.py`
- adaptive control logic: `draft_controller.py`
- policy engine and baselines: `policy_engine.py`
- KV cache coordination: `kv_cache.py`
- plotting and export utilities: `visualizer.py`
- hardware spec collector (Windows): `collect_hw_specs.ps1`
- generated outputs: `plots/`

## Features

- Backends:
  - `gguf` via `llama-cpp-python`.
  - `openrouter` via HTTP API.
  - `simulation` for controlled experiments without local model files.
- Policies:
  - `ensemble`, `bandit`, `ema`, `heuristic`
  - `fixed_depth`, `parallel_sd`, `dynamic_lookahead`, `specdec_plus_approx`
- Experiment modes:
  - single prompt generation
  - benchmark mode with repeats and CI reporting
  - cross-method comparison mode with table/plot export
  - interactive shell mode

## Requirements

- Python 3.10+
- Windows, Linux, or macOS (this repo currently includes a Windows hardware script)
- Python packages:
  - numpy
  - psutil
  - python-dotenv
  - matplotlib
  - llama-cpp-python (optional, required for GGUF backend)

Install example:

```bash
pip install numpy psutil python-dotenv matplotlib llama-cpp-python
```

If you only want simulation mode:

```bash
pip install numpy psutil python-dotenv matplotlib
```

## Quick start

Clone and enter the repo:

```bash
git clone https://github.com/sadrasa97/adaptive-speculate-decoding/
cd adaptive-speculate-decoding
```

### 1. Single prompt run

```bash
python main.py --backend simulation --prompt "Explain adaptive speculative decoding on CPUs." --max-tokens 128
```

### 2. Benchmark run (recommended repeats for CI)

```bash
python main.py --backend simulation --benchmark --repeats 5 --plot-dir plots
```

### 3. Compare methods on same prompts/backend

```bash
python main.py --backend simulation --compare-methods --repeats 5 --plot-dir plots
```

### 4. Long-context comparison

```bash
python main.py --backend simulation --compare-methods --long-context --repeats 5 --plot-dir plots
```

### 5. Interactive mode

```bash
python main.py --backend simulation --interactive
```

Type `status` in interactive mode to print runtime status JSON.

## Using GGUF backend

By default, `main.py` points to local Windows paths for model and draft model. Override them on your machine:

```bash
python main.py \
  --backend gguf \
  --model /path/to/target.gguf \
  --draft-model /path/to/draft.gguf \
  --benchmark --repeats 5
```

Notes:
- If `llama-cpp-python` is not installed or model path is invalid, backend may fall back to simulation.
- If draft model cannot be loaded, runs continue in non-speculative autoregressive path.

## Using OpenRouter backend

Set environment variables (or pass CLI flags):

```bash
export OPENROUTER_API_KEY=your_key
export OPENROUTER_MODEL=openrouter/free
python main.py --backend openrouter --benchmark --repeats 5
```

PowerShell example:

```powershell
$env:OPENROUTER_API_KEY="your_key"
$env:OPENROUTER_MODEL="openrouter/free"
python main.py --backend openrouter --benchmark --repeats 5
```

## CLI reference

Key arguments from `main.py`:

- `--backend {auto,gguf,openrouter,simulation}`
- `--policy {heuristic,bandit,ema,ensemble,fixed_depth,parallel_sd,dynamic_lookahead,specdec_plus_approx}`
- `--benchmark`
- `--compare-methods`
- `--long-context`
- `--interactive`
- `--prompt <text>`
- `--max-tokens <int>`
- `--repeats <int>` (use 5+ for more defensible CI)
- `--plot-dir <path>`
- `--model <path>` and `--draft-model <path>`
- OpenRouter options: `--openrouter-api-key`, `--openrouter-model`, `--openrouter-base-url`, `--openrouter-site-url`, `--openrouter-site-name`

## Outputs

Typical outputs include:

- per-method benchmark plots under `plots/<method>/`
- cross-method comparison assets under `plots/_comparison/`
- markdown/latex comparison table exports
- `comparison_summary.json` for downstream reporting
- run log file (default: `adaptive_sd.log`)

## Reproducibility

### Capture machine specs

On Windows PowerShell:

```powershell
.\collect_hw_specs.ps1
```

This generates a timestamped `system_specs_YYYYMMDD_HHMMSS.txt` report.

### Validity practices already implemented in this codebase

- Runtime modules are reset between benchmark prompts to avoid cumulative contamination.
- Non-speculative paths report speculative metrics as unavailable rather than fake placeholders.
- Baseline source and sample sufficiency are surfaced in outputs.
- Sensitive CLI args such as API keys are redacted in logs.

## Example snapshot (current repository artifacts)

From `plots/_comparison/comparison_summary.json` (GGUF, repeats=5, 15 runs per method):

- `specdec_plus_approx`: mean TPS 3.78, speedup 0.771x, AR-fallback 13.7%
- `fixed_depth`: mean TPS 3.25, speedup 0.664x, AR-fallback 46.3%
- `ensemble`: mean TPS 3.20, speedup 0.653x, AR-fallback 37.3%

Interpretation: on this specific hardware/backend setup, all listed methods are below 1.0x speedup versus no-speculation baseline; the framework still helps compare stability, fallback contamination, and compute-efficiency tradeoffs.

## Known limitations

- Baseline methods `specdec_plus_approx` and `dynamic_lookahead` are rule-based approximations, not full reproductions of original trained systems.
- Welch t-test p-values use SciPy if available; otherwise the code falls back to a normal approximation.
- Results are backend- and hardware-sensitive; avoid generalizing from a single machine.

## Suggested workflow for a new experiment

1. Capture hardware snapshot using `collect_hw_specs.ps1`.
2. Run `--compare-methods --repeats 5` on one backend.
3. Save `plots/_comparison/*` artifacts.
4. Re-run on another backend (`gguf` vs `simulation` or `openrouter`) for contrast.
5. Report both speed and efficiency metrics (WCF, ITL CV, AR-fallback).

## License

No license file is currently present in this repository. Add one if you plan public reuse.
