```markdown
# Adaptive Speculative Decoding for CPU‑Constrained GGUF Models

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production‑ready framework that accelerates large language model inference on CPU‑limited systems using **adaptive speculative decoding**. It dynamically adjusts draft depth, selects the optimal policy, and monitors runtime metrics to maximise throughput while minimising latency variance and wasted compute—even when absolute speedup is modest.

---

## 🚀 Key Features

- **Real & Simulation Backends** – Run with `llama-cpp-python` (GGUF), [OpenRouter](https://openrouter.ai/) API, or a fully synthetic simulation (for testing without hardware).
- **Adaptive Draft Control** – Adjusts speculation depth based on CPU load, acceptance rate, entropy, context length, and even ITL coefficient of variation.
- **Multiple Policy Engines** – Choose from *Heuristic*, *Bandit*, *EMA*, or *Ensemble* to decide the draft depth.
- **KV Cache Coordination** – Quantises draft KV entries (INT8) for memory efficiency, with hit/miss tracking and rollback support.
- **Rich Runtime Monitoring** – Tracks TPS, inter‑token latency (ITL), CPU utilisation, memory bandwidth, entropy, and more.
- **Publication‑Ready Visualisations** – Generates PNG plots: efficiency dashboard, overview, time‑series, controller analysis, policy analysis, KV stats, and phase breakdown.
- **Scientific Validation** – Provides sanity checks, baseline TPS measurement, and confidence in speculative efficiency metrics.

---

## 🏗️ Architecture Overview

The system is modular, with five core components:

| Module | Responsibility |
|--------|----------------|
| `monitor.py` | Runtime monitoring: collects snapshots of TPS, latency, CPU, entropy, ITL variance, etc. |
| `draft_controller.py` | Adaptive draft depth controller with oscillation detection and safety rules. |
| `policy_engine.py` | Policy selection (Heuristic / Bandit / EMA / Ensemble) and workload detection. |
| `kv_cache.py` | KV cache coordination with INT8 quantisation, shadow buffer, and hit/miss tracking. |
| `main.py` | Main inference engine, command‑line interface, benchmarking, and plotting. |
| `visualizer.py` | Generates multi‑panel PNG plots from benchmark runs. |

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- (Optional) A GGUF model file (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf)) and `llama-cpp-python`
- (Optional) An OpenRouter API key

### Steps
```bash
# Clone the repository
git clone https://github.com/sadrasa97/adaptive-speculative-decoding.git
cd adaptive-speculative-decoding

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install llama-cpp-python with CPU optimisations
pip install llama-cpp-python
```

### `requirements.txt` example
```
numpy>=1.24.0
psutil>=5.9.0
matplotlib>=3.6.0
python-dotenv>=1.0.0
```

> **Note**: `llama-cpp-python` is not required if you use simulation or OpenRouter.

---

## 🧪 Usage

### Command‑Line Interface

```bash
python main.py [options]
```

**Basic examples**:
- Run a single prompt with real GGUF models:
  ```bash
  python main.py --model /path/to/target.gguf --draft-model /path/to/draft.gguf --prompt "Write a Python quicksort" --max-tokens 200
  ```
- Run benchmark on default prompts (simulation if no model):
  ```bash
  python main.py --benchmark
  ```
- Interactive chat:
  ```bash
  python main.py --interactive
  ```
- Use OpenRouter backend:
  ```bash
  python main.py --backend openrouter --openrouter-api-key $OPENROUTER_API_KEY --openrouter-model openai/gpt-4o-mini --prompt "Explain quantum computing"
  ```

### Key Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Path to target GGUF model. |
| `--draft-model` | Path to draft GGUF model. |
| `--prompt` | Input prompt (if not using interactive/benchmark). |
| `--max-tokens` | Maximum tokens to generate (default 256). |
| `--temperature` | Sampling temperature (default 0.7). |
| `--depth` | Initial draft depth (default 4). |
| `--n-ctx` | Context size (default 4096). |
| `--threads` | Number of CPU threads (0 = auto). |
| `--policy` | Policy type: `heuristic`, `bandit`, `ema`, `ensemble` (default: `ensemble`). |
| `--backend` | `auto`, `gguf`, `openrouter`, `simulation`. |
| `--benchmark` | Run benchmark on a set of prompts. |
| `--interactive` | Start interactive chat session. |
| `--plot-dir` | Directory to save PNG plots (default `plots`). |

See `python main.py --help` for all options.

---

## 📊 Output & Visualisations

When running benchmarks, the system:
1. Prints a **diagnostic sanity check** to verify correctness of speculative metrics.
2. Displays a **comprehensive evaluation** including throughput, latency percentiles, efficiency, and phase breakdown.
3. Saves **PNG plots** in the specified plot directory:
   - `00_efficiency_dashboard.png` – primary figure highlighting wasted compute and ITL CV.
   - `01_overview_dashboard.png` – TPS, speedup, acceptance, latency, efficiency, draft token budget.
   - `02_timeseries_run*.png` – per‑step TPS, ITL, acceptance, depth, CPU, entropy.
   - `03_controller_analysis.png` – decision reasons and depth distribution.
   - `04_policy_analysis.png` – EMA rewards, policy contribution, reward/regret.
   - `05_kv_cache.png` – hit rate, rollbacks, compression, memory pressure.
   - `06_phase_latency.png` – phase breakdown and latency percentiles.

If both GGUF and simulation runs exist, `07_backend_comparison.png` is also generated.

---

## ⚙️ How It Works (Brief)

1. **Baseline TPS Measurement** – The engine first measures the target model’s throughput without speculation to establish a reference.
2. **Draft Generation** – The draft model proposes a sequence of `depth` tokens.
3. **Verification** – The target model processes the draft tokens one by one; if a token matches, it is accepted; otherwise, the first mismatch is corrected.
4. **KV Cache** – Draft KV entries are quantised and stored; accepted tokens are synced, rejected tokens are rolled back.
5. **Monitoring** – Every step records TPS, ITL, CPU, entropy, acceptance ratio, ITL variance, and more.
6. **Adaptive Control** – The draft controller uses EMA‑filtered metrics to adjust depth and mode (disabled/conservative/normal/aggressive) based on safety rules.
7. **Policy Engine** – The selected policy (heuristic, bandit, EMA, or ensemble) suggests a depth, which the controller may adopt if confidence is high.
8. **Workload Detection** – The policy engine classifies the prompt (coding, reasoning, chat, creative) and adapts accordingly.

---

## 🧠 Design Philosophy

- **Memory‑Constraint First** – The system is tuned for CPU‑limited environments where memory bandwidth and latency variance matter more than absolute token throughput.
- **Scientific Rigour** – All metrics are validated with sanity checks; the simulation model is calibrated against real‑world behaviour.
- **Modularity** – Each component can be swapped or tuned independently.
- **Extensibility** – The code supports custom policies, backends, and visualisations.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, discuss them first.

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 📚 References

- [LLM Inference on CPU](https://github.com/ggerganov/llama.cpp) – GGUF format and `llama-cpp-python`.
- OpenRouter – unified API for many LLMs.

---



Built with ❤️ for the open‑source AI community.
```
