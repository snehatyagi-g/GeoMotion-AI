# GeoMotion AI – An AI-Based Intelligent Vehicle Movement and Map-Matching System

GeoMotion AI is an interactive Streamlit project for exploring vehicle trajectories, road networks, and map-matching methods. It uses a synthetic benchmark to compare GPS observations with routes inferred from a road network.

## Main features

- Explore benchmark scenarios, trajectories, road networks, and evaluation results.
- Compare noisy GPS observations with network-based matched routes.
- Run and compare three map-matching algorithms:
  - **Nearest Road** — selects the closest road segment for each observation.
  - **HMM/Viterbi** — uses a Hidden Markov Model and Viterbi decoding to infer a likely road sequence.
  - **Rule-Based Adaptive Radius** — adapts its candidate search radius using observation and road-context features.
- Replay a selected synthetic trajectory in the Live Map-Matching Demo.
- Review observation-level predictions and benchmark metrics, with available exports.

## Synthetic benchmark

The dashboard uses the versioned synthetic benchmark under `data/processed/synthetic_v2/`. It contains generated road scenarios, noisy observations, features, ground-truth labels, and adaptive-radius labels for train, validation, and test splits. Trajectories are sampled at 1, 5, 15, 30, 45, 60, and 120 second intervals.

The benchmark is designed for controlled algorithm comparison. Its scenarios, noise, and labels are generated from stated assumptions and are not independent real-world measurements.

## Installation

Use Python 3.12 or another Python version compatible with the pinned dependencies. From the project root, create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the dashboard

From the project root, run:

```bash
streamlit run app.py
```

## Project structure

```text
GeoMotion-AI/
├── app.py                         # Streamlit dashboard entrypoint
├── requirements.txt               # Pinned runtime dependencies
├── src/                            # Matching, road-network, and evaluation modules
├── data/
│   └── processed/synthetic_v2/     # Dashboard benchmark data
├── results/
│   ├── python_metrics/             # Python benchmark metrics
│   └── python_figures/             # Dashboard result figures
├── tests/                          # Python and JavaScript tests
└── .streamlit/
    └── config.toml                 # Streamlit theme and app configuration
```

## Results disclaimer

Results on this synthetic benchmark do **not** establish real-world performance, accuracy, or suitability for live vehicle navigation. The benchmark is a controlled development and comparison tool; real-world claims require independent validation using appropriately sourced data.
