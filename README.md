# BAH — Air-Gapped Predictive NOC Copilot

## Requirements
- Python 3.10+
- [Ollama](https://ollama.com) installed, with `llama3.2` model pulled

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install streamlit networkx matplotlib joblib pandas numpy tensorflow ollama
```

## Start Ollama (required for the copilot's remediation feature)

```bash
ollama pull llama3.2
ollama serve
```

## Run the app

```bash
streamlit run app.py
```

Model files (`network_model.keras`, `model_features.pkl`, `model_threshold.pkl`) are already in the repo, so no training step is needed.
