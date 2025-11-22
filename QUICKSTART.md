# 🚀 Quick Start - Credit Risk Escalation System

Everything you need to install, train, and run the project.

---

## 📦 Prerequisites
- Python 3.10–3.12
- Git (optional, for cloning)
- macOS/Linux or Windows

---

## ⚙️ Setup

### Option A: One-liner (recommended)
```bash
chmod +x setup.sh
./setup.sh
```
This creates `uom_venv` (or `.venv`) and installs from `requirements.txt`.

### Option B: Manual virtualenv
```bash
python3 -m venv uom_venv
source uom_venv/bin/activate     # Windows: uom_venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## ▶️ Run the App (best for raw data)
The Streamlit app includes the full preprocessing pipeline and supports manual entry, sample data, and CSV upload.
```bash
source uom_venv/bin/activate
streamlit run app.py
```

---

## 🧪 Quick CLI Prediction
There are two CLI modes:

- Raw data: use the Streamlit app (recommended) which handles preprocessing end-to-end.
- Preprocessed data: use the CLI on data shaped like `data/splits/X_test.csv`.

```bash
# Preprocessed batch predictions
source uom_venv/bin/activate
python predict_and_decide.py --input data/splits/X_test.csv --output results/reports/predictions.csv
```

Notes:
- `predict_and_decide.py` expects preprocessed feature columns identical to training.
- For raw CSVs with arbitrary columns, use `streamlit run app.py`.

---

## 🏋️ Train Models (optional)
Trains baseline, bootstrap ensemble for uncertainty, and escalation system. Saves to `results/models/`.
```bash
source uom_venv/bin/activate
python train_and_save.py   --data-path data/raw/LC_loans_granting_model_dataset.csv
```
Artifacts:
- `results/models/preprocessor.pkl`
- `results/models/bootstrap_ensemble.pkl`
- `results/models/escalation_system.pkl`
- `results/reports/` (metrics, summaries)

---

## ✅ Tests & Smoke Checks
```bash
# Unit tests
source uom_venv/bin/activate
pytest -v

# Smoke test (verifies key files)
./smoke_test.sh
```

---

## 📁 Project Structure (essentials)
- `app.py`: Streamlit web UI (handles raw data)
- `train_and_save.py`: End-to-end training pipeline
- `predict_and_decide.py`: CLI predictions on preprocessed data
- `src/`: Preprocessing, uncertainty, and escalation logic
- `data/raw/`: Source dataset(s)
- `data/splits/`: Preprocessed train/val/test matrices
- `results/models/`: Saved models and metadata
- `results/reports/`: CSV/JSON reports

---

## 💡 Common Tasks
```bash
# 1) Start UI for raw data
streamlit run app.py

# 2) Predict on preprocessed features
python predict_and_decide.py --input data/splits/X_test.csv --output results/reports/predictions.csv

# 3) Retrain everything from scratch
python train_and_save.py --data-path data/raw/LC_loans_granting_model_dataset.csv
```

---

## 🛠️ Troubleshooting
- Activate the environment: `source uom_venv/bin/activate` (Windows PowerShell: `uom_venv\Scripts\Activate.ps1`)
- Reinstall deps: `pip install -r requirements.txt`
- Streamlit not found → install: `pip install streamlit`
- If models are missing, run training: `python train_and_save.py`

---

## 📚 References
- Use `README.md` for overview
- Use `PROJECT_GUIDE.md` for deep-dive architecture and methodology

