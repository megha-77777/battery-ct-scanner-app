# 🔋 Battery CT Scanner App

Streamlit-based app simulating an IR + AI battery defect scanner.

## Features:
- Upload simulated IR image
- AI predicts: Healthy / Bulging / Cracked

## How to Run Locally:
```bash
pip install -r requirements.txt
python placeholder_model.py  # Creates battery_model.h5
streamlit run app.py
