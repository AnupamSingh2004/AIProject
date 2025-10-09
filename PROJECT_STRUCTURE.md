# Project Structure
src/
├── skin_tone_analyzer.py
├── clothing_detector.py
├── color_analyzer.py
├── recommendation_engine.py
├── outfit_scorer.py
└── utils.py

data/
├── raw/                 # Original Kaggle dataset
├── processed/          # Processed data
└── user_uploads/       # User uploaded images

models/
└── saved_models/       # Saved model weights

tests/
└── test_*.py          # Unit tests

app/
└── streamlit_app.py   # Demo application

notebooks/
└── experiments/       # Jupyter notebooks for experiments
