# Phishing URL Detector

A machine learning tool that classifies URLs as phishing or legitimate using
classical ML models trained on a public dataset. Built as a learning project
combining AI/ML and cybersecurity concepts.

> **Status:** In development. Day 1 of 7 — environment setup complete.

## Why

This project ties together two areas I'm interested in: machine learning and
cybersecurity. Phishing is one of the most common attack vectors, and detecting
malicious URLs is a well-defined supervised learning problem that's a good
testbed for learning classical ML end-to-end.

## Planned features

- [x] Project scaffolding and environment setup
- [ ] Exploratory data analysis on the PhiUSIIL phishing URL dataset
- [ ] Logistic Regression and Random Forest classifiers
- [ ] Model comparison with precision, recall, F1, and confusion matrices
- [ ] CLI tool for classifying URLs with confidence scores
- [ ] Feature importance analysis to explain predictions
- [ ] Pytest test suite

## Tech stack

Python 3.11, Pandas, Scikit-learn, Matplotlib, Seaborn, Typer, Joblib, pytest

## Dataset

[PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
from the UCI Machine Learning Repository — ~235K URLs with 54 pre-extracted
features. The dataset is not included in this repo; download it separately
into the `data/` directory.

## Quickstart

```bash
# Clone and set up
git clone https://github.com/cdowning2022/phishing-url-detector.git
cd phishing-url-detector
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download the dataset into data/ (see Dataset section above)

# Run the exploration notebook
jupyter notebook notebooks/01_exploration.ipynb
```

## Project structure

```
phishing-url-detector/
├── data/             # Dataset (not committed)
├── notebooks/        # Exploratory analysis
├── src/              # Source code
├── models/           # Trained models (not committed)
├── tests/            # Pytest tests
└── docs/             # Methodology writeup
```

## Results

*Coming after Day 4.*

## What I learned

*Coming after Day 7.*

## Ethics

This tool is for educational and defensive research purposes only. It is not
intended to be used against live infrastructure or for any unauthorized
activity.

## Author

Cole Downing — [GitHub](https://github.com/cdowning2022) · [LinkedIn](https://www.linkedin.com/in/cole-downing-991309218)
