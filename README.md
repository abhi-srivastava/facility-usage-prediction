# Facility Usage Prediction System

A working prototype that predicts, for a residential community's shared
facilities (gym, pool, courts, clubhouse, etc.), a resident's **next
booking**: which facility, which day of week, which hour, and when to send
a booking reminder ("nudge"). Built for the Anacity/Anarock Facility Usage
Prediction System brief.

See [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) for
the full write-up of data, features, modelling choices, results, and
limitations. This README covers setup and how to run it.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
```

That single command:
1. Generates the synthetic dataset (`data/bookings.csv`) if it doesn't exist yet.
2. Trains the four prediction models on a chronological training split (`src/train.py`).
3. Evaluates them on the held-out, chronologically-later test set and writes
   the prediction-review spreadsheet (`src/evaluate.py`).
4. Runs a forward-looking prediction for every resident's next booking, as of
   "today" (`src/predict.py`).

Takes about a minute on a laptop. Each stage can also be run on its own:

```bash
python -m src.generate_dataset   # regenerate data/bookings.csv from scratch
python -m src.train               # train + persist models/*.joblib
python -m src.evaluate            # score the held-out test set
python -m src.predict             # forward-looking next-booking predictions
```

## Deliverables in this repo

| Item | Where |
|---|---|
| Synthetic dataset | [`data/bookings.csv`](data/bookings.csv) |
| Technical documentation | [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) |
| Prediction review output (predicted vs. actual, per booking) | [`outputs/predictions_review.csv`](outputs/predictions_review.csv), [`outputs/predictions_review.xlsx`](outputs/predictions_review.xlsx) |
| Evaluation metrics | [`outputs/metrics.json`](outputs/metrics.json) |
| Forward next-booking predictions | [`outputs/next_booking_predictions.csv`](outputs/next_booking_predictions.csv) |
| Trained model artifacts | [`models/`](models/) |

## Project layout

```
facility-usage-prediction/
├── data/
│   └── bookings.csv              synthetic booking history (the dataset)
├── src/
│   ├── generate_dataset.py       synthetic data generator
│   ├── features.py               leakage-safe feature engineering
│   ├── models.py                 model definitions + design matrix
│   ├── train.py                  fit & persist the four models
│   ├── evaluate.py                score on held-out test set, write review output
│   └── predict.py                forward-looking prediction for all residents
├── models/                       trained model artifacts (.joblib)
├── outputs/                       metrics + prediction review tables
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md
├── run_pipeline.py                one-command end-to-end run
└── requirements.txt
```
