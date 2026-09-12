"""
End-to-end entry point: generate data (if missing) -> train -> evaluate ->
predict -> model comparison / rolling CV -> EDA figures -> dashboard.

Run: python run_pipeline.py   (takes ~1-2 minutes)
"""
from src import train, evaluate, predict, model_comparison, eda, dashboard

if __name__ == "__main__":
    print("=== 1/6 Training ===")
    train.main()
    print("\n=== 2/6 Evaluating on held-out chronological test set ===")
    evaluate.main()
    print("\n=== 3/6 Forward prediction for all residents ===")
    predict.main()
    print("\n=== 4/6 Model comparison + rolling-origin cross-validation ===")
    model_comparison.main()
    print("\n=== 5/6 Dataset EDA figures ===")
    eda.main()
    print("\n=== 6/6 Building review dashboard ===")
    dashboard.main()
