"""
End-to-end entry point: generate data (if missing) -> train -> evaluate -> predict.

Run: python run_pipeline.py
"""
from src import train, evaluate, predict

if __name__ == "__main__":
    print("=== 1/3 Training ===")
    train.main()
    print("\n=== 2/3 Evaluating on held-out chronological test set ===")
    evaluate.main()
    print("\n=== 3/3 Forward prediction for all residents ===")
    predict.main()
