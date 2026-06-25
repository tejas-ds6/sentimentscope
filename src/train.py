"""
Training script for SentimentScope.
Usage:
    python src/train.py --data data/semeval.json --epochs 3 --batch_size 16
    python src/train.py --demo   # quick smoke-test with synthetic data
"""

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from model import SentimentScopeModel
from data_loader import load_semeval_json, load_csv, train_val_split, make_demo_dataset


def main():
    parser = argparse.ArgumentParser(description="Train SentimentScope")
    parser.add_argument("--data",       type=str,   default=None)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--save_path",  type=str,   default="model_output")
    parser.add_argument("--demo",       action="store_true",
                        help="Use synthetic data for a quick smoke-test")
    args = parser.parse_args()

    if args.demo:
        print("Running in DEMO mode with 200 synthetic samples …")
        data = make_demo_dataset(200)
    elif args.data and args.data.endswith(".json"):
        data = load_semeval_json(args.data)
    elif args.data and args.data.endswith(".csv"):
        data = load_csv(args.data)
    else:
        print("Provide --data <path.json|path.csv> or use --demo")
        sys.exit(1)

    train_data, val_data = train_val_split(data)
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    model   = SentimentScopeModel()
    history = model.train(train_data, val_data,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr=args.lr,
                          save_path=args.save_path)

    print("\nTraining complete.")
    print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
