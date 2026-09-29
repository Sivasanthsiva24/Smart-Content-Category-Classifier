"""Train and save the content classifier from the sample CSV."""

import csv
from pathlib import Path

from model import CATEGORIES, MODEL_PATH, train_model, save_model


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset" / "news_dataset.csv"


def main() -> None:
    with DATASET_PATH.open(encoding="utf-8", newline="") as dataset_file:
        rows = list(csv.DictReader(dataset_file))

    if not rows:
        raise ValueError(f"No training examples found in {DATASET_PATH}")

    texts = [row["text"].strip() for row in rows]
    labels = [row["category"].strip() for row in rows]
    invalid_labels = sorted(set(labels) - set(CATEGORIES))
    if invalid_labels:
        raise ValueError(f"Unknown category labels: {', '.join(invalid_labels)}")

    model = train_model(texts, labels)
    save_model(model, MODEL_PATH)
    print(f"Trained on {len(rows)} examples across {len(set(labels))} categories.")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()