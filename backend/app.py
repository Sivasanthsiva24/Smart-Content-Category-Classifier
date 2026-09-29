"""Flask API and web server for Smart Content Category Classifier."""

import csv
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from model import CATEGORIES, MODEL_PATH, load_model, predict, train_model, save_model


ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
DATASET_PATH = ROOT / "dataset" / "news_dataset.csv"
app = Flask(__name__)


def get_model():
    if not MODEL_PATH.exists():
        with DATASET_PATH.open(encoding="utf-8", newline="") as dataset_file:
            rows = list(csv.DictReader(dataset_file))
        if not rows:
            raise RuntimeError("The training dataset is empty.")
        model = train_model(
            [row["text"] for row in rows], [row["category"] for row in rows]
        )
        save_model(model, MODEL_PATH)
    return load_model(MODEL_PATH)


classifier = get_model()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_asset(filename: str):
    return send_from_directory(FRONTEND_DIR, filename)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "categories": CATEGORIES})


@app.post("/predict")
def classify():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON object containing a text field."}), 400

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter some content to classify."}), 400
    if len(text) > 12000:
        return jsonify({"error": "Content must be 12,000 characters or fewer."}), 413

    return jsonify(predict(text.strip(), classifier))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)