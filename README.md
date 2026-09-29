# Smart Content Category Classifier

A small full-stack text classifier that sorts writing into Sports, Technology, Business, Politics, Entertainment, Health, Science, Education, World News, or Environment. It uses TF-IDF features and a Multinomial Naive Bayes classifier, trained on the included CSV dataset. Flask serves both the JSON API and the responsive browser interface. The classifier uses Python's standard library, so it does not need NumPy or SciPy native extensions.

## Run locally

Requires Python 3.10 or newer. From the project root, create and activate a virtual environment, install dependencies, train the model, and start Flask:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
python backend\train.py
python backend\app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). If PowerShell prevents environment activation, run the venv interpreter directly with `.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`, then replace `python` in the last two commands with `.\.venv\Scripts\python.exe`.

The saved model is written to `backend/classifier.json`. If no saved model exists, starting the Flask app trains it from the sample CSV automatically. After changing the dataset, run `python backend/train.py` again to update the model.

## API

`POST /predict` accepts JSON and returns a category, model confidence, extracted keywords, and a short explanation:

```json
{
  "text": "A new artificial intelligence technology was launched today."
}
```

The response includes `category`, `confidence`, `keywords`, and `explanation`. Empty, invalid, or oversized requests return a JSON error with an appropriate HTTP status. A `GET /health` endpoint reports server status and available categories.

Example PowerShell request:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method Post -ContentType 'application/json' -Body '{"text":"A new artificial intelligence technology was launched today."}'
```

The sample corpus is deliberately small and educational. Confidence is the logistic regression model's probability estimate, not a calibrated guarantee; predictions on specialized, ambiguous, or out-of-domain writing may be inaccurate. Add representative labeled examples to `dataset/news_dataset.csv` and retrain for better results. Do not load a model artifact from an untrusted source, as joblib uses Python pickle serialization.