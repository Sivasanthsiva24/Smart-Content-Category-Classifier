"""TF-IDF based content classification model."""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


CATEGORIES = [
    "Sports",
    "Technology",
    "Business",
    "Politics",
    "Entertainment",
    "Health",
    "Science",
    "Education",
    "World News",
    "Environment",
]

MODEL_PATH = Path(__file__).with_name("classifier.json")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "had", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "will", "with",
}


def preprocess(text: str) -> str:
    """Normalize text while retaining useful words and numbers."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def _term_counts(text: str) -> Counter[str]:
    words = re.findall(r"[a-z0-9]+", preprocess(text))
    terms = words + [f"{left} {right}" for left, right in zip(words, words[1:])]
    return Counter(terms)


def _tfidf_vector(counts: Counter[str], vocabulary: dict[str, int], idf: list[float]) -> dict[int, float]:
    weighted = {
        vocabulary[term]: (1.0 + math.log(count)) * idf[vocabulary[term]]
        for term, count in counts.items()
        if term in vocabulary
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm:
        return {index: value / norm for index, value in weighted.items()}
    return weighted


def train_model(texts: list[str], labels: list[str]) -> dict[str, Any]:
    if not texts or len(texts) != len(labels):
        raise ValueError("Training texts and labels must be nonempty and have equal lengths.")

    token_counts = [_term_counts(text) for text in texts]
    document_frequency: Counter[str] = Counter()
    for counts in token_counts:
        document_frequency.update(counts.keys())

    vocabulary_terms = sorted(document_frequency)
    vocabulary = {term: index for index, term in enumerate(vocabulary_terms)}
    idf = [
        math.log((1 + len(texts)) / (1 + document_frequency[term])) + 1.0
        for term in vocabulary_terms
    ]
    vectors = [_tfidf_vector(counts, vocabulary, idf) for counts in token_counts]
    classes = sorted(set(labels))
    class_counts = Counter(labels)
    class_feature_totals = {category: [0.0] * len(vocabulary) for category in classes}

    for vector, category in zip(vectors, labels):
        totals = class_feature_totals[category]
        for index, value in vector.items():
            totals[index] += value

    alpha = 1.0
    log_priors = {
        category: math.log(class_counts[category] / len(labels))
        for category in classes
    }
    log_likelihoods = {}
    for category in classes:
        totals = class_feature_totals[category]
        denominator = sum(totals) + alpha * len(vocabulary)
        log_likelihoods[category] = [
            math.log((value + alpha) / denominator) for value in totals
        ]

    return {
        "classes": classes,
        "vocabulary": vocabulary,
        "idf": idf,
        "log_priors": log_priors,
        "log_likelihoods": log_likelihoods,
    }


def save_model(model: dict[str, Any], path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=True), encoding="utf-8")


def load_model(path: Path = MODEL_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_keywords(text: str, model: dict[str, Any], limit: int = 5) -> list[str]:
    cleaned = preprocess(text)
    terms = [term for term in cleaned.split() if term not in STOP_WORDS and len(term) > 2]
    vocabulary = model["vocabulary"]
    weights = _tfidf_vector(_term_counts(text), vocabulary, model["idf"])
    ranked = [
        term
        for index, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True)
        for term in [next(word for word, word_index in vocabulary.items() if word_index == index)]
    ]

    keywords: list[str] = []
    for term in ranked + terms:
        for word in term.split():
            if word not in keywords and word not in STOP_WORDS and len(word) > 2:
                keywords.append(word)
            if len(keywords) == limit:
                return keywords
    return keywords


def explain_prediction(category: str, keywords: list[str]) -> str:
    if not keywords:
        return f"The language and patterns in this text most closely match {category.lower()} content."
    evidence = ", ".join(f"“{word}”" for word in keywords[:3])
    return f"Terms such as {evidence} align with common {category.lower()} topics and reporting."


def predict(text: str, model: dict[str, Any]) -> dict:
    vocabulary = model["vocabulary"]
    weights = _tfidf_vector(_term_counts(text), vocabulary, model["idf"])
    scores = {
        category: model["log_priors"][category]
        + sum(
            value * model["log_likelihoods"][category][index]
            for index, value in weights.items()
        )
        for category in model["classes"]
    }
    best_score = max(scores.values())
    exponentials = {category: math.exp(score - best_score) for category, score in scores.items()}
    total = sum(exponentials.values())
    category = max(scores, key=scores.get)
    keywords = extract_keywords(text, model)
    return {
        "category": category,
        "confidence": round(exponentials[category] / total * 100, 1),
        "keywords": keywords,
        "explanation": explain_prediction(category, keywords),
    }