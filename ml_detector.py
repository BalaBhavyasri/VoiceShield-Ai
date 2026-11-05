import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple
import numpy as np

# ==============================================================================
# VoiceShield AI: Lightweight NumPy-Powered TF-IDF + Naive Bayes ML Classifier
# Provides calibrated scam probability and semantic pattern scoring.
# ==============================================================================

class LightweightScamClassifier:
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.scam_log_probs: np.ndarray = np.array([])
        self.legit_log_probs: np.ndarray = np.array([])
        self.log_prior_scam: float = math.log(0.5)
        self.log_prior_legit: float = math.log(0.5)
        self.is_trained: bool = False

    def tokenize(self, text: str) -> List[str]:
        """Normalize and tokenize text into words and bigrams."""
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in clean.split() if len(w) > 2]
        # Also include adjacent bigrams for phrase matching
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + bigrams

    def fit(self, texts: List[str], labels: List[str]):
        """Train Naive Bayes model on texts with labels ('scam' vs 'legitimate')."""
        tokenized_corpus = [self.tokenize(t) for t in texts]
        doc_count = len(texts)

        # Build vocabulary from top terms
        doc_freq = Counter()
        for doc in tokenized_corpus:
            unique_terms = set(doc)
            for t in unique_terms:
                doc_freq[t] += 1

        # Keep terms that appear in at least 2 documents
        vocab_terms = [term for term, cnt in doc_freq.most_common(1200) if cnt >= 1]
        self.vocabulary = {term: idx for idx, term in enumerate(vocab_terms)}
        vocab_size = len(self.vocabulary)

        if vocab_size == 0:
            return

        # Calculate IDF
        self.idf = np.zeros(vocab_size)
        for term, idx in self.vocabulary.items():
            self.idf[idx] = math.log((1 + doc_count) / (1 + doc_freq[term])) + 1.0

        # Term counts per class
        scam_counts = np.zeros(vocab_size)
        legit_counts = np.zeros(vocab_size)
        scam_docs = 0
        legit_docs = 0

        for doc, lbl in zip(tokenized_corpus, labels):
            is_scam = (lbl.lower().strip() == "scam")
            if is_scam:
                scam_docs += 1
            else:
                legit_docs += 1

            for term in doc:
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    if is_scam:
                        scam_counts[idx] += 1.0
                    else:
                        legit_counts[idx] += 1.0

        # Smoothing & log probability computation
        alpha = 1.0
        total_scam_words = np.sum(scam_counts) + alpha * vocab_size
        total_legit_words = np.sum(legit_counts) + alpha * vocab_size

        self.scam_log_probs = np.log((scam_counts + alpha) / total_scam_words)
        self.legit_log_probs = np.log((legit_counts + alpha) / total_legit_words)

        self.log_prior_scam = math.log(max(1, scam_docs) / doc_count)
        self.log_prior_legit = math.log(max(1, legit_docs) / doc_count)
        self.is_trained = True

    def predict_proba(self, text: str) -> float:
        """Return scam probability (0.0 to 1.0) using log-odds."""
        if not self.is_trained or not text.strip():
            return 0.0

        tokens = self.tokenize(text)
        log_prob_scam = self.log_prior_scam
        log_prob_legit = self.log_prior_legit

        for t in tokens:
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                log_prob_scam += self.scam_log_probs[idx]
                log_prob_legit += self.legit_log_probs[idx]

        # Convert log odds to calibrated sigmoid probability
        log_ratio = log_prob_scam - log_prob_legit
        # Clamp log_ratio to prevent overflow
        log_ratio = max(-15.0, min(15.0, log_ratio))
        prob = 1.0 / (1.0 + math.exp(-log_ratio))
        return float(prob)


# Global classifier instance
_CLASSIFIER: LightweightScamClassifier = LightweightScamClassifier()


def init_ml_classifier(dataset_path: str):
    """Load dataset and train the lightweight ML classifier."""
    import csv
    if not os.path.exists(dataset_path):
        return

    texts = []
    labels = []
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("text") or "").strip()
            lbl = (row.get("label") or "").strip().lower()
            if t and lbl in {"scam", "legitimate"}:
                texts.append(t)
                labels.append(lbl)

    if texts:
        _CLASSIFIER.fit(texts, labels)


def predict_ml_scam(transcript: str) -> Dict[str, Any]:
    """
    Score a transcript using the trained ML model.
    Returns:
        ml_score: 0 to 100
        ml_probability: 0.0 to 1.0
        ml_label: 'Scam' or 'Legitimate'
    """
    if not transcript or not transcript.strip():
        return {"ml_score": 0, "ml_probability": 0.0, "ml_label": "Legitimate"}

    prob = _CLASSIFIER.predict_proba(transcript)
    ml_score = int(round(prob * 100))
    label = "Scam" if prob >= 0.50 else "Legitimate"

    return {
        "ml_score": ml_score,
        "ml_probability": round(prob, 4),
        "ml_label": label
    }


def compute_hybrid_assessment(rule_result: Dict[str, Any], ml_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines rule-based heuristic findings with ML semantic scoring into a single unified result.
    """
    rule_score = int(rule_result.get("risk_score", 0))
    ml_score = int(ml_result.get("ml_score", 0))
    findings = rule_result.get("findings", [])

    # If high severity rule is matched (e.g. OTP theft or Digital arrest), guarantee minimum score
    has_high_rule = any(f.get("severity", "").lower() == "high" for f in findings)

    # Weighted ensemble: 55% rule heuristic + 45% ML confidence
    blended = (0.55 * rule_score) + (0.45 * ml_score)

    if has_high_rule:
        hybrid_score = min(100, max(65, int(round(blended))))
    elif findings:
        hybrid_score = min(100, max(35, int(round(blended))))
    else:
        # If no rules fired, rely on ML but cap at 75 to avoid uncontrolled false positives
        hybrid_score = min(75, int(round(ml_score * 0.75)))

    # Determine hybrid risk tier
    if hybrid_score >= 60:
        hybrid_level = "High"
    elif hybrid_score >= 25:
        hybrid_level = "Caution"
    else:
        hybrid_level = "No patterns detected"

    merged = dict(rule_result)
    merged["ml_score"] = ml_score
    merged["ml_probability"] = ml_result.get("ml_probability", 0.0)
    merged["ml_label"] = ml_result.get("ml_label", "Legitimate")
    merged["hybrid_score"] = hybrid_score
    merged["hybrid_level"] = hybrid_level

    return merged
