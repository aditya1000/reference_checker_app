\
import math
import json
import regex as rx
import numpy as np
from typing import Dict, List, Tuple, Optional

from sklearn.metrics.pairwise import cosine_similarity

# Sentence embeddings
from sentence_transformers import SentenceTransformer

# Optional NLI
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

_SENT_MODEL = None
_NLI = None

def _get_sent_model():
    global _SENT_MODEL
    if _SENT_MODEL is None:
        _SENT_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SENT_MODEL

def _get_nli_pipeline():
    global _NLI
    if _NLI is None and _HAS_TRANSFORMERS:
        # Try roberta-large-mnli, fallback to bart-large-mnli
        try:
            _NLI = pipeline("text-classification", model="roberta-large-mnli")
        except Exception:
            _NLI = pipeline("text-classification", model="facebook/bart-large-mnli")
    return _NLI

def _sentencize(text: str) -> List[str]:
    # naive sentence split
    text = rx.sub(r'\s+', ' ', text)
    sents = rx.split(r'(?<=[\.\?\!])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 0]

def build_index(source_text: str):
    """Create a simple sentence index with embeddings."""
    if not source_text:
        return {"sentences": [], "embeddings": None}
    sents = _sentencize(source_text)[:2000]  # cap to avoid huge memory
    model = _get_sent_model()
    emb = model.encode(sents, normalize_embeddings=True)
    return {"sentences": sents, "embeddings": emb}

def _nli_label(claim: str, premise: str) -> Tuple[str, float]:
    nli = _get_nli_pipeline()
    if nli is None:
        return ("unknown", 0.0)
    res = nli({"text": premise, "text_pair": claim}, truncation=True, max_length=512)
    if isinstance(res, list): res = res[0]
    label = res["label"].lower()
    score = float(res["score"])
    # normalize to [0,1] for entailment confidence; invert for contrad.
    if "entail" in label:
        return ("entails", score)
    if "contrad" in label:
        return ("contradicts", score)
    return ("neutral", score)

def best_support_for_claim(claim: str, index, use_nli: bool = True, topk: int = 7) -> Dict:
    """Return best evidence sentences and a verdict label."""
    if not index or not index["sentences"]:
        return {"label":"no_source_text","score":0.0,"evidence":[]}

    model = _get_sent_model()
    c_emb = model.encode([claim], normalize_embeddings=True)
    sims = cosine_similarity(c_emb, index["embeddings"])[0]
    top_idx = np.argsort(-sims)[:topk].tolist()
    evidence = [{"sentence": index["sentences"][i], "sim": float(sims[i])} for i in top_idx]

    # Aggregate: if top similarity is high and NLI says entail, mark supported
    verdict = "not_found"
    confidence = float(np.max(sims)) if len(sims) else 0.0
    if use_nli:
        labels = []
        ent_scores = []
        contr_scores = []
        for ev in evidence[:3]:
            lab, sc = _nli_label(claim, ev["sentence"])
            ev["nli_label"] = lab
            ev["nli_score"] = sc
            labels.append(lab)
            if lab == "entails":
                ent_scores.append(sc)
            if lab == "contradicts":
                contr_scores.append(sc)
        if ent_scores and (max(ent_scores) > 0.65 or confidence > 0.55):
            verdict = "supported"
            confidence = max(ent_scores + [confidence])
        elif contr_scores and max(contr_scores) > 0.65:
            verdict = "contradicted_risk"
            confidence = max(contr_scores)
        else:
            verdict = "not_found"
            confidence = max([confidence] + ent_scores + contr_scores)
    else:
        if confidence > 0.6:
            verdict = "supported"
        elif confidence < 0.25:
            verdict = "not_found"

    return {"label": verdict, "score": float(confidence), "evidence": evidence}
