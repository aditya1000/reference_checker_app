\
import math
import json
import logging
import regex as rx
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.metrics.pairwise import cosine_similarity

# Sentence embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Enhanced claim verification with multi-stage analysis."""
    if not index or not index["sentences"]:
        return {"label":"no_source_text","score":0.0,"evidence":[], "confidence_factors": {}}

    # Stage 1: Semantic similarity
    model = _get_sent_model()
    c_emb = model.encode([claim], normalize_embeddings=True)
    sims = cosine_similarity(c_emb, index["embeddings"])[0]
    top_idx = np.argsort(-sims)[:topk].tolist()
    evidence = [{"sentence": index["sentences"][i], "sim": float(sims[i])} for i in top_idx]

    # Stage 2: Enhanced NLI analysis
    confidence_factors = {
        "max_similarity": float(np.max(sims)) if len(sims) else 0.0,
        "avg_top3_similarity": float(np.mean([sims[i] for i in top_idx[:3]])) if len(top_idx) >= 3 else 0.0,
        "entailment_count": 0,
        "contradiction_count": 0,
        "max_entailment_score": 0.0,
        "max_contradiction_score": 0.0
    }

    if use_nli:
        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []
        
        for ev in evidence[:5]:  # Analyze top 5 sentences
            lab, sc = _nli_label(claim, ev["sentence"])
            ev["nli_label"] = lab
            ev["nli_score"] = sc
            
            if lab == "entails":
                entailment_scores.append(sc)
                confidence_factors["entailment_count"] += 1
            elif lab == "contradicts":
                contradiction_scores.append(sc)
                confidence_factors["contradiction_count"] += 1
            else:
                neutral_scores.append(sc)
        
        if entailment_scores:
            confidence_factors["max_entailment_score"] = max(entailment_scores)
            confidence_factors["avg_entailment_score"] = sum(entailment_scores) / len(entailment_scores)
        
        if contradiction_scores:
            confidence_factors["max_contradiction_score"] = max(contradiction_scores)
            confidence_factors["avg_contradiction_score"] = sum(contradiction_scores) / len(contradiction_scores)

    # Stage 3: Multi-factor decision making
    verdict, final_confidence = _make_verification_decision(confidence_factors, use_nli)
    
    # Stage 4: Add reasoning explanation
    explanation = _generate_explanation(confidence_factors, verdict, final_confidence)

    return {
        "label": verdict, 
        "score": float(final_confidence), 
        "evidence": evidence,
        "confidence_factors": confidence_factors,
        "explanation": explanation
    }

def _make_verification_decision(factors: Dict, use_nli: bool) -> Tuple[str, float]:
    """Make verification decision based on multiple confidence factors."""
    max_sim = factors["max_similarity"]
    max_ent = factors.get("max_entailment_score", 0.0)
    max_contr = factors.get("max_contradiction_score", 0.0)
    ent_count = factors.get("entailment_count", 0)
    contr_count = factors.get("contradiction_count", 0)
    
    if use_nli:
        # Strong entailment evidence
        if max_ent > 0.8 and ent_count >= 2:
            return "strongly_supported", max_ent
        elif max_ent > 0.7 and (ent_count >= 1 or max_sim > 0.6):
            return "supported", max_ent
        elif max_ent > 0.6 and max_sim > 0.5:
            return "weakly_supported", (max_ent + max_sim) / 2
        
        # Contradiction detection
        elif max_contr > 0.8:
            return "contradicted", max_contr
        elif max_contr > 0.7 and contr_count >= 2:
            return "likely_contradicted", max_contr
        
        # High similarity but neutral/low NLI
        elif max_sim > 0.7 and max_ent < 0.5:
            return "related_but_insufficient", max_sim
        elif max_sim > 0.5:
            return "weakly_related", max_sim
        else:
            return "not_found", max(max_sim, max_ent, max_contr)
    else:
        # Similarity-only fallback
        if max_sim > 0.75:
            return "supported", max_sim
        elif max_sim > 0.6:
            return "weakly_supported", max_sim
        elif max_sim > 0.4:
            return "related_but_insufficient", max_sim
        else:
            return "not_found", max_sim

def _generate_explanation(factors: Dict, verdict: str, confidence: float) -> str:
    """Generate human-readable explanation of the verification result."""
    max_sim = factors["max_similarity"]
    ent_count = factors.get("entailment_count", 0)
    contr_count = factors.get("contradiction_count", 0)
    max_ent = factors.get("max_entailment_score", 0.0)
    max_contr = factors.get("max_contradiction_score", 0.0)
    
    explanations = []
    
    if verdict.startswith("supported"):
        explanations.append(f"High semantic similarity (max: {max_sim:.2f})")
        if ent_count > 0:
            explanations.append(f"{ent_count} sentence(s) show entailment (max score: {max_ent:.2f})")
    elif verdict.startswith("contradicted"):
        explanations.append(f"Found contradiction evidence (score: {max_contr:.2f})")
        if contr_count > 1:
            explanations.append(f"Multiple contradictory sentences detected ({contr_count})")
    elif verdict == "related_but_insufficient":
        explanations.append(f"High textual similarity ({max_sim:.2f}) but insufficient semantic entailment")
    elif verdict == "not_found":
        explanations.append(f"Low similarity to source content (max: {max_sim:.2f})")
    
    return "; ".join(explanations) if explanations else "No clear relationship found"

def analyze_citation_patterns(claims: List[Dict]) -> Dict:
    """Analyze patterns in citation verification results."""
    patterns = {
        "total_claims": len(claims),
        "verdict_distribution": defaultdict(int),
        "confidence_stats": {
            "mean": 0.0,
            "median": 0.0,
            "high_confidence": 0,  # > 0.8
            "medium_confidence": 0,  # 0.5-0.8
            "low_confidence": 0   # < 0.5
        },
        "citation_type_performance": defaultdict(lambda: {"count": 0, "avg_confidence": 0.0}),
        "potential_issues": []
    }
    
    if not claims:
        return patterns
    
    confidences = []
    
    for claim in claims:
        verdict = claim.get("label", "unknown")
        confidence = claim.get("score", 0.0)
        citation_type = claim.get("citation_type", "unknown")
        
        patterns["verdict_distribution"][verdict] += 1
        confidences.append(confidence)
        
        # Track performance by citation type
        type_stats = patterns["citation_type_performance"][citation_type]
        type_stats["count"] += 1
        type_stats["avg_confidence"] = (
            (type_stats["avg_confidence"] * (type_stats["count"] - 1) + confidence) / 
            type_stats["count"]
        )
        
        # Confidence categorization
        if confidence > 0.8:
            patterns["confidence_stats"]["high_confidence"] += 1
        elif confidence > 0.5:
            patterns["confidence_stats"]["medium_confidence"] += 1
        else:
            patterns["confidence_stats"]["low_confidence"] += 1
        
        # Flag potential issues
        if verdict in ["contradicted", "likely_contradicted"]:
            patterns["potential_issues"].append({
                "type": "contradiction",
                "claim": claim.get("claim_text", "")[:100] + "...",
                "confidence": confidence
            })
        elif verdict == "not_found" and citation_type == "numeric":
            patterns["potential_issues"].append({
                "type": "missing_reference",
                "claim": claim.get("claim_text", "")[:100] + "...",
                "marker": claim.get("marker", "")
            })
    
    # Calculate statistics
    if confidences:
        patterns["confidence_stats"]["mean"] = sum(confidences) / len(confidences)
        patterns["confidence_stats"]["median"] = sorted(confidences)[len(confidences) // 2]
    
    return patterns
