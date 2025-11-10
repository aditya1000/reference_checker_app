"""
Evaluation framework for reference checker with metrics and performance analysis.
"""
import json
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReferenceCheckerEvaluator:
    """Comprehensive evaluation system for reference checking accuracy."""
    
    def __init__(self, log_dir: str = "evaluation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            "precision": {},
            "recall": {},
            "f1_score": {},
            "accuracy": {},
            "confusion_matrix": {},
            "performance_by_type": {}
        }
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_log = []
        
    def evaluate_single_case(self, 
                           predicted_label: str, 
                           true_label: str,
                           confidence: float,
                           case_metadata: Dict) -> Dict:
        """Evaluate a single reference checking case."""
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "predicted_label": predicted_label,
            "true_label": true_label,
            "confidence": confidence,
            "is_correct": predicted_label == true_label,
            "metadata": case_metadata
        }
        
        # Calculate label-specific metrics
        if true_label in ["supported", "strongly_supported", "weakly_supported"]:
            result["true_category"] = "correct_citation"
        elif true_label in ["contradicted", "likely_contradicted"]:
            result["true_category"] = "incorrect_citation"
        else:
            result["true_category"] = "unclear_citation"
            
        if predicted_label in ["supported", "strongly_supported", "weakly_supported"]:
            result["predicted_category"] = "correct_citation"
        elif predicted_label in ["contradicted", "likely_contradicted"]:
            result["predicted_category"] = "incorrect_citation"
        else:
            result["predicted_category"] = "unclear_citation"
        
        self.results_log.append(result)
        return result
    
    def evaluate_batch(self, 
                      predictions: List[Dict], 
                      ground_truth: List[Dict]) -> Dict:
        """Evaluate a batch of predictions against ground truth."""
        
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        batch_results = []
        
        for pred, truth in zip(predictions, ground_truth):
            result = self.evaluate_single_case(
                predicted_label=pred.get("label", "unknown"),
                true_label=truth.get("label", "unknown"),
                confidence=pred.get("score", 0.0),
                case_metadata={
                    "claim_id": truth.get("id", ""),
                    "citation_type": truth.get("citation_type", ""),
                    "domain": truth.get("domain", ""),
                    "complexity": truth.get("complexity", "medium")
                }
            )
            batch_results.append(result)
        
        return self.calculate_batch_metrics(batch_results)
    
    def calculate_batch_metrics(self, batch_results: List[Dict]) -> Dict:
        """Calculate comprehensive metrics for a batch of results."""
        
        total = len(batch_results)
        if total == 0:
            return {}
        
        # Basic accuracy
        correct = sum(1 for r in batch_results if r["is_correct"])
        accuracy = correct / total
        
        # Category-wise metrics
        categories = ["correct_citation", "incorrect_citation", "unclear_citation"]
        metrics = {}
        
        for category in categories:
            true_positives = sum(1 for r in batch_results 
                               if r["true_category"] == category and 
                                  r["predicted_category"] == category)
            
            false_positives = sum(1 for r in batch_results 
                                if r["true_category"] != category and 
                                   r["predicted_category"] == category)
            
            false_negatives = sum(1 for r in batch_results 
                                if r["true_category"] == category and 
                                   r["predicted_category"] != category)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f"{category}_precision"] = precision
            metrics[f"{category}_recall"] = recall
            metrics[f"{category}_f1"] = f1
        
        # Confidence analysis
        confidences = [r["confidence"] for r in batch_results]
        correct_confidences = [r["confidence"] for r in batch_results if r["is_correct"]]
        incorrect_confidences = [r["confidence"] for r in batch_results if not r["is_correct"]]
        
        metrics.update({
            "overall_accuracy": accuracy,
            "total_cases": total,
            "correct_cases": correct,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "avg_correct_confidence": sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0,
            "avg_incorrect_confidence": sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0,
        })
        
        # Performance by citation type
        citation_types = set(r["metadata"]["citation_type"] for r in batch_results)
        for cit_type in citation_types:
            type_results = [r for r in batch_results if r["metadata"]["citation_type"] == cit_type]
            type_correct = sum(1 for r in type_results if r["is_correct"])
            type_total = len(type_results)
            
            metrics[f"{cit_type}_accuracy"] = type_correct / type_total if type_total > 0 else 0
            metrics[f"{cit_type}_count"] = type_total
        
        return metrics
    
    def save_evaluation_report(self, metrics: Dict, filename: Optional[str] = None) -> str:
        """Save evaluation report to file."""
        
        if filename is None:
            filename = f"evaluation_report_{self.session_id}.json"
        
        filepath = self.log_dir / filename
        
        report = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "detailed_results": self.results_log,
            "summary": self._generate_summary(metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {filepath}")
        return str(filepath)
    
    def _generate_summary(self, metrics: Dict) -> Dict:
        """Generate human-readable summary of evaluation results."""
        
        total = metrics.get("total_cases", 0)
        accuracy = metrics.get("overall_accuracy", 0)
        
        summary = {
            "performance_grade": self._grade_performance(accuracy),
            "key_findings": [],
            "recommendations": []
        }
        
        # Key findings
        if accuracy >= 0.9:
            summary["key_findings"].append("Excellent overall accuracy achieved")
        elif accuracy >= 0.8:
            summary["key_findings"].append("Good overall accuracy with room for improvement")
        else:
            summary["key_findings"].append("Accuracy below acceptable threshold")
        
        # Check for specific issues
        correct_cit_precision = metrics.get("correct_citation_precision", 0)
        incorrect_cit_recall = metrics.get("incorrect_citation_recall", 0)
        
        if correct_cit_precision < 0.8:
            summary["key_findings"].append("High false positive rate for correct citations")
            summary["recommendations"].append("Improve precision for detecting correct citations")
        
        if incorrect_cit_recall < 0.7:
            summary["key_findings"].append("Missing many incorrect citations")
            summary["recommendations"].append("Enhance detection of contradictory or incorrect citations")
        
        # Citation type performance analysis
        citation_types = [k for k in metrics.keys() if k.endswith("_accuracy") and k != "overall_accuracy"]
        if citation_types:
            best_type = max(citation_types, key=lambda x: metrics[x])
            worst_type = min(citation_types, key=lambda x: metrics[x])
            
            summary["key_findings"].append(f"Best performance on {best_type.replace('_accuracy', '')} citations")
            if metrics[worst_type] < 0.7:
                summary["key_findings"].append(f"Poor performance on {worst_type.replace('_accuracy', '')} citations")
                summary["recommendations"].append(f"Improve handling of {worst_type.replace('_accuracy', '')} citation style")
        
        return summary
    
    def _grade_performance(self, accuracy: float) -> str:
        """Assign letter grade based on accuracy."""
        if accuracy >= 0.95:
            return "A+"
        elif accuracy >= 0.9:
            return "A"
        elif accuracy >= 0.85:
            return "B+"
        elif accuracy >= 0.8:
            return "B"
        elif accuracy >= 0.75:
            return "C+"
        elif accuracy >= 0.7:
            return "C"
        else:
            return "D"
    
    def generate_confusion_matrix(self) -> pd.DataFrame:
        """Generate confusion matrix from logged results."""
        
        if not self.results_log:
            return pd.DataFrame()
        
        # Get all unique categories
        all_categories = set()
        for result in self.results_log:
            all_categories.add(result["true_category"])
            all_categories.add(result["predicted_category"])
        
        all_categories = sorted(list(all_categories))
        
        # Initialize confusion matrix
        matrix = pd.DataFrame(0, index=all_categories, columns=all_categories)
        
        # Fill confusion matrix
        for result in self.results_log:
            true_cat = result["true_category"]
            pred_cat = result["predicted_category"]
            matrix.loc[true_cat, pred_cat] += 1
        
        return matrix
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze patterns in incorrect predictions."""
        
        errors = [r for r in self.results_log if not r["is_correct"]]
        
        if not errors:
            return {"message": "No errors to analyze"}
        
        patterns = {
            "total_errors": len(errors),
            "error_by_type": {},
            "error_by_domain": {},
            "confidence_distribution": {
                "high_confidence_errors": 0,  # confidence > 0.8
                "medium_confidence_errors": 0,  # 0.5 < confidence <= 0.8
                "low_confidence_errors": 0    # confidence <= 0.5
            },
            "common_error_types": []
        }
        
        # Analyze error patterns
        for error in errors:
            # By citation type
            cit_type = error["metadata"]["citation_type"]
            patterns["error_by_type"][cit_type] = patterns["error_by_type"].get(cit_type, 0) + 1
            
            # By domain
            domain = error["metadata"]["domain"]
            patterns["error_by_domain"][domain] = patterns["error_by_domain"].get(domain, 0) + 1
            
            # By confidence
            conf = error["confidence"]
            if conf > 0.8:
                patterns["confidence_distribution"]["high_confidence_errors"] += 1
            elif conf > 0.5:
                patterns["confidence_distribution"]["medium_confidence_errors"] += 1
            else:
                patterns["confidence_distribution"]["low_confidence_errors"] += 1
        
        # Identify common error patterns
        if patterns["confidence_distribution"]["high_confidence_errors"] > len(errors) * 0.3:
            patterns["common_error_types"].append("High confidence false positives")
        
        most_error_type = max(patterns["error_by_type"], key=patterns["error_by_type"].get) if patterns["error_by_type"] else None
        if most_error_type and patterns["error_by_type"][most_error_type] > len(errors) * 0.4:
            patterns["common_error_types"].append(f"Systematic issues with {most_error_type} citations")
        
        return patterns

def create_performance_dashboard(evaluator: ReferenceCheckerEvaluator) -> str:
    """Create a simple text-based performance dashboard."""
    
    if not evaluator.results_log:
        return "No evaluation data available"
    
    metrics = evaluator.calculate_batch_metrics(evaluator.results_log)
    confusion_matrix = evaluator.generate_confusion_matrix()
    error_patterns = evaluator.analyze_error_patterns()
    
    dashboard = f"""
=== REFERENCE CHECKER PERFORMANCE DASHBOARD ===
Session: {evaluator.session_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 OVERALL PERFORMANCE:
• Total Cases: {metrics.get('total_cases', 0)}
• Accuracy: {metrics.get('overall_accuracy', 0):.2%}
• Grade: {evaluator._grade_performance(metrics.get('overall_accuracy', 0))}

📈 CATEGORY PERFORMANCE:
• Correct Citations - Precision: {metrics.get('correct_citation_precision', 0):.2%}, Recall: {metrics.get('correct_citation_recall', 0):.2%}
• Incorrect Citations - Precision: {metrics.get('incorrect_citation_precision', 0):.2%}, Recall: {metrics.get('incorrect_citation_recall', 0):.2%}
• Unclear Citations - Precision: {metrics.get('unclear_citation_precision', 0):.2%}, Recall: {metrics.get('unclear_citation_recall', 0):.2%}

🎯 CONFIDENCE ANALYSIS:
• Average Confidence: {metrics.get('avg_confidence', 0):.2f}
• Correct Predictions: {metrics.get('avg_correct_confidence', 0):.2f}
• Incorrect Predictions: {metrics.get('avg_incorrect_confidence', 0):.2f}

⚠️  ERROR ANALYSIS:
• Total Errors: {error_patterns.get('total_errors', 0)}
• High Confidence Errors: {error_patterns.get('confidence_distribution', {}).get('high_confidence_errors', 0)}
• Most Problematic Type: {max(error_patterns.get('error_by_type', {'None': 0}), key=error_patterns.get('error_by_type', {'None': 0}).get)}

📋 CONFUSION MATRIX:
{confusion_matrix.to_string() if not confusion_matrix.empty else 'No data'}
"""
    
    return dashboard