#!/usr/bin/env python3
"""
Comprehensive evaluation runner for the reference checker system.
Executes the full 1000-sample test dataset and generates detailed reports.
"""
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Import our enhanced checker modules
from checker.parsing import extract_pdf_text, parse_reference_entries, extract_in_text_citations
from checker.retrieval import resolve_reference, fetch_source_text
from checker.verification import build_index, best_support_for_claim, analyze_citation_patterns
from checker.evaluation import ReferenceCheckerEvaluator, create_performance_dashboard

# Import test dataset generator
from test_dataset_generator import CitationTestDataset, save_test_dataset

class ReferenceCheckerTestRunner:
    """Comprehensive test runner for reference checker evaluation."""
    
    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.evaluator = ReferenceCheckerEvaluator(str(self.results_dir / "logs"))
        
    def run_full_evaluation(self, use_cached_dataset: bool = True) -> Dict:
        """Run complete evaluation with 1000 test cases."""
        
        print("🚀 Starting comprehensive reference checker evaluation...")
        start_time = time.time()
        
        # Step 1: Generate or load test dataset
        dataset = self._get_test_dataset(use_cached_dataset)
        
        # Step 2: Run evaluation on correct citations
        print("\\n📊 Evaluating correct citations (500 samples)...")
        correct_results = self._evaluate_citation_batch(
            dataset["correct_citations"], 
            expected_correct=True
        )
        
        # Step 3: Run evaluation on incorrect citations  
        print("\\n⚠️  Evaluating incorrect citations (500 samples)...")
        incorrect_results = self._evaluate_citation_batch(
            dataset["incorrect_citations"], 
            expected_correct=False
        )
        
        # Step 4: Analyze overall performance
        print("\\n📈 Computing comprehensive metrics...")
        all_results = correct_results + incorrect_results
        
        # Prepare ground truth in expected format
        ground_truth = []
        for i, citation in enumerate(dataset["correct_citations"]):
            ground_truth.append({
                "id": citation["id"],
                "label": citation["ground_truth_label"],
                "citation_type": citation["citation_type"],
                "domain": citation["domain"],
                "complexity": citation["complexity"]
            })
        
        for citation in dataset["incorrect_citations"]:
            ground_truth.append({
                "id": citation["id"], 
                "label": citation["ground_truth_label"],
                "citation_type": citation["citation_type"],
                "domain": citation["domain"],
                "complexity": citation["complexity"]
            })
        
        # Calculate final metrics
        final_metrics = self.evaluator.evaluate_batch(all_results, ground_truth)
        
        # Step 5: Generate comprehensive report
        report = self._generate_final_report(
            dataset, all_results, final_metrics, time.time() - start_time
        )
        
        # Step 6: Save results
        report_file = self._save_results(report)
        
        print(f"\\n✅ Evaluation complete! Results saved to {report_file}")
        print(f"⏱️  Total time: {report['evaluation_summary']['execution_time']:.1f} seconds")
        print(f"🎯 Overall accuracy: {final_metrics.get('overall_accuracy', 0):.2%}")
        
        return report
    
    def _get_test_dataset(self, use_cached: bool) -> Dict:
        """Get test dataset, generating if needed."""
        
        dataset_file = "reference_checker_test_dataset.json"
        
        if use_cached and Path(dataset_file).exists():
            print(f"📂 Loading cached dataset from {dataset_file}")
            with open(dataset_file, 'r') as f:
                return json.load(f)
        else:
            print("🔬 Generating new test dataset...")
            generator = CitationTestDataset()
            dataset = generator.generate_dataset()
            save_test_dataset(dataset, dataset_file)
            return dataset
    
    def _evaluate_citation_batch(self, citations: List[Dict], expected_correct: bool) -> List[Dict]:
        """Evaluate a batch of citations and return results."""
        
        results = []
        total = len(citations)
        
        for i, citation in enumerate(citations, 1):
            try:
                print(f"Processing {i}/{total}: {citation['id']}")
                
                # Simulate the reference checking process
                result = self._check_single_citation(citation)
                results.append(result)
                
                # Progress update every 50 items
                if i % 50 == 0:
                    print(f"  ✅ Completed {i}/{total} citations")
                    
            except Exception as e:
                print(f"  ❌ Error processing {citation['id']}: {str(e)}")
                # Create error result
                results.append({
                    "id": citation["id"],
                    "label": "error", 
                    "score": 0.0,
                    "error": str(e)
                })
        
        return results
    
    def _check_single_citation(self, citation: Dict) -> Dict:
        """Check a single citation and return verification result."""
        
        # Create mock reference data based on citation
        ref_data = {
            "title": citation["source_title"],
            "authors": citation["source_authors"], 
            "year": citation["source_year"],
            "best_id": citation.get("source_doi") or citation.get("source_pmid"),
            "doi": citation.get("source_doi"),
            "pmid": citation.get("source_pmid"),
            "status": "exists" if citation["ground_truth_label"] != "not_found" else "not_found"
        }
        
        # Build source index from abstract
        source_text = citation["source_abstract"]
        if source_text and source_text != "This paper does not exist.":
            index = build_index(source_text)
        else:
            index = {"sentences": [], "embeddings": None}
        
        # Verify claim against source
        if index["sentences"]:
            verification_result = best_support_for_claim(
                citation["claim_text"], 
                index,
                use_nli=True,
                topk=7
            )
        else:
            verification_result = {
                "label": "no_source_text",
                "score": 0.0,
                "evidence": [],
                "explanation": "No source text available"
            }
        
        return {
            "id": citation["id"],
            "label": verification_result["label"],
            "score": verification_result["score"],
            "evidence_count": len(verification_result.get("evidence", [])),
            "explanation": verification_result.get("explanation", ""),
            "citation_type": citation["citation_type"],
            "domain": citation["domain"],
            "complexity": citation["complexity"],
            "error_type": citation.get("error_type")
        }
    
    def _generate_final_report(self, dataset: Dict, results: List[Dict], 
                             metrics: Dict, execution_time: float) -> Dict:
        """Generate comprehensive evaluation report."""
        
        # Analyze citation patterns
        citation_analysis = analyze_citation_patterns(results)
        
        # Performance dashboard
        dashboard = create_performance_dashboard(self.evaluator)
        
        # Error analysis
        error_patterns = self.evaluator.analyze_error_patterns()
        
        # Confusion matrix
        confusion_matrix = self.evaluator.generate_confusion_matrix()
        
        report = {
            "evaluation_summary": {
                "dataset_version": dataset["metadata"]["version"],
                "total_samples": len(results),
                "execution_time": execution_time,
                "timestamp": self.evaluator.session_id,
                "overall_accuracy": metrics.get("overall_accuracy", 0),
                "performance_grade": self.evaluator._grade_performance(metrics.get("overall_accuracy", 0))
            },
            "detailed_metrics": metrics,
            "citation_analysis": citation_analysis,
            "error_patterns": error_patterns,
            "confusion_matrix": confusion_matrix.to_dict() if hasattr(confusion_matrix, 'to_dict') else str(confusion_matrix),
            "performance_dashboard": dashboard,
            "sample_results": results[:10],  # First 10 results as examples
            "recommendations": self._generate_recommendations(metrics, error_patterns),
            "dataset_metadata": dataset["metadata"]
        }
        
        return report
    
    def _generate_recommendations(self, metrics: Dict, error_patterns: Dict) -> List[str]:
        """Generate improvement recommendations based on results."""
        
        recommendations = []
        
        accuracy = metrics.get("overall_accuracy", 0)
        if accuracy < 0.8:
            recommendations.append("Overall accuracy is below acceptable threshold. Focus on improving core algorithms.")
        
        # Citation type specific recommendations
        citation_types = ["numeric", "author-year", "hybrid"]
        for cit_type in citation_types:
            type_accuracy = metrics.get(f"{cit_type}_accuracy", 0)
            if type_accuracy < 0.75:
                recommendations.append(f"Improve {cit_type} citation recognition and processing.")
        
        # Error pattern based recommendations
        high_conf_errors = error_patterns.get("confidence_distribution", {}).get("high_confidence_errors", 0)
        total_errors = error_patterns.get("total_errors", 1)
        
        if high_conf_errors / total_errors > 0.3:
            recommendations.append("High confidence errors detected. Review confidence calibration.")
        
        # NLI specific recommendations
        correct_precision = metrics.get("correct_citation_precision", 0)
        incorrect_recall = metrics.get("incorrect_citation_recall", 0)
        
        if correct_precision < 0.8:
            recommendations.append("Too many false positives. Tighten verification thresholds.")
        
        if incorrect_recall < 0.7:
            recommendations.append("Missing incorrect citations. Improve contradiction detection.")
        
        if len(recommendations) == 0:
            recommendations.append("Performance is satisfactory. Consider fine-tuning for specific domains.")
        
        return recommendations
    
    def _save_results(self, report: Dict) -> str:
        """Save evaluation results to files."""
        
        timestamp = self.evaluator.session_id
        
        # Save comprehensive report
        report_file = self.results_dir / f"evaluation_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save dashboard as text
        dashboard_file = self.results_dir / f"dashboard_{timestamp}.txt"
        with open(dashboard_file, 'w') as f:
            f.write(report["performance_dashboard"])
        
        # Save metrics summary
        metrics_file = self.results_dir / f"metrics_summary_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "summary": report["evaluation_summary"],
                "key_metrics": report["detailed_metrics"],
                "recommendations": report["recommendations"]
            }, f, indent=2)
        
        return str(report_file)

def run_quick_test(sample_size: int = 50) -> Dict:
    """Run a quick evaluation with smaller sample size for testing."""
    
    print(f"🔬 Running quick test with {sample_size} samples...")
    
    # Generate small dataset
    generator = CitationTestDataset()
    dataset = generator.generate_dataset()
    
    # Take subset
    correct_subset = dataset["correct_citations"][:sample_size//2]
    incorrect_subset = dataset["incorrect_citations"][:sample_size//2]
    
    small_dataset = {
        "metadata": dataset["metadata"],
        "correct_citations": correct_subset,
        "incorrect_citations": incorrect_subset
    }
    
    # Run evaluation
    runner = ReferenceCheckerTestRunner("quick_test_results")
    runner._get_test_dataset = lambda x: small_dataset  # Override with small dataset
    
    return runner.run_full_evaluation(use_cached_dataset=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run reference checker evaluation")
    parser.add_argument("--quick", action="store_true", help="Run quick test with 50 samples")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size for quick test")
    parser.add_argument("--no-cache", action="store_true", help="Generate new dataset instead of using cached")
    
    args = parser.parse_args()
    
    if args.quick:
        result = run_quick_test(args.sample_size)
    else:
        runner = ReferenceCheckerTestRunner()
        result = runner.run_full_evaluation(use_cached_dataset=not args.no_cache)
    
    print("\\n🎉 Evaluation completed successfully!")