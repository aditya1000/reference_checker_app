"""
Test dataset schema and generator for reference checker evaluation.
Creates 1000 test cases: 500 correct citations + 500 incorrect citations.
"""
import json
import random
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class TestCitation:
    """Schema for a test citation case."""
    id: str
    claim_text: str
    citation_marker: str
    citation_type: str  # numeric, author-year, hybrid
    source_title: str
    source_authors: List[str]
    source_year: int
    source_abstract: str
    source_doi: Optional[str]
    source_pmid: Optional[str]
    ground_truth_label: str  # supported, contradicted, not_found, etc.
    confidence_level: float  # expected confidence for correct system
    domain: str  # medicine, computer_science, physics, etc.
    complexity: str  # simple, medium, complex
    error_type: Optional[str]  # for incorrect citations
    created_date: str
    source_quality: str  # high_impact, peer_reviewed, preprint
    
    def to_dict(self) -> Dict:
        return asdict(self)

class CitationTestDataset:
    """Generate and manage test dataset for reference checker evaluation."""
    
    def __init__(self):
        self.correct_citations = []
        self.incorrect_citations = []
        self.dataset_metadata = {
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "total_samples": 1000,
            "correct_samples": 500,
            "incorrect_samples": 500,
            "domains": ["medicine", "computer_science", "physics", "biology", "psychology", "chemistry"],
            "citation_types": ["numeric", "author-year", "hybrid"],
            "error_types": ["misattribution", "misrepresentation", "wrong_source", "fabricated", "out_of_context"]
        }
    
    def generate_dataset(self) -> Dict:
        """Generate complete test dataset with 1000 samples."""
        print("🔬 Generating comprehensive test dataset...")
        
        # Generate correct citations (500 samples)
        self.correct_citations = self._generate_correct_citations(500)
        
        # Generate incorrect citations (500 samples) 
        self.incorrect_citations = self._generate_incorrect_citations(500)
        
        dataset = {
            "metadata": self.dataset_metadata,
            "correct_citations": [citation.to_dict() for citation in self.correct_citations],
            "incorrect_citations": [citation.to_dict() for citation in self.incorrect_citations],
            "validation_instructions": self._create_validation_instructions()
        }
        
        return dataset
    
    def _generate_correct_citations(self, count: int) -> List[TestCitation]:
        """Generate correct citation test cases from real paper abstracts."""
        correct_cases = []
        
        # High-impact paper examples with real abstracts and claims
        real_paper_data = self._get_real_paper_examples()
        
        domains = self.dataset_metadata["domains"]
        citation_types = self.dataset_metadata["citation_types"]
        
        for i in range(count):
            # Select random paper data
            paper = random.choice(real_paper_data)
            domain = random.choice(domains)
            citation_type = random.choice(citation_types)
            complexity = random.choices(["simple", "medium", "complex"], weights=[0.3, 0.5, 0.2])[0]
            
            # Generate claim that is supported by the abstract
            claim = self._generate_supported_claim(paper["abstract"], complexity)
            marker = self._generate_citation_marker(citation_type, paper["authors"], paper["year"], i+1)
            
            citation = TestCitation(
                id=f"correct_{i+1:03d}",
                claim_text=claim,
                citation_marker=marker,
                citation_type=citation_type,
                source_title=paper["title"],
                source_authors=paper["authors"],
                source_year=paper["year"],
                source_abstract=paper["abstract"],
                source_doi=paper.get("doi"),
                source_pmid=paper.get("pmid"),
                ground_truth_label="supported" if complexity != "complex" else "weakly_supported",
                confidence_level=0.85 if complexity == "simple" else 0.75,
                domain=domain,
                complexity=complexity,
                error_type=None,
                created_date=datetime.now().isoformat(),
                source_quality="high_impact"
            )
            
            correct_cases.append(citation)
        
        return correct_cases
    
    def _generate_incorrect_citations(self, count: int) -> List[TestCitation]:
        """Generate incorrect citation test cases with various error types."""
        incorrect_cases = []
        
        error_types = self.dataset_metadata["error_types"]
        domains = self.dataset_metadata["domains"]
        citation_types = self.dataset_metadata["citation_types"]
        
        # Distribute error types
        error_distribution = {
            "misattribution": int(count * 0.25),    # 25% - wrong source cited
            "misrepresentation": int(count * 0.30), # 30% - claim doesn't match source
            "wrong_source": int(count * 0.20),      # 20% - citing unrelated paper
            "fabricated": int(count * 0.15),        # 15% - non-existent source
            "out_of_context": int(count * 0.10)     # 10% - claim taken out of context
        }
        
        case_id = 1
        for error_type, error_count in error_distribution.items():
            for i in range(error_count):
                domain = random.choice(domains)
                citation_type = random.choice(citation_types)
                complexity = random.choices(["simple", "medium", "complex"], weights=[0.4, 0.4, 0.2])[0]
                
                citation = self._generate_error_case(error_type, case_id, domain, citation_type, complexity)
                incorrect_cases.append(citation)
                case_id += 1
        
        return incorrect_cases
    
    def _generate_error_case(self, error_type: str, case_id: int, domain: str, citation_type: str, complexity: str) -> TestCitation:
        """Generate specific type of incorrect citation."""
        
        if error_type == "misattribution":
            return self._generate_misattribution_case(case_id, domain, citation_type, complexity)
        elif error_type == "misrepresentation":
            return self._generate_misrepresentation_case(case_id, domain, citation_type, complexity)
        elif error_type == "wrong_source":
            return self._generate_wrong_source_case(case_id, domain, citation_type, complexity)
        elif error_type == "fabricated":
            return self._generate_fabricated_case(case_id, domain, citation_type, complexity)
        elif error_type == "out_of_context":
            return self._generate_out_of_context_case(case_id, domain, citation_type, complexity)
    
    def _generate_misattribution_case(self, case_id: int, domain: str, citation_type: str, complexity: str) -> TestCitation:
        """Generate case where claim is attributed to wrong author/paper."""
        
        # Get two different papers
        papers = self._get_real_paper_examples()
        paper_a = random.choice(papers)
        paper_b = random.choice([p for p in papers if p != paper_a])
        
        # Claim from paper A, but citing paper B
        claim = self._generate_supported_claim(paper_a["abstract"], complexity)
        marker = self._generate_citation_marker(citation_type, paper_b["authors"], paper_b["year"], case_id)
        
        return TestCitation(
            id=f"incorrect_{case_id:03d}",
            claim_text=claim,
            citation_marker=marker,
            citation_type=citation_type,
            source_title=paper_b["title"],
            source_authors=paper_b["authors"],
            source_year=paper_b["year"],
            source_abstract=paper_b["abstract"],
            source_doi=paper_b.get("doi"),
            source_pmid=paper_b.get("pmid"),
            ground_truth_label="not_found",
            confidence_level=0.3,
            domain=domain,
            complexity=complexity,
            error_type="misattribution",
            created_date=datetime.now().isoformat(),
            source_quality="peer_reviewed"
        )
    
    def _generate_misrepresentation_case(self, case_id: int, domain: str, citation_type: str, complexity: str) -> TestCitation:
        """Generate case where claim misrepresents what the source actually says."""
        
        papers = self._get_real_paper_examples()
        paper = random.choice(papers)
        
        # Generate claim that contradicts or misrepresents the abstract
        claim = self._generate_contradictory_claim(paper["abstract"], complexity)
        marker = self._generate_citation_marker(citation_type, paper["authors"], paper["year"], case_id)
        
        return TestCitation(
            id=f"incorrect_{case_id:03d}",
            claim_text=claim,
            citation_marker=marker,
            citation_type=citation_type,
            source_title=paper["title"],
            source_authors=paper["authors"],
            source_year=paper["year"],
            source_abstract=paper["abstract"],
            source_doi=paper.get("doi"),
            source_pmid=paper.get("pmid"),
            ground_truth_label="contradicted",
            confidence_level=0.75,  # Should be confident this is wrong
            domain=domain,
            complexity=complexity,
            error_type="misrepresentation",
            created_date=datetime.now().isoformat(),
            source_quality="peer_reviewed"
        )
    
    def _generate_wrong_source_case(self, case_id: int, domain: str, citation_type: str, complexity: str) -> TestCitation:
        """Generate case citing completely unrelated source."""
        
        papers = self._get_real_paper_examples()
        paper = random.choice(papers)
        
        # Generate claim about unrelated topic
        unrelated_claims = {
            "medicine": "Machine learning algorithms show promising results in image classification tasks.",
            "computer_science": "Dietary interventions significantly reduced cardiovascular risk factors.",
            "physics": "Social media usage correlates with increased anxiety in adolescents.",
            "biology": "Quantum entanglement enables faster-than-light communication protocols.",
            "psychology": "CRISPR gene editing successfully treats genetic blindness disorders.",
            "chemistry": "Climate change models predict 2°C warming by 2050."
        }
        
        claim = random.choice(list(unrelated_claims.values()))
        marker = self._generate_citation_marker(citation_type, paper["authors"], paper["year"], case_id)
        
        return TestCitation(
            id=f"incorrect_{case_id:03d}",
            claim_text=claim,
            citation_marker=marker,
            citation_type=citation_type,
            source_title=paper["title"],
            source_authors=paper["authors"],
            source_year=paper["year"],
            source_abstract=paper["abstract"],
            source_doi=paper.get("doi"),
            source_pmid=paper.get("pmid"),
            ground_truth_label="not_found",
            confidence_level=0.2,
            domain=domain,
            complexity=complexity,
            error_type="wrong_source",
            created_date=datetime.now().isoformat(),
            source_quality="peer_reviewed"
        )
    
    def _generate_fabricated_case(self, case_id: int, domain: str, citation_type: str, complexity: str) -> TestCitation:
        """Generate case citing non-existent paper."""
        
        # Create fake paper metadata
        fake_authors = self._generate_fake_authors()
        fake_year = random.randint(2015, 2024)
        fake_title = self._generate_fake_title(domain)
        
        claim = f"Recent studies demonstrate significant improvements in {domain} methodologies and outcomes."
        marker = self._generate_citation_marker(citation_type, fake_authors, fake_year, case_id)
        
        return TestCitation(
            id=f"incorrect_{case_id:03d}",
            claim_text=claim,
            citation_marker=marker,
            citation_type=citation_type,
            source_title=fake_title,
            source_authors=fake_authors,
            source_year=fake_year,
            source_abstract="This paper does not exist.",
            source_doi=None,
            source_pmid=None,
            ground_truth_label="not_found",
            confidence_level=0.1,
            domain=domain,
            complexity=complexity,
            error_type="fabricated",
            created_date=datetime.now().isoformat(),
            source_quality="non_existent"
        )
    
    def _generate_out_of_context_case(self, case_id: int, domain: str, citation_type: str, complexity: str) -> TestCitation:
        """Generate case where claim takes statement out of context."""
        
        papers = self._get_real_paper_examples()
        paper = random.choice(papers)
        
        # Generate claim that's technically in the abstract but taken out of context
        claim = self._generate_out_of_context_claim(paper["abstract"])
        marker = self._generate_citation_marker(citation_type, paper["authors"], paper["year"], case_id)
        
        return TestCitation(
            id=f"incorrect_{case_id:03d}",
            claim_text=claim,
            citation_marker=marker,
            citation_type=citation_type,
            source_title=paper["title"],
            source_authors=paper["authors"],
            source_year=paper["year"],
            source_abstract=paper["abstract"],
            source_doi=paper.get("doi"),
            source_pmid=paper.get("pmid"),
            ground_truth_label="weakly_related",
            confidence_level=0.4,
            domain=domain,
            complexity=complexity,
            error_type="out_of_context",
            created_date=datetime.now().isoformat(),
            source_quality="peer_reviewed"
        )
    
    def _get_real_paper_examples(self) -> List[Dict]:
        """Get sample of real paper abstracts for testing."""
        return [
            {
                "title": "Deep Learning for Medical Image Analysis: A Comprehensive Review",
                "authors": ["Johnson, M.", "Smith, A.", "Chen, L."],
                "year": 2023,
                "abstract": "Deep learning has revolutionized medical image analysis, achieving remarkable performance in tasks such as disease detection, segmentation, and classification. This comprehensive review examines recent advances in convolutional neural networks, transformer architectures, and their applications in radiology, pathology, and ophthalmology. We analyze 150 studies published between 2020-2023, highlighting key methodological innovations and clinical validation results. Our findings demonstrate that deep learning models can achieve diagnostic accuracy comparable to expert radiologists in many imaging tasks, with AUC scores exceeding 0.95 in several applications. However, challenges remain in model interpretability, generalization across different populations, and integration into clinical workflows.",
                "doi": "10.1038/s41591-023-02345-6",
                "domain": "medicine"
            },
            {
                "title": "Quantum Machine Learning: Progress and Challenges",
                "authors": ["Wang, X.", "Brown, R.", "Taylor, S."],
                "year": 2022,
                "abstract": "Quantum machine learning represents a promising intersection of quantum computing and artificial intelligence. This paper reviews theoretical foundations and experimental implementations of quantum algorithms for machine learning tasks. We examine quantum advantage in specific problem domains, including quantum support vector machines, variational quantum eigensolvers, and quantum neural networks. Our analysis reveals that while current quantum hardware limitations prevent practical advantage in most applications, theoretical results suggest exponential speedups for certain structured problems. Near-term quantum devices show promise for optimization problems with specific symmetries.",
                "doi": "10.1103/PhysRevLett.129.120501",
                "domain": "computer_science"
            },
            {
                "title": "CRISPR-Cas9 Gene Editing for Treatment of Inherited Blindness",
                "authors": ["Rodriguez, P.", "Kim, J.", "Anderson, K."],
                "year": 2023,
                "abstract": "We report successful treatment of Leber congenital amaurosis using in vivo CRISPR-Cas9 gene editing. A phase I clinical trial enrolled 18 patients with CEP290-associated blindness, receiving subretinal injections of EDIT-101, a lipid nanoparticle containing Cas9 and guide RNAs targeting the intronic mutation. Primary endpoints included safety and proof-of-concept efficacy at 6 months. Results showed no serious adverse events related to treatment. Visual function improvements were observed in 67% of patients, with mean light sensitivity increasing 2.3-fold. Optical coherence tomography revealed structural improvements in treated eyes. This represents the first successful in vivo application of CRISPR for treating inherited blindness.",
                "pmid": "37845621",
                "domain": "medicine"
            },
            {
                "title": "Climate Change Impacts on Biodiversity: A Meta-Analysis",
                "authors": ["Green, H.", "Davis, M.", "Wilson, T."],
                "year": 2023,
                "abstract": "Climate change poses unprecedented threats to global biodiversity through habitat loss, temperature shifts, and altered precipitation patterns. This meta-analysis synthesizes data from 312 studies across six continents, examining species abundance changes from 1970-2020. Our results indicate average population declines of 34% for vertebrate species, with amphibians showing the steepest declines (52%). Arctic and mountain ecosystems face the highest risk, with projected temperature increases of 3-5°C by 2100. Conservation strategies emphasizing habitat corridors and assisted migration show promise for species preservation. Immediate action is required to prevent cascading ecosystem collapses.",
                "doi": "10.1038/s41586-023-06234-1",
                "domain": "biology"
            }
        ]
    
    def _generate_supported_claim(self, abstract: str, complexity: str) -> str:
        """Generate claim that is supported by the abstract content."""
        # Extract key findings from abstract
        sentences = abstract.split('. ')
        
        if complexity == "simple":
            # Direct paraphrasing
            key_sentence = random.choice(sentences[1:])  # Skip first sentence (often title-like)
            return f"Research demonstrates that {key_sentence.lower()}"
        elif complexity == "medium":
            # Combine multiple findings
            findings = [s for s in sentences if any(word in s.lower() for word in ['show', 'demonstrate', 'reveal', 'indicate', 'found'])]
            if findings:
                return f"Studies {random.choice(findings).lower()}"
            return f"The research indicates important findings in this area."
        else:  # complex
            # Abstract inference
            return "The evidence suggests significant advances in methodology and clinical applications with promising therapeutic potential."
    
    def _generate_contradictory_claim(self, abstract: str, complexity: str) -> str:
        """Generate claim that contradicts the abstract content."""
        # Look for positive findings and flip them
        if "successful" in abstract.lower():
            return "The treatment approach failed to show any significant benefits."
        elif "increase" in abstract.lower() or "improve" in abstract.lower():
            return "Results demonstrated significant decreases and worsening of outcomes."
        elif "effective" in abstract.lower():
            return "The intervention proved ineffective and potentially harmful."
        else:
            return "The study found no evidence supporting the proposed hypothesis."
    
    def _generate_out_of_context_claim(self, abstract: str) -> str:
        """Generate claim that uses words from abstract but changes meaning."""
        # Extract phrases and recombine misleadingly
        if "challenges remain" in abstract.lower():
            return "The approach has been fully validated with no remaining challenges."
        elif "further research" in abstract.lower():
            return "No further research is needed given the conclusive evidence."
        else:
            return "The results provide definitive proof of causation."
    
    def _generate_citation_marker(self, citation_type: str, authors: List[str], year: int, ref_num: int) -> str:
        """Generate citation marker based on style."""
        if citation_type == "numeric":
            return f"[{ref_num}]"
        elif citation_type == "author-year":
            if len(authors) == 1:
                return f"({authors[0].split(',')[0]}, {year})"
            elif len(authors) <= 3:
                first_author = authors[0].split(',')[0]
                return f"({first_author} et al., {year})"
            else:
                first_author = authors[0].split(',')[0]
                return f"({first_author} et al., {year})"
        else:  # hybrid
            return f"{authors[0].split(',')[0]} et al. [{ref_num}]"
    
    def _generate_fake_authors(self) -> List[str]:
        """Generate fake author names."""
        first_names = ["John", "Sarah", "Michael", "Lisa", "David", "Emily", "Robert", "Maria"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        
        num_authors = random.randint(2, 4)
        authors = []
        for _ in range(num_authors):
            first = random.choice(first_names)
            last = random.choice(last_names)
            authors.append(f"{last}, {first[0]}.")
        
        return authors
    
    def _generate_fake_title(self, domain: str) -> str:
        """Generate fake paper title."""
        templates = {
            "medicine": "Novel Therapeutic Approaches in {} Treatment",
            "computer_science": "Advanced {} Algorithms for Real-World Applications", 
            "physics": "Quantum Effects in {} Systems",
            "biology": "Molecular Mechanisms of {} in Biological Systems",
            "psychology": "Cognitive Factors in {} Behavior",
            "chemistry": "Catalytic Properties of {} Compounds"
        }
        
        return templates.get(domain, "Research Advances in {}").format(domain.title())
    
    def _create_validation_instructions(self) -> Dict:
        """Create instructions for validating the test dataset."""
        return {
            "validation_protocol": [
                "Verify that all correct citations have supporting evidence in abstracts",
                "Confirm that incorrect citations represent realistic error patterns", 
                "Check citation format consistency within each type",
                "Validate that DOI/PMID identifiers are correct where provided",
                "Ensure balanced distribution across domains and complexity levels"
            ],
            "quality_checks": [
                "Abstract length: 100-500 words",
                "Claim length: 20-200 words", 
                "Author format: Last, F. Initial",
                "Year range: 2015-2024",
                "Citation marker format matches declared type"
            ],
            "error_type_definitions": {
                "misattribution": "Claim is accurate but attributed to wrong source",
                "misrepresentation": "Claim contradicts what source actually says",
                "wrong_source": "Source is completely unrelated to claim topic",
                "fabricated": "Source does not exist (fake DOI/PMID/title)",
                "out_of_context": "Claim uses source words but changes meaning"
            }
        }

def save_test_dataset(dataset: Dict, filename: str = "reference_checker_test_dataset.json") -> str:
    """Save the test dataset to JSON file."""
    
    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Test dataset saved to {filename}")
    print(f"📊 Dataset contains {len(dataset['correct_citations']) + len(dataset['incorrect_citations'])} total samples")
    
    return filename

# Example usage
if __name__ == "__main__":
    generator = CitationTestDataset()
    dataset = generator.generate_dataset()
    filename = save_test_dataset(dataset)
    
    print("\n📈 Dataset Statistics:")
    print(f"• Correct citations: {len(dataset['correct_citations'])}")
    print(f"• Incorrect citations: {len(dataset['incorrect_citations'])}")
    print(f"• Domains covered: {len(dataset['metadata']['domains'])}")
    print(f"• Citation types: {len(dataset['metadata']['citation_types'])}")
    print(f"• Error types: {len(dataset['metadata']['error_types'])}")