# Reference Checker (Local App)

A local tool to verify references in a manuscript and flag potential mis-citations.

**What it does**
1. **Extracts references** from an uploaded PDF (or accepts a BibTeX file).
2. **Verifies existence** of each bibliographic entry via DOI / PMID / arXiv lookup (Crossref, PubMed, arXiv).
3. **Checks claim support (beta):** finds in‑text citation sentences, retrieves the cited source (abstract or open access text), and runs semantic matching + NLI to label as **Supported / Not Found / Contradicted (risk)**.
4. Produces an **interactive report** plus CSV/JSON exports.

> ⚠️ Claim checking is best‑effort. It uses abstracts or OA text when available and may be limited by PDF parsing and access constraints.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Optional environment variables
Create a `.env` file in this folder to improve retrieval:
```
UNPAYWALL_EMAIL=your_email@example.com   # for Unpaywall OA lookups
NCBI_EMAIL=your_email@example.com        # for PubMed polite usage
CROSSREF_MAILTO=your_email@example.com   # for Crossref polite usage
```

## How it works (high level)
- **Parsing:** finds the "References" section in the PDF; extracts DOIs/PMIDs/arXiv IDs; queries Crossref/OpenAlex/PubMed when missing.
- **Existence check:** resolves identifiers and basic metadata.
- **Claim extraction:** detects in‑text markers like `[12]`, `(Smith, 2020)`, `Smith et al. (2020)`; grabs the surrounding sentence(s) as the claim.
- **Support check:** embeds claim and source sentences with Sentence Transformers and runs MNLI when available. Labels each claim–source pair and shows top matching snippets.
- **Outputs:** interactive table, per‑citation details, and downloadable CSV/JSON.

## Notes
- PDF parsing is noisy; for best results, upload clean PDFs or add a BibTeX file of the references.
- Heuristics support common citation styles (numeric, author‑year). Edge cases may need manual review.
- The NLI model (roberta‑large‑mnli or bart‑large‑mnli) downloads on first run (~1.3GB). If unavailable, the app falls back to similarity‑only heuristics.
