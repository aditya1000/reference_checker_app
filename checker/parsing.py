\
import io
import re
import json
import logging
import regex as rx
from typing import List, Dict, Optional, Tuple
from pypdf import PdfReader

# Enhanced ID patterns
DOI_RX = rx.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', rx.I)
ARXIV_RX = rx.compile(r'arXiv:\s*([0-9]{4}\.[0-9]{4,5})(v\d+)?', rx.I)
PMID_RX = rx.compile(r'\bPMID[:\s]*([0-9]{6,8})\b', rx.I)
PLOS_ONE_RX = rx.compile(r'10\.1371/journal\.pone\.[0-9]+', rx.I)

# Enhanced reference section detection patterns
REF_SECTION_PATTERNS = [
    r'\n\s*(?:references|bibliography|works\s+cited|literature\s+cited|cited\s+references)\s*\n',
    r'\n\s*\d+\.\s*references\s*\n',
    r'\n\s*(?:references|bibliography)\s*$',
    r'\n\s*(?:references|bibliography)\s*\.?\s*\n'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced citation patterns supporting multiple styles
NUMERIC_PATTERNS = [
    rx.compile(r'\[(\d{1,3}(?:,\s*\d{1,3})*(?:-\d{1,3})?)\]'),  # [1], [1,2], [1-3]
    rx.compile(r'\((\d{1,3}(?:,\s*\d{1,3})*(?:-\d{1,3})?)\)'),  # (1), (1,2), (1-3)
    rx.compile(r'(?:^|\s)(\d{1,3})(?=\s|$|[,.;])'),              # Superscript style: word¹
]

AUTHOR_YEAR_PATTERNS = [
    # Standard parenthetical: (Smith, 2020), (Smith & Jones, 2020)
    rx.compile(r'\(([A-Z][A-Za-z\-\'\'\. ]+?)(?:\s*&\s*[A-Z][A-Za-z\-\'\'\. ]+?)?,\s*(20[0-4]\d|19\d{2})[a-z]?\)'),
    
    # Multiple authors with et al.: (Smith et al., 2020)
    rx.compile(r'\(([A-Z][A-Za-z\-\'\'\. ]+?)\s+et\s+al\.?,\s*(20[0-4]\d|19\d{2})[a-z]?\)'),
    
    # Narrative style: Smith et al. (2020), Smith and Jones (2020)
    rx.compile(r'([A-Z][A-Za-z\-\'\'\. ]+?)\s+et\s+al\.?\s*\((20[0-4]\d|19\d{2})[a-z]?\)'),
    rx.compile(r'([A-Z][A-Za-z\-\'\'\. ]+?)(?:\s+and\s+[A-Z][A-Za-z\-\'\'\. ]+?)?\s*\((20[0-4]\d|19\d{2})[a-z]?\)'),
    
    # Multiple citations: (Smith, 2020; Jones, 2021)
    rx.compile(r'\(([A-Z][A-Za-z\-\'\'\. ]+?(?:,\s*20\d{2}[a-z]?(?:;\s*[A-Z][A-Za-z\-\'\'\. ]+?,\s*20\d{2}[a-z]?)*)?)\)'),
]

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Enhanced PDF text extraction with better error handling and cleaning."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                # Basic text cleaning
                text = _clean_extracted_text(text)
                parts.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i}: {e}")
                parts.append("")
        
        full_text = "\n".join(parts)
        logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")
        return full_text
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""

def _clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text."""
    # Remove excessive whitespace
    text = rx.sub(r'\s+', ' ', text)
    # Fix common PDF extraction issues
    text = rx.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between words
    text = rx.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
    text = rx.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
    # Remove page headers/footers patterns
    text = rx.sub(r'\n\s*\d+\s*\n', '\n', text)  # Remove page numbers
    text = rx.sub(r'\n\s*[Pp]age\s+\d+\s*\n', '\n', text)
    return text.strip()

def _detect_references_block(text: str) -> Tuple[str, float]:
    """Enhanced reference section detection with confidence scoring."""
    low = text.lower()
    best_pos = -1
    confidence = 0.0
    
    # Try multiple patterns with different confidence levels
    patterns = [
        (r'\n\s*references\s*\n', 0.9),
        (r'\n\s*bibliography\s*\n', 0.85),
        (r'\n\s*works\s+cited\s*\n', 0.8),
        (r'\n\s*literature\s+cited\s*\n', 0.8),
        (r'\n\s*cited\s+references\s*\n', 0.75),
        (r'\n\s*\d+\.\s*references\s*\n', 0.7),
        (r'\breferences\b', 0.4),  # Lower confidence fallback
    ]
    
    for pattern, conf in patterns:
        matches = list(rx.finditer(pattern, low))
        if matches:
            # Use the last match (most likely to be the references section)
            match = matches[-1]
            pos = match.start()
            
            # Higher confidence if it's in the last third of the document
            if pos > len(text) * 0.67:
                conf += 0.1
                
            if conf > confidence:
                best_pos = pos
                confidence = conf
    
    if best_pos == -1:
        # Fallback: last 25% of document
        logger.warning("No reference section found, using last 25% of document")
        start = int(len(text) * 0.75)
        return text[start:], 0.2
    
    logger.info(f"Reference section detected with confidence: {confidence:.2f}")
    return text[best_pos:], confidence

def _split_reference_entries(block: str) -> List[str]:
    """Enhanced reference entry splitting with multiple strategies."""
    lines = [l.strip() for l in block.splitlines() if l.strip()]
    entries = []
    
    # Strategy 1: Numbered references [1], (1), 1.
    current_entry = []
    entry_patterns = [
        r'^\s*\[\d+\]\s*',  # [1]
        r'^\s*\(\d+\)\s*',  # (1)
        r'^\s*\d+\.\s+',    # 1.
        r'^\s*\d+\s+',      # 1 (with space)
    ]
    
    for line in lines:
        is_new_entry = any(rx.match(pattern, line) for pattern in entry_patterns)
        
        if is_new_entry:
            if current_entry:
                entry_text = " ".join(current_entry)
                if len(entry_text) > 20:  # Minimum length for valid reference
                    entries.append(_clean_reference_entry(entry_text))
            current_entry = [line]
        else:
            # Check if this line looks like a continuation
            if (line and not line[0].islower() and 
                len(line) > 10 and 
                not any(rx.match(pattern, line) for pattern in entry_patterns)):
                # Might be a new entry without numbering
                if current_entry and len(" ".join(current_entry)) > 50:
                    entry_text = " ".join(current_entry)
                    entries.append(_clean_reference_entry(entry_text))
                    current_entry = [line]
                else:
                    current_entry.append(line)
            else:
                current_entry.append(line)
    
    # Add the last entry
    if current_entry:
        entry_text = " ".join(current_entry)
        if len(entry_text) > 20:
            entries.append(_clean_reference_entry(entry_text))
    
    logger.info(f"Extracted {len(entries)} reference entries")
    return entries

def _clean_reference_entry(entry: str) -> str:
    """Clean and normalize a reference entry."""
    # Normalize whitespace
    entry = rx.sub(r'\s+', ' ', entry)
    # Remove leading numbers/brackets
    entry = rx.sub(r'^\s*(\[\d+\]|\(\d+\)|\d+\.?)\s*', '', entry)
    # Remove trailing punctuation
    entry = entry.strip(' .;,')
    return entry

def parse_reference_entries(pdf_text: str, bib_bytes: Optional[bytes] = None) -> List[Dict]:
    refs: List[Dict] = []
    if bib_bytes:
        try:
            content = bib_bytes.decode("utf-8", errors="ignore")
            # naive bibtex parsing: split by @
            for rec in content.split("@"):
                rec = rec.strip()
                if not rec:
                    continue
                entry = {"raw": rec}
                doi = DOI_RX.search(rec)
                arx = ARXIV_RX.search(rec)
                pmid = PMID_RX.search(rec)
                if doi: entry["doi"] = doi.group(1)
                if arx: entry["arxiv_id"] = arx.group(1)
                if pmid: entry["pmid"] = pmid.group(1)
                # author-year guess
                m = rx.search(r'author\s*=\s*[{"]?([^}"]+)', rec, rx.I)
                y = rx.search(r'year\s*=\s*[{"]?(\d{4})', rec, rx.I)
                if m and y:
                    entry["author_key"] = (m.group(1).split("and")[0].split(",")[0].strip(), y.group(1))
                refs.append(entry)
        except Exception:
            pass

    block, confidence = _detect_references_block(pdf_text)
    entries = _split_reference_entries(block)
    for i, e in enumerate(entries, start=1):
        entry = {"raw": e, "index": i}
        doi = DOI_RX.search(e)
        arx = ARXIV_RX.search(e)
        pmid = PMID_RX.search(e)
        if doi: entry["doi"] = doi.group(1)
        if arx: entry["arxiv_id"] = arx.group(1)
        if pmid: entry["pmid"] = pmid.group(1)
        # author-year heuristic
        m = rx.search(r'^([A-Z][A-Za-z\-\’\'\. ]+)', e)
        y = rx.search(r'(20[0-4]\d|19\d{2})', e)
        if m and y:
            entry["author_key"] = (m.group(1).strip(), y.group(1))
        refs.append(entry)
    # de-duplicate by raw content
    seen = set()
    uniq = []
    for r in refs:
        k = r.get("raw","")[:200]
        if k in seen: continue
        seen.add(k)
        uniq.append(r)
    return uniq

def extract_in_text_citations(pdf_text: str, refs: List[Dict]) -> List[Dict]:
    """Enhanced in-text citation extraction with better pattern matching."""
    claims = []
    
    # Build lookup maps
    number_map = {}
    author_year_map = {}
    
    for r in refs:
        idx = r.get("index")
        if idx:
            number_map[str(idx)] = r
        
        # Build author-year lookup
        author_key = r.get("author_key")
        if author_key and isinstance(author_key, tuple) and len(author_key) == 2:
            author_year_map[(author_key[0].lower(), author_key[1])] = r
    
    # Split into sentences with better boundaries
    sentences = _extract_sentences_with_citations(pdf_text)
    
    for sentence in sentences:
        # Extract numeric citations
        claims.extend(_extract_numeric_citations(sentence, number_map))
        
        # Extract author-year citations
        claims.extend(_extract_author_year_citations(sentence, refs, author_year_map))
    
    # Remove duplicates and very short claims
    unique_claims = []
    seen_claims = set()
    
    for claim in claims:
        claim_key = (claim["claim_text"], claim["marker"])
        if (claim_key not in seen_claims and 
            len(claim["claim_text"]) > 30):  # Minimum sentence length
            seen_claims.add(claim_key)
            unique_claims.append(claim)
    
    logger.info(f"Extracted {len(unique_claims)} unique citation claims")
    return unique_claims

def _extract_sentences_with_citations(text: str) -> List[str]:
    """Extract sentences that likely contain citations."""
    # Clean and normalize text
    text = rx.sub(r'\s+', ' ', text)
    
    # Split into sentences
    sentences = rx.split(r'(?<=[\.\?\!])\s+', text)
    
    # Filter for sentences containing citation markers
    citation_sentences = []
    citation_indicators = [r'\[\d+\]', r'\(\w+,?\s*\d{4}\)', r'\w+\s+et\s+al\.?\s*\(']
    
    for sentence in sentences:
        sentence = sentence.strip()
        if (len(sentence) > 20 and 
            any(rx.search(pattern, sentence) for pattern in citation_indicators)):
            citation_sentences.append(sentence)
    
    return citation_sentences

def _extract_numeric_citations(sentence: str, number_map: Dict) -> List[Dict]:
    """Extract numeric citation patterns from sentence."""
    claims = []
    
    for pattern in NUMERIC_PATTERNS:
        for match in pattern.finditer(sentence):
            citation_text = match.group(1)
            
            # Handle ranges and multiple citations
            numbers = _parse_citation_numbers(citation_text)
            
            for num in numbers:
                ref = number_map.get(str(num), {})
                claims.append({
                    "marker": f"[{num}]",
                    "claim_text": sentence.strip(),
                    "ref_index": ref.get("index"),
                    "ref_best_id": ref.get("best_id") if ref else None,
                    "citation_type": "numeric"
                })
    
    return claims

def _extract_author_year_citations(sentence: str, refs: List[Dict], author_year_map: Dict) -> List[Dict]:
    """Extract author-year citation patterns from sentence."""
    claims = []
    
    for pattern in AUTHOR_YEAR_PATTERNS:
        for match in pattern.finditer(sentence):
            author_text = match.group(1)
            year = match.group(2) if len(match.groups()) > 1 else None
            
            # Find matching reference
            ref_best = _match_author_year_reference(author_text, year, refs, author_year_map)
            
            claims.append({
                "marker": match.group(0),
                "claim_text": sentence.strip(),
                "ref_index": ref_best.get("index") if ref_best else None,
                "ref_best_id": ref_best.get("best_id") if ref_best else None,
                "citation_type": "author-year"
            })
    
    return claims

def _parse_citation_numbers(citation_text: str) -> List[int]:
    """Parse citation numbers from text like '1,2,5-7' -> [1,2,5,6,7]."""
    numbers = []
    parts = citation_text.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle ranges
            try:
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                numbers.extend(range(start_num, end_num + 1))
            except ValueError:
                continue
        else:
            # Single number
            try:
                numbers.append(int(part))
            except ValueError:
                continue
    
    return numbers

def _match_author_year_reference(author_text: str, year: str, refs: List[Dict], author_year_map: Dict) -> Dict:
    """Match author-year citation to reference using fuzzy matching."""
    # Try exact match first
    author_clean = author_text.lower().strip()
    if year and (author_clean, year) in author_year_map:
        return author_year_map[(author_clean, year)]
    
    # Fuzzy matching
    best_match = {}
    best_score = 0
    
    for ref in refs:
        author_key = ref.get("author_key")
        if not author_key or not isinstance(author_key, tuple):
            continue
            
        ref_author, ref_year = author_key
        
        # Year must match if provided
        if year and year != ref_year:
            continue
        
        # Calculate author similarity
        score = _calculate_author_similarity(author_text, ref_author)
        
        if score > best_score and score > 0.7:  # Minimum similarity threshold
            best_score = score
            best_match = ref
    
    return best_match

def _calculate_author_similarity(author1: str, author2: str) -> float:
    """Calculate similarity between author names."""
    # Simple similarity based on common words and character overlap
    words1 = set(author1.lower().split())
    words2 = set(author2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity for words
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0
