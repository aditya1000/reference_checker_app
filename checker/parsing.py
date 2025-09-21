\
import io
import re
import json
import regex as rx
from typing import List, Dict, Optional
from pypdf import PdfReader

DOI_RX = rx.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', rx.I)
ARXIV_RX = rx.compile(r'arXiv:\s*([0-9]{4}\.[0-9]{4,5})(v\d+)?', rx.I)
PMID_RX = rx.compile(r'\bPMID[:\s]*([0-9]{6,8})\b', rx.I)

# Author–year patterns e.g., (Smith, 2020) or Smith et al. (2020)
AUTHORYEAR_PAREN_RX = rx.compile(r'\(([A-Z][A-Za-z\-\’\'\. ]+?),\s*(20[0-4]\d|19\d{2})\)')
AUTHORYEAR_INFIX_RX = rx.compile(r'([A-Z][A-Za-z\-\’\'\. ]+?)\s+et al\.\s*\((20[0-4]\d|19\d{2})\)')

NUMERIC_CIT_RX = rx.compile(r'\[(\d{1,3})\]')

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    text = "\n".join(parts)
    return text

def _detect_references_block(text: str) -> str:
    anchors = ["\nreferences\n", "\nreferences\r", "\nbibliography\n", "\nworks cited\n"]
    low = text.lower()
    pos = -1
    for a in anchors:
        pos = low.rfind(a)
        if pos != -1:
            break
    if pos == -1:
        # Fallback: last 25% of document
        start = int(len(text) * 0.75)
        return text[start:]
    return text[pos:]

def _split_reference_entries(block: str) -> List[str]:
    # Split on double newlines or lines starting like [1], 1. , etc.
    lines = [l.strip() for l in block.splitlines()]
    entries = []
    cur = []
    for ln in lines:
        if rx.match(r'^\s*(\[\d+\]|^\d+\.\s)', ln):
            if cur:
                entries.append(" ".join(cur))
                cur = []
            cur.append(ln)
        elif ln == "" and cur:
            entries.append(" ".join(cur))
            cur = []
        else:
            cur.append(ln)
    if cur:
        entries.append(" ".join(cur))
    # Clean long entries
    entries = [rx.sub(r'\s+', ' ', e).strip(' .;') for e in entries if len(e) > 5]
    return entries

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

    block = _detect_references_block(pdf_text)
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
    claims = []
    # Build a simple lookup map by number and author-year
    number_map = {}
    for r in refs:
        idx = r.get("index")
        if idx:
            number_map[str(idx)] = r
    # For each citation marker, capture its sentence
    sentences = rx.split(r'(?<=[\.\?\!])\s+', rx.sub(r'\s+', ' ', pdf_text))
    for s in sentences:
        # numeric
        for m in NUMERIC_CIT_RX.finditer(s):
            n = m.group(1)
            ref = number_map.get(n, {})
            claims.append({
                "marker": f"[{n}]",
                "claim_text": s.strip(),
                "ref_index": ref.get("index"),
                "ref_best_id": ref.get("best_id") if ref else None
            })
        # author-year (parenthetical)
        for m in AUTHORYEAR_PAREN_RX.finditer(s):
            author = m.group(1).split()[:1][0]
            year = m.group(2)
            # fuzzy pick: first matching author_key
            ref_best = None
            for r in refs:
                ak = r.get("author_key")
                if ak and (author.lower() in ak[0].lower()) and (year == ak[1]):
                    ref_best = r
                    break
            claims.append({
                "marker": f"({m.group(1)}, {year})",
                "claim_text": s.strip(),
                "ref_index": ref_best.get("index") if ref_best else None,
                "ref_best_id": ref_best.get("best_id") if ref_best else None
            })
    return claims
