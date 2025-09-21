\
import os
import time
import json
import base64
import regex as rx
import requests
from typing import Dict, Optional

DOI_RX = rx.compile(r'(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', rx.I)
ARXIV_RX = rx.compile(r'([0-9]{4}\.[0-9]{4,5})')
PMID_RX = rx.compile(r'\b([0-9]{6,8})\b')

def polite_headers():
    hdrs = {"User-Agent": "LocalReferenceChecker/0.1"}
    cm = os.getenv("CROSSREF_MAILTO")
    if cm:
        hdrs["mailto"] = cm
    return hdrs

def resolve_reference(ref: Dict) -> Dict:
    """Return dict with best_id, type, title, authors, year, doi/pmid/arxiv."""
    out = {**ref}
    doi = ref.get("doi")
    pmid = ref.get("pmid")
    arx = ref.get("arxiv_id")

    # Try to find DOI via Crossref if missing
    if not any([doi, pmid, arx]):
        q = ref.get("raw","")[:300]
        try:
            r = requests.get("https://api.crossref.org/works", params={"query.bibliographic": q, "rows":1}, headers=polite_headers(), timeout=20)
            if r.ok:
                items = r.json().get("message",{}).get("items",[])
                if items:
                    doi = items[0].get("DOI")
        except Exception:
            pass

    # Prefer DOI; then PMID; then arXiv
    best_id = None
    id_type = None
    if doi:
        # Verify DOI existence
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", headers=polite_headers(), timeout=20)
            if r.ok:
                data = r.json().get("message", {})
                out.update({
                    "title": "; ".join(data.get("title", [])[:1]) or None,
                    "year": (data.get("issued",{}).get("date-parts",[[None]])[0][0]),
                    "journal": data.get("container-title", [None])[0],
                    "publisher": data.get("publisher"),
                    "status": "exists"
                })
                best_id = doi
                id_type = "doi"
        except Exception:
            pass

    if not best_id and pmid:
        try:
            e = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                             params={"db":"pubmed","id":pmid,"retmode":"json","email":os.getenv("NCBI_EMAIL","")}, timeout=20)
            if e.ok:
                data = e.json().get("result",{}).get(pmid,{})
                out.update({
                    "title": data.get("title"),
                    "year": int(data.get("pubdate", "0")[:4]) if data.get("pubdate") else None,
                    "journal": data.get("fulljournalname"),
                    "publisher": "NCBI",
                    "status": "exists"
                })
                best_id = pmid
                id_type = "pmid"
        except Exception:
            pass

    if not best_id and arx:
        try:
            r = requests.get("https://export.arxiv.org/api/query", params={"id_list": arx}, timeout=20)
            if r.ok:
                out.update({"title": f"arXiv:{arx}", "year": None, "journal": "arXiv", "publisher":"arXiv", "status":"exists"})
                best_id = arx
                id_type = "arxiv"
        except Exception:
            pass

    out["best_id"] = best_id
    out["id_type"] = id_type
    out["doi"] = doi
    out["pmid"] = pmid
    out["arxiv_id"] = arx
    if not best_id:
        out["status"] = out.get("status") or "not_found"
    return out

def fetch_source_text(ref_meta: Dict) -> str:
    """Fetch text for the reference: abstract or OA full text if available."""
    id_type = ref_meta.get("id_type")
    best_id = ref_meta.get("best_id")
    # DOI path: try Crossref abstract, then Unpaywall OA link
    if id_type == "doi":
        # Try Crossref (abstracts are rare)
        try:
            r = requests.get(f"https://api.crossref.org/works/{best_id}", headers=polite_headers(), timeout=20)
            if r.ok:
                msg = r.json().get("message",{})
                abstract = msg.get("abstract")
                if abstract:
                    return rx.sub(r'<[^>]+>', ' ', abstract)
        except Exception:
            pass
        # Try Unpaywall for OA
        email = os.getenv("UNPAYWALL_EMAIL")
        if email:
            try:
                u = requests.get(f"https://api.unpaywall.org/v2/{best_id}", params={"email":email}, timeout=20)
                if u.ok:
                    data = u.json()
                    url = (data.get("best_oa_location") or {}).get("url_for_pdf") or (data.get("best_oa_location") or {}).get("url")
                    if url:
                        txt = _download_and_extract_text(url)
                        if txt:
                            return txt
            except Exception:
                pass
        return ""

    if id_type == "pmid":
        try:
            f = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                             params={"db":"pubmed","id":best_id,"retmode":"xml","email":os.getenv("NCBI_EMAIL","")}, timeout=20)
            if f.ok:
                return f.text
        except Exception:
            pass
        return ""

    if id_type == "arxiv":
        try:
            # Get the abstract via arXiv API (Atom)
            r = requests.get("https://export.arxiv.org/api/query", params={"id_list": best_id}, timeout=20)
            if r.ok:
                return r.text
        except Exception:
            pass
        return ""
    return ""

def _download_and_extract_text(url: str) -> str:
    # Very simple handler: try html then pdf
    try:
        r = requests.get(url, timeout=25)
        ct = r.headers.get("Content-Type","")
        if "text/html" in ct:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "lxml")
                return soup.get_text(" ", strip=True)
            except Exception:
                return ""
        if "application/pdf" in ct or url.lower().endswith(".pdf"):
            from pypdf import PdfReader
            import io
            pdf = PdfReader(io.BytesIO(r.content))
            parts = []
            for p in pdf.pages:
                try:
                    parts.append(p.extract_text() or "")
                except Exception:
                    pass
            return "\n".join(parts)
    except Exception:
        return ""
    return ""
