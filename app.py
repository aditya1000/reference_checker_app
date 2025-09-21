\
import os
import io
import json
import time
import tempfile
import regex as re
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from checker.parsing import extract_pdf_text, parse_reference_entries, extract_in_text_citations
from checker.retrieval import resolve_reference, fetch_source_text
from checker.verification import build_index, best_support_for_claim

st.set_page_config(page_title="Reference Checker", layout="wide")

st.title("🔎 Reference Checker (Local)")

with st.sidebar:
    st.header("Options")
    run_claim_check = st.checkbox("Run claim support check (beta)", value=True, help="Find sentences around citations and test if the cited source supports them.")
    nli_enabled = st.checkbox("Use NLI (requires model download)", value=True, help="Entailment via roberta-large-mnli or bart-large-mnli.")
    topk = st.slider("Top-K source sentences", min_value=3, max_value=20, value=7, step=1)
    st.markdown("---")
    st.caption("Tip: Add a `.env` file with emails for polite API usage (Unpaywall, Crossref, NCBI).")

st.write("Upload a manuscript PDF. Optionally, add a BibTeX file of references to improve matching.")

col1, col2 = st.columns(2)
pdf_file = col1.file_uploader("Manuscript PDF", type=["pdf"])
bib_file = col2.file_uploader("Optional: References BibTeX", type=["bib","bibtex"])

if st.button("Run Checker", disabled=pdf_file is None):
    if not pdf_file:
        st.error("Please upload a PDF first.")
        st.stop()

    with st.spinner("Extracting text from PDF..."):
        pdf_bytes = pdf_file.read()
        pdf_text = extract_pdf_text(pdf_bytes)

    st.success(f"Extracted {len(pdf_text):,} characters of text")

    with st.spinner("Parsing references..."):
        refs = parse_reference_entries(pdf_text, bib_bytes=bib_file.read() if bib_file else None)
    st.write(f"Found **{len(refs)}** reference candidates")

    # Resolve references (existence check)
    resolved_rows = []
    progress = st.progress(0.0, text="Resolving references via DOI / PubMed / arXiv...")
    for i, r in enumerate(refs, start=1):
        try:
            meta = resolve_reference(r)
        except Exception as e:
            meta = {"status":"error", "error":str(e), **r}
        resolved_rows.append(meta)
        progress.progress(i/len(refs), text=f"Resolved {i}/{len(refs)}")

    df_refs = pd.DataFrame(resolved_rows)
    st.subheader("Reference existence & metadata")
    st.dataframe(df_refs.fillna(""), use_container_width=True)
    csv = df_refs.to_csv(index=False).encode("utf-8")
    st.download_button("Download references CSV", data=csv, file_name="reference_existence.csv", mime="text/csv")

    results = []
    if run_claim_check:
        with st.spinner("Extracting in‑text citation claims..."):
            claims = extract_in_text_citations(pdf_text, refs=df_refs.to_dict(orient="records"))

        st.write(f"Found **{len(claims)}** claim sentences containing citation markers")

        # Build per-source indices and check claims
        with st.spinner("Fetching source text & building indices..."):
            source_cache = {}
            for ref in resolved_rows:
                if ref.get("best_id"):
                    text = fetch_source_text(ref)
                    source_cache[ref["best_id"]] = build_index(text)

        st.subheader("Claim support (beta)")
        bar = st.progress(0.0, text="Verifying claims against cited sources...")
        for i, c in enumerate(claims, start=1):
            ref_id = c.get("ref_best_id")
            idx = source_cache.get(ref_id)
            verdict = {"label":"not_checked", "score":0.0, "evidence":[]}
            if idx:
                verdict = best_support_for_claim(c["claim_text"], idx, use_nli=nli_enabled, topk=topk)
            row = {**c, **verdict}
            results.append(row)
            bar.progress(i/len(claims), text=f"Verified {i}/{len(claims)}")

        df_claims = pd.DataFrame(results)
        st.dataframe(df_claims.fillna(""), use_container_width=True)

        st.download_button(
            "Download claim support CSV",
            data=df_claims.to_csv(index=False).encode("utf-8"),
            file_name="claim_support.csv",
            mime="text/csv",
        )

    st.success("Done.")
