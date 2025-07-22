import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
from collections import defaultdict

def load_outlines(outline_dir, allowed_docs):
    outlines = {}
    for path in Path(outline_dir).glob("*.json"):
        doc_name = path.stem
        if f"{doc_name}.pdf" not in allowed_docs:
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "outline" in data:
                outlines[doc_name] = data["outline"]
    return outlines

def load_text_blocks(blocks_path, allowed_docs):
    with open(blocks_path, "r", encoding="utf-8") as f:
        blocks = json.load(f)
    if blocks and isinstance(blocks[0], dict) and "document" in blocks[0]:
        return [b for b in blocks if f"{b['document']}.pdf" in allowed_docs]
    return blocks

def build_chunks_for_doc(doc_name, outline, text_blocks):
    blocks_by_page = defaultdict(list)
    for block in text_blocks:
        blocks_by_page[block["page"]].append(block)
    outline_sorted = sorted(outline, key=lambda h: (h["page"], h["level"]))
    chunks = []
    for idx, heading in enumerate(outline):
        start_page = heading["page"]
        start_title = heading["text"].strip()
        start_level = heading["level"]
        next_idx = idx + 1
        while next_idx < len(outline):
            if outline[next_idx]["level"] <= start_level:
                break
            next_idx += 1
        end_page = outline[next_idx]["page"] if next_idx < len(outline) else None
        chunk_text = []
        heading_found = False
        for block in text_blocks:
            if block["page"] < start_page:
                continue
            if end_page is not None and block["page"] >= end_page:
                break
            if block["page"] == start_page:
                if block["text"].strip() == start_title:
                    heading_found = True
                    continue
                if not heading_found:
                    continue
            chunk_text.append(block["text"])
        if not chunk_text:
            chunk_text = [start_title]
        chunks.append({
            "document": doc_name,
            "section_title": start_title,
            "page_number": start_page,
            "heading_level": start_level,
            "chunk_text": " ".join(chunk_text).strip()
        })
    return chunks

def get_level_num(level):
    if isinstance(level, str) and level.startswith("H"):
        try:
            return int(level[1:])
        except:
            return 99
    return 99

def summarize_text(text, max_sentences=3):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return " ".join(sentences[:max_sentences]).strip()

def best_section_title(chunk):
    heading = chunk["section_title"].strip()
    text = chunk["chunk_text"].strip()
    if text and len(text) > len(heading) + 10:
        sentences = re.split(r'(?<=[.!?]) +', text)
        first_sentence = sentences[0].strip()
        if first_sentence.lower() != heading.lower() and len(first_sentence.split()) > 3:
            return first_sentence
    return heading

def analyze_subsection(chunk, persona, job, model, query_emb, n_sentences=3):
    text = chunk["chunk_text"].strip()
    if not text:
        return chunk["section_title"].strip()
    # Split into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) == 1:
        return sentences[0]
    # Embed all sentences in batch
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    # Compute similarity to query
    sims = util.cos_sim(query_emb, sent_embs)[0].cpu().numpy()
    # Get top n_sentences by similarity, preserve order
    top_idx = np.argsort(sims)[-n_sentences:][::-1]
    top_idx = sorted(top_idx)  # preserve original order in text
    selected = [sentences[i] for i in top_idx]
    # Optionally prepend a persona/job-aware intro
    intro = f"For {persona.lower()}, {job.lower()}: " if persona and job else ""
    return intro + " ".join(selected).strip()

def main():
    parser = argparse.ArgumentParser(description="Advanced Persona-Driven RAG Pipeline")
    parser.add_argument("--persona", type=str, help="Persona description")
    parser.add_argument("--job", type=str, help="Job to be done")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top sections to return")
    parser.add_argument("--output", type=str, default="output/rag_output.json", help="Output JSON path")
    args = parser.parse_args()

    persona = args.persona or input("Enter persona: ")
    job = args.job or input("Enter job to be done: ")
    top_n = args.top_n
    output_path = args.output

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    OUTLINE_DIR = PROJECT_ROOT / "output"
    BLOCKS_PATH = PROJECT_ROOT / "data" / "processed" / "pdf_text_blocks.json"
    PDFS1B_DIR = PROJECT_ROOT / "sample_dataset" / "pdfs-1b"

    print("\n[1/7] Listing allowed PDFs from sample_dataset/pdfs-1b/ ...")
    allowed_docs = []
    for pdf in sorted(PDFS1B_DIR.glob("*.pdf")):
        allowed_docs.append(pdf.name)
    if not allowed_docs:
        print("No PDFs found in sample_dataset/pdfs-1b/.")
        sys.exit(1)
    print(f"Allowed input documents: {allowed_docs}")

    print("[2/7] Loading outlines...")
    outlines = load_outlines(OUTLINE_DIR, set(allowed_docs))
    if not outlines:
        print("No outlines found for the allowed PDFs. Run the extraction pipeline first.")
        sys.exit(1)
    print(f"Loaded outlines for {len(outlines)} documents.")

    print("[3/7] Loading text blocks...")
    text_blocks = load_text_blocks(BLOCKS_PATH, set(allowed_docs))

    print("[4/7] Building advanced chunks for all documents...")
    all_chunks = []
    for doc_name, outline in outlines.items():
        doc_chunks = build_chunks_for_doc(doc_name, outline, text_blocks)
        all_chunks.extend(doc_chunks)
    print(f"Built {len(all_chunks)} chunks.")

    print("[5/7] Loading embedding model (paraphrase-MiniLM-L6-v2)...")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")

    query = f"{persona} needs to: {job}"
    print(f"[6/7] Embedding query and all chunks...")
    query_emb = model.encode(query, convert_to_tensor=True)
    for c in all_chunks:
        c["embedding"] = model.encode(c["chunk_text"], convert_to_tensor=True)
        c["similarity"] = util.cos_sim(query_emb, c["embedding"]).item()

    ranked_chunks = sorted(all_chunks, key=lambda x: x["similarity"], reverse=True)
    top_chunks = ranked_chunks[:top_n]

    print(f"[7/7] Assembling output JSON and saving top {top_n} results...")
    subsection_analysis = []
    for chunk in top_chunks:
        doc_chunks = [c for c in all_chunks if c["document"] == chunk["document"]]
        this_level = get_level_num(chunk["heading_level"])
        idx = doc_chunks.index(chunk)
        sub_chunks = []
        for c in doc_chunks[idx+1:]:
            if get_level_num(c["heading_level"]) > this_level:
                sub_chunks.append(c)
            else:
                break
        if sub_chunks:
            best_sub = max(sub_chunks, key=lambda x: x["similarity"])
            refined_text = analyze_subsection(best_sub, persona, job, model, query_emb, n_sentences=3)
            page_number = best_sub["page_number"]
        else:
            refined_text = analyze_subsection(chunk, persona, job, model, query_emb, n_sentences=3)
            page_number = chunk["page_number"]
        subsection_analysis.append({
            "document": chunk["document"] + ".pdf",
            "refined_text": refined_text,
            "page_number": page_number
        })

    final = {
        "metadata": {
            "input_documents": allowed_docs,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": chunk["document"] + ".pdf",
                "section_title": best_section_title(chunk),
                "importance_rank": i + 1,
                "page_number": chunk["page_number"]
            }
            for i, chunk in enumerate(top_chunks)
        ],
        "subsection_analysis": subsection_analysis
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"✅ RAG output saved to: {output_path}")

if __name__ == "__main__":
    main() 