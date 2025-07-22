import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

def load_outlines(folder):
    outlines = []
    for path in Path(folder).glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "outline" in data:
                title = data.get("title", "Untitled")
                for block in data["outline"]:
                    outlines.append({
                        "document": path.stem,
                        "title": title,
                        **block
                    })
    return pd.DataFrame(outlines)

def chunk_sections(df):
    chunks = []
    current_chunk = None
    for i, row in df.iterrows():
        if row["level"] == "H1":
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = {
                "document": row["document"],
                "page": row["page"],
                "section_title": row["text"],
                "chunk_text": row["text"]
            }
        elif current_chunk:
            current_chunk["chunk_text"] += " " + row["text"]
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Persona-Driven RAG Pipeline")
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

    print("\n[1/5] Loading outlines...")
    outline_df = load_outlines(OUTLINE_DIR)
    if outline_df.empty:
        print("No outlines found in output/. Run the extraction pipeline first.")
        sys.exit(1)
    print(f"Loaded {len(outline_df)} heading blocks from {OUTLINE_DIR}")

    print("[2/5] Chunking sections by H1 headings...")
    chunks = chunk_sections(outline_df)
    print(f"Chunked into {len(chunks)} sections")

    print("[3/5] Loading embedding model (paraphrase-MiniLM-L6-v2)...")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device="cpu")

    query = f"{persona} needs to: {job}"
    print(f"[4/5] Embedding query: {query}")
    query_emb = model.encode(query, convert_to_tensor=True)

    print("Embedding and scoring all sections...")
    for c in chunks:
        c["embedding"] = model.encode(c["chunk_text"], convert_to_tensor=True)
        c["similarity"] = util.cos_sim(query_emb, c["embedding"]).item()

    ranked_chunks = sorted(chunks, key=lambda x: x["similarity"], reverse=True)

    print(f"[5/5] Assembling output JSON and saving top {top_n} results...")
    final = {
        "metadata": {
            "documents": list(sorted(set([c["document"] for c in ranked_chunks]))),
            "persona": persona,
            "job_to_be_done": job,
            "processed_at": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": c["document"],
                "page": c["page"],
                "section_title": c["section_title"],
                "importance_rank": i + 1
            }
            for i, c in enumerate(ranked_chunks[:top_n])
        ],
        "subsection_analysis": [
            {
                "document": c["document"],
                "page": c["page"],
                "section_title": c["section_title"],
                "refined_text": c["chunk_text"]
            }
            for c in ranked_chunks[:top_n]
        ]
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"✅ RAG output saved to: {output_path}")

if __name__ == "__main__":
    main() 