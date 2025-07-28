<h1 align="center">🧠 Persona-Driven Document Intelligence</h1>
<h3 align="center">Adobe India Hackathon 2025 – Challenge 1B</h3>


---

## 📘 Project Overview

This repository contains our solution for **Challenge 1B** of the Adobe India Hackathon 2025. The goal is to extract *persona-relevant* content from **unstructured multilingual PDFs**, structuring the data into a meaningful format for downstream use.

Our system is:
- 🔍 **Context-Aware**: Uses semantic search instead of keyword matching
- 🌐 **Multilingual**: Supports 50+ languages using transformer embeddings
- 🧩 **Modular**: Clean structure with preprocessing, chunking, and retrieval phases
- 📤 **Output Ready**: Structured JSON output with headings and relevant content

A detailed breakdown of our approach is available in [`approach_explanation.md`](./approach_explanation.md).

---

## ⚙️ How It Works (In Short)

1. **PDF Ingestion** using `LangChain` and `PyPDFLoader`
2. **Chunking** with boundary-preserving token split logic
3. **Semantic Embedding** via `paraphrase-multilingual-MiniLM-L12-v2`
4. **Vector Search** with ChromaDB to retrieve relevant chunks
5. **Heuristic Heading Detection** and clean JSON formatting

---

## 🐳 Docker Instructions

### 1️⃣ Build the Image

```bash
docker build --platform linux/amd64 -t pdf-intelligence-engine:round1b .
