# Adobe Hackathon - Challenge 1B  
## 🧠 Approach Explanation for Persona-Driven Document Intelligence

## 🧩 Overview
This project implements an intelligent, multilingual PDF document processing pipeline that extracts and ranks relevant sections from unstructured documents based on a user’s **persona** and **job description**. The system is built to be domain-agnostic, scalable, and designed for efficient and fast processing in recruitment, personalization, and document analysis.

---

## ⚙️ Methodology

### 1. 📥 Document Ingestion and Preprocessing
All PDFs placed in the `/input` directory are automatically processed. The system uses **LangChain’s PyPDFLoader** to extract raw text from each document.

To maintain semantic coherence, the text is split using `RecursiveCharacterTextSplitter` with:
- `chunk_size = 1000`
- `chunk_overlap = 200`  
This ensures boundary-sensitive content (e.g., long paragraphs or headings) is preserved across splits.

---

### 2. 🌐 Semantic Embedding and Retrieval
Each chunk is embedded using the **`paraphrase-multilingual-MiniLM-L12-v2`** model from HuggingFace. This supports over 50 languages, enabling robust **language-agnostic** processing.

Embeddings are stored in a **Chroma vector store**, enabling fast semantic similarity retrieval. The **query vector** is created by combining the **persona** and **job description**, representing the user's information need.

---

### 3. 🎯 Relevance-Based Chunk Selection
The system retrieves the top `k` most relevant chunks based on cosine similarity. Duplicates and near-duplicates are filtered out to ensure diversity in extracted content.

---

### 4. 🧹 Heading Extraction and Structuring
For each selected chunk:
- A **heuristic-based heading** is derived from the first non-empty line (stripped and capitalized).
- The chunk body is cleaned of redundant characters and formatting issues.
This provides a readable and organized final structure.

---

### 5. 📤 Output Generation
The final result is exported as a structured JSON file containing:
- 📌 **Metadata**: Input filenames, persona, job role, and timestamp
- 🧩 **Extracted Sections**: Headings and cleaned, relevant content
- 🧠 **Subsection Analysis**: Subsection-wise breakdowns and insight extraction

The output is fully compatible with downstream applications such as NLP pipelines, custom dashboards, or LLM-based retrieval systems.

---

## 🌟 Key Technical Highlights

- ✅ **Multilingual Support**: Supports global datasets across languages
- ✅ **Semantic Search > Keyword Matching**
- ✅ **Domain-Agnostic Architecture**
- ✅ **Fast Local Inference** using HuggingFace + Chroma
- ✅ **Modular & Lightweight Codebase**
- ✅ **Structured JSON Output** for clean integration

---

## 🧠 Summary
This pipeline transforms unstructured PDFs into structured, persona-relevant summaries. It combines **semantic intelligence**, **multilingual embedding**, and **intuitive document structuring** to deliver precise and useful insights. Its versatility makes it applicable across domains and industries — from tech hiring to academic screening and knowledge mining. 🧾🔍
