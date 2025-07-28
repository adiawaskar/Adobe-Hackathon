# 🧠 Approach Explanation for Persona-Driven Document Intelligence

## 📌 Objective

To build an intelligent document analysis system that extracts and prioritizes the most relevant sections from a collection of PDFs, tailored to a specific **persona** and their **job-to-be-done**. The system must be **multilingual**, **domain-agnostic**, and **fallback-safe** in diverse real-world scenarios.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **LangChain** – for pipeline orchestration
- **OpenAI GPT-4 / GPT-3.5** – for natural language understanding
- **pdfplumber** – for PDF text extraction
- **scikit-learn** – for sentence embeddings & similarity
- **NumPy** – for vector manipulation
- **Pydantic** – for schema validation

---

## 📂 Input Specification

- Directory: `./input`
- Files: 3–10 PDF files related to a common domain
- `persona` (str): Role of the user (e.g., “PhD Researcher”)
- `job` (str): Specific task they want to accomplish

---

## 🧭 Step-by-Step Approach

### 1. **PDF Preprocessing**
- Each PDF is parsed using `pdfplumber`.
- Text is extracted page-wise and split into **sections** using regex-based title heuristics.

### 2. **Section Heading Detection**
- Headings are identified using:
  - Font size and layout (where possible)
  - Text patterns (capital letters, bold, colons)
  - First-level sentence segmentation

### 3. **Multilingual Support**
- All extracted text chunks are automatically detected and translated (if needed) using **OpenAI function calling**.
- Translation only occurs if the chunk's language mismatches the persona/job language.

### 4. **Relevance Scoring**
- Each chunk is embedded using OpenAI Embeddings.
- A relevance score is calculated via **cosine similarity** between:
  - The embedding of `(persona + job)`
  - The embedding of each section

### 5. **Ranking and Filtering**
- Top relevant chunks (by cosine score) are ranked.
- Redundant or low-content sections are filtered using a **minimum token threshold** and repetition penalty.

### 6. **Structured Output**
- Output format:
```json
{
  "metadata": {
    "input_documents": [...],
    "persona": "...",
    "job": "...",
    "processing_timestamp": "..."
  },
  "extracted_sections": [
    {
      "document": "...",
      "page_number": ...,
      "section_title": "...",
      "importance_rank": ...
    },
    ...
  ],
  "subsection_analysis": [
    {
      "document": "...",
      "refined_text": "...",
      "page_number": ...
    },
    ...
  ]
}
