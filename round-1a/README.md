# Adobe Hackathon - Challenge 1A
## 🧠 Structured Heading Extraction from Unstructured PDFs

## 🧩 Overview

This project implements a **precise and intelligent document structuring engine** that extracts hierarchical **headings (H1, H2, H3)** and the **document title** from unstructured PDF files. It's optimized for real-world corporate documents and reports where **formatting inconsistencies**, **noisy metadata**, and **layout variations** are common.

Whether you're building:
- 🔎 **Searchable knowledge bases**
- 📚 **Semantic retrievers**
- ✂️ **Document summarization pipelines**

This tool gives you clean, structured outlines in a **universally readable JSON format**.

---

## ⚙️ Methodology

### 1. 📥 PDF Parsing with Visual Heuristics

Each PDF in the `input/` directory is parsed using **PyMuPDF (`fitz`)**, which captures:
- Text content  
- Font size & boldness  
- Y-axis position (vertical layout)  
- Casing patterns (e.g., **UPPERCASE**, *Title Case*)

All lines are normalized and pre-filtered to remove **noise patterns** such as:
- Page numbers  
- Disclaimers  
- Repeated headers/footers

---

### 2. 🧠 Heading Detection Logic

We dynamically **cluster font sizes** to identify the top three font levels. Then, each line is **scored** using a custom rule-based system based on:
- Font size weight  
- Bold text bonus  
- ALL CAPS emphasis  
- Title Case detection

**Heading levels:**
- 🟩 `H1`: Major sections (largest & boldest)  
- 🟨 `H2`: Subsections  
- 🟦 `H3`: Minor sections

📌 The first major H1 is promoted as the **document title**, unless otherwise found.

---

### 3. 🔗 Span Merging & Line Continuity

Adjacent spans with:
- Similar font size  
- Close Y-position  
- Matching font style and page

...are **merged intelligently** to preserve heading integrity even when broken by layout engines.

---

### 4. 📤 JSON Output Generation

Once headings are detected and scored, the output is exported in a clean **JSON format**:

```json
{
  "title": "Annual Financial Report 2023",
  "outline": [
    { "level": "H1", "text": "Executive Summary", "page": 1 },
    { "level": "H2", "text": "Company Overview", "page": 2 },
    { "level": "H3", "text": "Mission Statement", "page": 2 }
  ],
  "source": "report.pdf"
}
```

---

## 🔑 Key Features


* 📊 **Dynamic Font Clustering**: Infers heading hierarchy by clustering font sizes on each PDF.
* 🧼 **Noise Filtering**: Removes repeated page headers, footers, and noise like "Page 1", dates, etc.
* 🔍 **Text Style Analysis**: Evaluates bold, all-caps, and title casing to reinforce heading confidence.
* 📐 **Visual Grouping**: Merges broken lines/spans based on proximity and font similarity.
* 📄 **Robust Output Schema**: Clean JSON with title, heading hierarchy, and page references.
* ⚡ **Fast and Lightweight**: Parses 50+ page reports in seconds with minimal memory.

## 🐳 Docker Setup


1.  **Build the Docker Image**
    ```bash
    docker build --platform linux/amd64 -t pdf-outline-extractor:round1a .
    ```

2.  **Run the Extractor**
    ```bash
    docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-outline-extractor:round1a

    ```
🔁 This mounts your local `input/` and `output/` folders into the container. Drop your PDFs into `input/`, and get clean outlines in `output/`.

---

## 🧠 Summary – Challenge 1A: Structured Heading Extraction

This pipeline intelligently transforms **unstructured PDFs into clean, structured outlines**, extracting the document **title and hierarchical headings (H1, H2, H3)**.

It uses:
- 📐 Visual heuristics  
- 🔤 Font clustering  
- 🧠 Text-style scoring  

…to detect structure in real-world PDFs — even with noisy metadata or inconsistent formatting.

### ✅ Ideal For:
- 🔍 Search & indexing engines  
- 🧾 Summarization workflows  
- 📚 Knowledge base pipelines  

**Output:** Clean, structured JSON – ready for any downstream task.  
**Bonus:** Fast, accurate, and minimal dependencies. 🚀📄
