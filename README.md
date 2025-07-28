
<p align="center">
  <img src="assets/adobe_logo.png" alt="Adobe Logo" height="100">
</p>

<h1 align="center">Persona-Driven Document Intelligence</h1>
<h3 align="center">Adobe India Hackathon 2025 — Round 2: "Connect What Matters — For the User Who Matters"</h3>

---

## 📘 Project Description

In an age of overwhelming information, discovering *what truly matters* within a document isn't just about keyword search — it's about context, purpose, and personalization.

This solution transforms a **collection of unstructured PDFs** into **structured, context-aware intelligence**, guided by a user-defined **persona** and their **task (job-to-be-done)**. It automatically identifies the most relevant sections and sub-sections from multi-page documents and outputs them in a ranked, machine-readable format.

> Think of it as a **document analyst in a box**, operating offline, in a single Docker container.

---

## 🎯 Objective

Given:
- A **set of 3–10 PDFs**
- A **persona** (e.g., Investment Analyst, PhD Student)
- A **task** (e.g., "Compare R&D trends", "Prepare a literature review")

The system should:
- Extract and rank **relevant sections**
- Perform **fine-grained subsection analysis**
- Output results in a structured JSON format

---

## 🧠 Core Capabilities

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🗂️ Multi-PDF Support             | Handles collections of up to 10 documents                                  |
| 🧑 Persona-aware Inference        | Considers user context while ranking sections                              |
| 🔍 Relevance-based Ranking        | Uses embeddings + heuristics to assign importance to content               |
| 📄 Sub-section Granularity        | Extracts and summarizes key paragraphs or bullets within a section         |
| 🌐 Multilingual Compatibility     | Supports English, Hindi, Japanese, and more (based on tokenizer model)     |
| 🐳 Dockerized + Offline Execution | CPU-only, ≤1GB model, no internet required                                 |

---

## 🧱 System Architecture

```text
User Input
 ├─ documents/ (3–10 PDFs)
 ├─ persona.txt
 └─ job_to_be_done.txt
         ↓
[ PDF Text & Layout Extractor ]
         ↓
[ Embedding Generator ]
         ↓
[ Relevance Ranker ]
         ↓
[ Sub-section Summarizer ]
         ↓
[ Structured Output JSON ]
" alt="Adobe Logo" height="100">
</p>

<h1 align="center">Persona-Driven Document Intelligence</h1>
<h3 align="center">Adobe India Hackathon 2025 — Round 2: "Connect What Matters — For the User Who Matters"</h3>

---

## 📘 Project Description

In an age of overwhelming information, discovering *what truly matters* within a document isn't just about keyword search — it's about context, purpose, and personalization.

This solution transforms a **collection of unstructured PDFs** into **structured, context-aware intelligence**, guided by a user-defined **persona** and their **task (job-to-be-done)**. It automatically identifies the most relevant sections and sub-sections from multi-page documents and outputs them in a ranked, machine-readable format.

> Think of it as a **document analyst in a box**, operating offline, in a single Docker container.

---

## 🎯 Objective

Given:
- A **set of 3–10 PDFs**
- A **persona** (e.g., Investment Analyst, PhD Student)
- A **task** (e.g., "Compare R&D trends", "Prepare a literature review")

The system should:
- Extract and rank **relevant sections**
- Perform **fine-grained subsection analysis**
- Output results in a structured JSON format

---

## 🧠 Core Capabilities

| Feature                          | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🗂️ Multi-PDF Support             | Handles collections of up to 10 documents                                  |
| 🧑 Persona-aware Inference        | Considers user context while ranking sections                              |
| 🔍 Relevance-based Ranking        | Uses embeddings + heuristics to assign importance to content               |
| 📄 Sub-section Granularity        | Extracts and summarizes key paragraphs or bullets within a section         |
| 🌐 Multilingual Compatibility     | Supports English, Hindi, Japanese, and more (based on tokenizer model)     |
| 🐳 Dockerized + Offline Execution | CPU-only, ≤1GB model, no internet required                                 |

---

## 🧱 System Architecture

```text
User Input
 ├─ documents/ (3–10 PDFs)
 ├─ persona.txt
 └─ job_to_be_done.txt
         ↓
[ PDF Text & Layout Extractor ]
         ↓
[ Embedding Generator ]
         ↓
[ Relevance Ranker ]
         ↓
[ Sub-section Summarizer ]
         ↓
[ Structured Output JSON ]
