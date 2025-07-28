import os
import json
from datetime import datetime
from typing import List, Dict
from pydantic import BaseModel
from collections import Counter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Pydantic models with corrected types
class Section(BaseModel):
    document: str
    section_title: str
    importance_rank: int
    page_number: int

class Subsection(BaseModel):
    document: str
    refined_text: str
    page_number: int

class Metadata(BaseModel):
    input_documents: List[str]
    persona: str
    job: str
    processing_timestamp: str
    # key_concepts: List[str]

class OutputModel(BaseModel):
    metadata: Metadata
    extracted_sections: List[Section]
    subsection_analysis: List[Subsection]

class DocumentProcessor:
    def __init__(self):
        # self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def _extract_key_concepts(self, documents: List[Dict], job_description: str) -> List[str]:
        """Dynamically identify important terms"""
        all_texts = [job_description] + [doc["text"] for doc in documents]
        
        # Extract noun phrases
        noun_phrases = []
        for text in all_texts:
            matches = re.finditer(r'([A-Z][a-z]+(?:\s+[A-Za-z][a-z]+)*)', text)
            noun_phrases.extend([match.group().lower() for match in matches])
        
        # Get important words using TF-IDF
        tfidf = TfidfVectorizer(stop_words='english', max_features=50)
        tfidf.fit(all_texts)
        important_words = tfidf.get_feature_names_out().tolist()
        
        # Combine and return top terms
        all_terms = noun_phrases + important_words
        return [term for term, _ in Counter(all_terms).most_common(30)]
    
    def _extract_title(self, text: str) -> str:
        """Extract meaningful section title based on structure, avoiding generic placeholders and full sentences."""

        lines = [line.strip() for line in text.split('\n') if line.strip()]

        generic_titles = {
            "introduction", "overview", "summary", "table of contents",
            "contents", "section", "relevant section", "page", "document"
        }

        candidate_titles = []
        for line in lines[:10]:  # Only look at the first few lines
            if any(line.startswith(c) for c in ('•', '-', '*', '»')) or line.endswith(':'):
                continue
            if line.endswith('.') or len(line.split()) > 10 or '.' in line:
                continue
            if line[0].isupper() and line.isascii():
                lower_line = line.strip().lower()
                if lower_line not in generic_titles and not re.match(r'^\d+(\.\d+)*$', lower_line):
                    candidate_titles.append(line)

        if candidate_titles:
            candidate_titles.sort(key=lambda x: (-len(x), x))
            return candidate_titles[0]

        return "Untitled Section"

    def _extract_description(self, text: str, job_terms: List[str]) -> str:
        """Extract relevant sentences, cleaned and coherent"""
        cleaned_text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for term in job_terms if term.lower() in sentence.lower())
            if score > 0:
                scored_sentences.append((score, sentence))

        scored_sentences.sort(reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:2]]
        return ' '.join(top_sentences).strip() if top_sentences else cleaned_text[:150].strip() + '...'

    def load_documents(self, input_folder: str) -> List[Dict]:
        """Load all PDF documents"""
        docs = []
        for filename in os.listdir(input_folder):
            if not filename.lower().endswith('.pdf'):
                continue
            try:
                loader = PyPDFLoader(os.path.join(input_folder, filename))
                pages = loader.load_and_split(self.text_splitter)
                for page in pages:
                    docs.append({
                        "text": page.page_content,
                        "metadata": {
                            "source": filename,
                            "page": page.metadata.get("page", 0)
                        }
                    })
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        return docs

    def analyze_documents(self, documents: List[Dict], persona: str, job: str) -> Dict:
        """Main analysis pipeline"""
        key_concepts = self._extract_key_concepts(documents, job)
        print(f"Identified key concepts: {key_concepts[:10]}...")

        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        query = f"{persona} {job} {' '.join(key_concepts[:5])}"
        docs = vectorstore.similarity_search(query, k=10)

        sections = []
        subsections = []
        seen_titles = set()
        rank = 1

        for doc in docs:
            content = doc.page_content
            title = self._extract_title(content).strip()

            if title.lower() == "untitled section":
                continue

            unique_key = f"{title.lower()}::{doc.metadata['source']}"
            if unique_key in seen_titles:
                continue
            seen_titles.add(unique_key)

            description = self._extract_description(content, key_concepts)

            sections.append(Section(
                document=doc.metadata["source"],
                page_number=doc.metadata["page"],
                section_title=title,
                importance_rank=rank
            ))

            subsections.append(Subsection(
                document=doc.metadata["source"],
                refined_text=description,
                page_number=doc.metadata["page"]
            ))

            rank += 1

        return {
            "extracted_sections": sections[:5],
            "subsection_analysis": subsections[:5],
            "key_concepts": key_concepts,
            "document_names": list(set(doc["metadata"]["source"] for doc in documents))
        }

def main():
    input_folder = "input"
    output_file = "output/output.json"
    persona = "Food Contractor"
    job = "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."

    processor = DocumentProcessor()
    documents = processor.load_documents(input_folder)
    if not documents:
        print("No documents found in input folder")
        return

    results = processor.analyze_documents(documents, persona, job)

    output = OutputModel(
        metadata=Metadata(
            input_documents=results["document_names"],
            persona=persona,
            job=job,
            processing_timestamp=datetime.now().isoformat()
        ),
        extracted_sections=results["extracted_sections"],
        subsection_analysis=results["subsection_analysis"]
    )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
