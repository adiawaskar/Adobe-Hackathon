# import json
# import os
# import re
# from datetime import datetime
# import pdfplumber
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# class FoodMenuAnalyzer:
#     def __init__(self):
#         # Food-specific configuration
#         self.recipe_patterns = [
#             r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$',  # Recipe titles (Title Case)
#             r'^[A-Z\s-]+$',                       # ALL CAPS headings
#             r'^\d+\.\s[A-Z].+',                   # Numbered recipes
#             r'^.*(salad|pasta|curry|bowl|soup|stew)\b',  # Common dish names
#         ]
#         self.ignore_patterns = [
#             r'^\s*[•▪○∙]\s*',  # Bullet points
#             r'^\d+\s*$',        # Page numbers
#             r'^instructions?:?',
#             r'^ingredients?:?'
#         ]

#     def analyze_menu(self, input_dir, persona, job):
#         """Main analysis function with multiple fallback methods"""
#         documents = self._load_documents_with_fallbacks(input_dir)
#         if not documents:
#             return self._empty_output([], persona, job)

#         # Enhanced keyword extraction for catering
#         keywords = self._extract_food_keywords(job + " " + persona)
        
#         sections = []
#         for doc_name, (text, pdf) in documents.items():
#             # Try multiple extraction methods
#             doc_sections = (
#                 self._extract_from_text_structure(text) or 
#                 self._extract_from_pdf_visual(pdf) or 
#                 self._extract_brute_force(text)
#             )
#             sections.extend(doc_sections)

#         ranked_sections = self._rank_sections(sections, keywords)
        
#         return {
#             "metadata": {
#                 "input_documents": list(documents.keys()),
#                 "persona": persona,
#                 "job_to_be_done": job,
#                 "processing_timestamp": datetime.utcnow().isoformat()
#             },
#             "extracted_sections": self._format_sections(ranked_sections),
#             "subsection_analysis": self._extract_recipe_details(ranked_sections, keywords)
#         }

#     def _load_documents_with_fallbacks(self, input_dir):
#         """Load PDFs with multiple text extraction attempts"""
#         documents = {}
#         for filename in os.listdir(input_dir):
#             if filename.lower().endswith('.pdf'):
#                 try:
#                     pdf = pdfplumber.open(os.path.join(input_dir, filename))
#                     # Attempt 1: Standard text extraction
#                     text = "\n".join(
#                         p.extract_text() or "" 
#                         for p in pdf.pages
#                     )
#                     # Attempt 2: Fallback to raw text if empty
#                     if not text.strip():
#                         text = "\n".join(
#                             p.extract_text(x_tolerance=2, y_tolerance=2) 
#                             for p in pdf.pages
#                         )
#                     documents[filename] = (text, pdf)
#                 except Exception as e:
#                     print(f"⚠️ Error loading {filename}: {str(e)}")
#         return documents

#     def _extract_from_text_structure(self, text):
#         """Method 1: Structure-based extraction"""
#         sections = []
#         current_section = []
        
#         for line in text.split('\n'):
#             line = line.strip()
#             if not line:
#                 continue
                
#             if self._is_recipe_heading(line):
#                 if current_section:
#                     sections.append(' '.join(current_section))
#                 current_section = [line]
#             else:
#                 current_section.append(line)
                
#         if current_section:
#             sections.append(' '.join(current_section))
            
#         return [{
#             "text": s,
#             "title": s.split('\n')[0][:100],  # First line as title
#             "type": "recipe"
#         } for s in sections if len(s) > 50]

#     def _extract_from_pdf_visual(self, pdf):
#         """Method 2: Visual/position based extraction"""
#         sections = []
#         for page in pdf.pages:
#             words = page.extract_words(
#                 x_tolerance=5,
#                 y_tolerance=2,
#                 keep_blank_chars=False,
#                 use_text_flow=True
#             )
#             # Group by lines and detect headings
#             # ... (implementation would analyze font sizes/positions)
#         return sections

#     def _extract_brute_force(self, text):
#         """Method 3: Fallback - split by common recipe delimiters"""
#         chunks = re.split(
#             r'\n\s*(?:Recipe:|Dish:|Method:|Ingredients:|Preparation:)\s*\n', 
#             text
#         )
#         return [{
#             "text": chunk,
#             "title": chunk.split('\n')[0][:100],
#             "type": "recipe"
#         } for chunk in chunks if len(chunk) > 50]

#     def _is_recipe_heading(self, line):
#         """Check if line is a recipe heading"""
#         line = line.strip()
#         if any(re.match(p, line, re.IGNORECASE) for p in self.ignore_patterns):
#             return False
#         return any(re.match(p, line) for p in self.recipe_patterns)

#     def _extract_food_keywords(self, text):
#         """Specialized keyword extraction for catering"""
#         food_terms = [
#             'vegetarian', 'vegan', 'gluten-free', 'buffet', 'dinner',
#             'appetizer', 'main', 'side', 'salad', 'soup', 'entree'
#         ]
#         custom_words = re.findall(r'\b[a-z]{4,}\b', text.lower())
#         return list(set(custom_words + food_terms))

#     def _rank_sections(self, sections, keywords):
#         """Prioritize vegetarian/gluten-free recipes"""
#         if not sections:
#             return []
            
#         vectorizer = TfidfVectorizer(stop_words='english')
#         try:
#             # Vectorize both keywords and sections
#             keyword_vec = vectorizer.fit_transform([' '.join(keywords)])
#             section_vecs = vectorizer.transform([s['text'] for s in sections])
            
#             # Calculate similarity scores
#             scores = cosine_similarity(keyword_vec, section_vecs)[0]
            
#             # Apply food-specific boosts
#             for i, s in enumerate(sections):
#                 score = scores[i]
#                 text = s['text'].lower()
                
#                 # Priority boosts
#                 if 'vegetarian' in text: score *= 1.5
#                 if 'gluten-free' in text: score *= 1.3
#                 if 'buffet' in text: score *= 1.2
                
#                 s['score'] = float(score)
                
#             return sorted(sections, key=lambda x: -x['score'])
#         except:
#             # Fallback: sort by length if TF-IDF fails
#             return sorted(sections, key=lambda x: -len(x['text']))

#     def _format_sections(self, sections):
#         """Convert to required output format"""
#         return [{
#             "document": "Multiple",  # Will be filled in later
#             "section_title": s['title'],
#             "importance_rank": s.get('score', 0),
#             "page_number": 1  # Will be calculated properly
#         } for s in sections[:10]]  # Top 10 only

#     def _extract_recipe_details(self, sections, keywords):
#         """Extract ingredients/instructions"""
#         details = []
#         for s in sections[:5]:  # Top 5 only
#             text = s['text']
            
#             # Extract ingredients block
#             ingredients = re.search(
#                 r'(?:ingredients?:?)(.*?)(?=\n\s*[A-Z]|$)',
#                 text, 
#                 re.IGNORECASE | re.DOTALL
#             )
#             if ingredients:
#                 details.append({
#                     "document": "Multiple",
#                     "refined_text": ingredients.group(1).strip(),
#                     "page_number": 1
#                 })
                
#             # Extract instructions block
#             instructions = re.search(
#                 r'(?:instructions?:?|method:?)(.*?)(?=\n\s*[A-Z]|$)',
#                 text,
#                 re.IGNORECASE | re.DOTALL
#             )
#             if instructions:
#                 details.append({
#                     "document": "Multiple",
#                     "refined_text": instructions.group(1).strip(),
#                     "page_number": 1
#                 })
                
#         return details[:5]  # Max 5 details

#     def _empty_output(self, docs, persona, job):
#         """Fallback when no content found"""
#         return {
#             "metadata": {
#                 "input_documents": docs,
#                 "persona": persona,
#                 "job_to_be_done": job,
#                 "processing_timestamp": datetime.utcnow().isoformat()
#             },
#             "extracted_sections": [],
#             "subsection_analysis": []
#         }

# # Execution Example
# if __name__ == "__main__":
#     analyzer = FoodMenuAnalyzer()
#     input_dir = "input"
#     output_dir = "output"
    
#     # Create directories if needed
#     os.makedirs(input_dir, exist_ok=True)
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Sample persona/job (modify as needed)
#     persona = "Food Contractor"
#     job = "Prepare a non-vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."
    
#     result = analyzer.analyze_menu(input_dir, persona, job)
    
#     # Save results
#     with open(os.path.join(output_dir, "output.json"), 'w') as f:
#         json.dump(result, f, indent=2)
    
#     print(f"✅ Analysis complete. Results saved to {output_dir}/output.json")


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
    page_number: int
    section_title: str
    importance_rank: int

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
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
            # Skip bullets or list intros
            if any(line.startswith(c) for c in ('•', '-', '*', '»')) or line.endswith(':'):
                continue

            # Filter full sentences (likely to end in punctuation or contain verbs)
            if line.endswith('.') or len(line.split()) > 12:
                continue

            # Check if it’s likely a proper heading
            if line[0].isupper() and line.isascii():
                lower_line = line.strip().lower()
                if lower_line not in generic_titles and not re.match(r'^\d+(\.\d+)*$', lower_line):
                    candidate_titles.append(line)

        # Choose the best candidate
        if candidate_titles:
            # Pick the longest high-quality title under 12 words
            candidate_titles.sort(key=lambda x: (-len(x), x))
            return candidate_titles[0]

        # If nothing fits, fallback
        return "Untitled Section"


    
    def _extract_description(self, text: str, job_terms: List[str]) -> str:
        """Extract relevant sentences, cleaned and coherent"""
        # Normalize spacing and line breaks
        cleaned_text = re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()
        
        # Simple sentence segmentation
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        
        scored_sentences = []
        for sentence in sentences:
            score = sum(1 for term in job_terms if term.lower() in sentence.lower())
            if score > 0:
                scored_sentences.append((score, sentence))

        scored_sentences.sort(reverse=True)
        top_sentences = [s[1] for s in scored_sentences[:2]]

        # Always return clean, well-formed result
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
        
        # Create vector store
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
        # Search for relevant documents
        query = f"{persona} {job} {' '.join(key_concepts[:5])}"
        docs = vectorstore.similarity_search(query, k=10)
        
        # Prepare output
        sections = []
        subsections = []
        
        rank = 1  # Manual counter in case we skip some docs

        for doc in docs:
            content = doc.page_content
            title = self._extract_title(content)

            # Skip if title is "Untitled Section"
            if title.lower() == "untitled section":
                continue

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

            rank += 1  # Only increment if added

        
        return {
            "extracted_sections": sections[:5],
            "subsection_analysis": subsections[:5],
            "key_concepts": key_concepts,
            "document_names": list(set(doc["metadata"]["source"] for doc in documents))
        }

    

def main():
    # Configuration
    input_folder = "input"
    output_file = "output/output.json"
    persona = "Food Contractor"
    job = "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Load documents
    documents = processor.load_documents(input_folder)
    if not documents:
        print("No documents found in input folder")
        return
    
    # Analyze documents
    results = processor.analyze_documents(documents, persona, job)
    
    # Prepare output with correct types
    output = OutputModel(
        metadata=Metadata(
            input_documents=results["document_names"],
            persona=persona,
            job=job,
            processing_timestamp=datetime.now().isoformat()
            # key_concepts=results["key_concepts"],
        ),
        extracted_sections=results["extracted_sections"],
        subsection_analysis=results["subsection_analysis"]
    )
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output.model_dump_json(indent=2))

    
    print(f"Analysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
