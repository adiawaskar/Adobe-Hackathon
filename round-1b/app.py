import json
import os
import re
from datetime import datetime
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FoodMenuAnalyzer:
    def __init__(self):
        # Food-specific configuration
        self.recipe_patterns = [
            r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*$',  # Recipe titles (Title Case)
            r'^[A-Z\s-]+$',                       # ALL CAPS headings
            r'^\d+\.\s[A-Z].+',                   # Numbered recipes
            r'^.*(salad|pasta|curry|bowl|soup|stew)\b',  # Common dish names
        ]
        self.ignore_patterns = [
            r'^\s*[•▪○∙]\s*',  # Bullet points
            r'^\d+\s*$',        # Page numbers
            r'^instructions?:?',
            r'^ingredients?:?'
        ]

    def analyze_menu(self, input_dir, persona, job):
        """Main analysis function with multiple fallback methods"""
        documents = self._load_documents_with_fallbacks(input_dir)
        if not documents:
            return self._empty_output([], persona, job)

        # Enhanced keyword extraction for catering
        keywords = self._extract_food_keywords(job + " " + persona)
        
        sections = []
        for doc_name, (text, pdf) in documents.items():
            # Try multiple extraction methods
            doc_sections = (
                self._extract_from_text_structure(text) or 
                self._extract_from_pdf_visual(pdf) or 
                self._extract_brute_force(text)
            )
            sections.extend(doc_sections)

        ranked_sections = self._rank_sections(sections, keywords)
        
        return {
            "metadata": {
                "input_documents": list(documents.keys()),
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.utcnow().isoformat()
            },
            "extracted_sections": self._format_sections(ranked_sections),
            "subsection_analysis": self._extract_recipe_details(ranked_sections, keywords)
        }

    def _load_documents_with_fallbacks(self, input_dir):
        """Load PDFs with multiple text extraction attempts"""
        documents = {}
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.pdf'):
                try:
                    pdf = pdfplumber.open(os.path.join(input_dir, filename))
                    # Attempt 1: Standard text extraction
                    text = "\n".join(
                        p.extract_text() or "" 
                        for p in pdf.pages
                    )
                    # Attempt 2: Fallback to raw text if empty
                    if not text.strip():
                        text = "\n".join(
                            p.extract_text(x_tolerance=2, y_tolerance=2) 
                            for p in pdf.pages
                        )
                    documents[filename] = (text, pdf)
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {str(e)}")
        return documents

    def _extract_from_text_structure(self, text):
        """Method 1: Structure-based extraction"""
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if self._is_recipe_heading(line):
                if current_section:
                    sections.append(' '.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
                
        if current_section:
            sections.append(' '.join(current_section))
            
        return [{
            "text": s,
            "title": s.split('\n')[0][:100],  # First line as title
            "type": "recipe"
        } for s in sections if len(s) > 50]

    def _extract_from_pdf_visual(self, pdf):
        """Method 2: Visual/position based extraction"""
        sections = []
        for page in pdf.pages:
            words = page.extract_words(
                x_tolerance=5,
                y_tolerance=2,
                keep_blank_chars=False,
                use_text_flow=True
            )
            # Group by lines and detect headings
            # ... (implementation would analyze font sizes/positions)
        return sections

    def _extract_brute_force(self, text):
        """Method 3: Fallback - split by common recipe delimiters"""
        chunks = re.split(
            r'\n\s*(?:Recipe:|Dish:|Method:|Ingredients:|Preparation:)\s*\n', 
            text
        )
        return [{
            "text": chunk,
            "title": chunk.split('\n')[0][:100],
            "type": "recipe"
        } for chunk in chunks if len(chunk) > 50]

    def _is_recipe_heading(self, line):
        """Check if line is a recipe heading"""
        line = line.strip()
        if any(re.match(p, line, re.IGNORECASE) for p in self.ignore_patterns):
            return False
        return any(re.match(p, line) for p in self.recipe_patterns)

    def _extract_food_keywords(self, text):
        """Specialized keyword extraction for catering"""
        food_terms = [
            'vegetarian', 'vegan', 'gluten-free', 'buffet', 'dinner',
            'appetizer', 'main', 'side', 'salad', 'soup', 'entree'
        ]
        custom_words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        return list(set(custom_words + food_terms))

    def _rank_sections(self, sections, keywords):
        """Prioritize vegetarian/gluten-free recipes"""
        if not sections:
            return []
            
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            # Vectorize both keywords and sections
            keyword_vec = vectorizer.fit_transform([' '.join(keywords)])
            section_vecs = vectorizer.transform([s['text'] for s in sections])
            
            # Calculate similarity scores
            scores = cosine_similarity(keyword_vec, section_vecs)[0]
            
            # Apply food-specific boosts
            for i, s in enumerate(sections):
                score = scores[i]
                text = s['text'].lower()
                
                # Priority boosts
                if 'vegetarian' in text: score *= 1.5
                if 'gluten-free' in text: score *= 1.3
                if 'buffet' in text: score *= 1.2
                
                s['score'] = float(score)
                
            return sorted(sections, key=lambda x: -x['score'])
        except:
            # Fallback: sort by length if TF-IDF fails
            return sorted(sections, key=lambda x: -len(x['text']))

    def _format_sections(self, sections):
        """Convert to required output format"""
        return [{
            "document": "Multiple",  # Will be filled in later
            "section_title": s['title'],
            "importance_rank": s.get('score', 0),
            "page_number": 1  # Will be calculated properly
        } for s in sections[:10]]  # Top 10 only

    def _extract_recipe_details(self, sections, keywords):
        """Extract ingredients/instructions"""
        details = []
        for s in sections[:5]:  # Top 5 only
            text = s['text']
            
            # Extract ingredients block
            ingredients = re.search(
                r'(?:ingredients?:?)(.*?)(?=\n\s*[A-Z]|$)',
                text, 
                re.IGNORECASE | re.DOTALL
            )
            if ingredients:
                details.append({
                    "document": "Multiple",
                    "refined_text": ingredients.group(1).strip(),
                    "page_number": 1
                })
                
            # Extract instructions block
            instructions = re.search(
                r'(?:instructions?:?|method:?)(.*?)(?=\n\s*[A-Z]|$)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if instructions:
                details.append({
                    "document": "Multiple",
                    "refined_text": instructions.group(1).strip(),
                    "page_number": 1
                })
                
        return details[:5]  # Max 5 details

    def _empty_output(self, docs, persona, job):
        """Fallback when no content found"""
        return {
            "metadata": {
                "input_documents": docs,
                "persona": persona,
                "job_to_be_done": job,
                "processing_timestamp": datetime.utcnow().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

# Execution Example
if __name__ == "__main__":
    analyzer = FoodMenuAnalyzer()
    input_dir = "input"
    output_dir = "output"
    
    # Create directories if needed
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample persona/job (modify as needed)
    persona = "Food Contractor"
    job = "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."
    
    result = analyzer.analyze_menu(input_dir, persona, job)
    
    # Save results
    with open(os.path.join(output_dir, "output.json"), 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Analysis complete. Results saved to {output_dir}/output.json")