# File: round1a/app.py
import json
import os
from typing import List, Dict, Optional
import pdfplumber
import re
from collections import defaultdict

class PDFOutlineExtractor:
    def __init__(self):
        self.heading_patterns = [
            (r'^(chapter|part|section)\s+\d+', 'H1'),
            (r'^\d+\.\d+', 'H2'),
            (r'^\d+\.\d+\.\d+', 'H3'),
            (r'^[A-Z][A-Z0-9\s]{15,}', 'H1'),  # All caps with length > 15
            (r'^[A-Z][a-z0-9\s]{10,}', 'H1'),  # Title case with length > 10
        ]
        self.min_font_size = 10  # Minimum font size to consider as heading
        self.max_font_size_diff = 2  # Maximum font size difference between levels

    def extract_outline(self, pdf_path: str) -> Dict:
        """Extract title and headings from PDF"""
        title = self._guess_title(pdf_path)
        outline = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_outline = self._process_page(page, page_num)
                outline.extend(page_outline)
        
        # Clean up the outline by removing duplicates and empty entries
        outline = self._clean_outline(outline)
        
        return {
            "title": title,
            "outline": outline
        }

    def _guess_title(self, pdf_path: str) -> str:
        """Guess the title from the first page"""
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            if text:
                first_line = text.split('\n')[0].strip()
                return first_line[:200]  # Limit title length
        return os.path.basename(pdf_path).replace('.pdf', '')

    def _process_page(self, page, page_num: int) -> List[Dict]:
        """Process a single page to extract headings"""
        headings = []
        
        # Extract text with positioning and font information
        words = page.extract_words(extra_attrs=["size", "fontname"])
        
        if not words:
            return headings
        
        # Group words by line
        lines = defaultdict(list)
        for word in words:
            # Use y0 (top) coordinate to group words into lines
            line_key = round(word['top'])
            lines[line_key].append(word)
        
        # Process each line
        for line_words in lines.values():
            if not line_words:
                continue
                
            line_text = ' '.join(w['text'] for w in line_words)
            line_text = line_text.strip()
            
            if not line_text:
                continue
                
            # Get average font size for the line
            avg_size = sum(w['size'] for w in line_words) / len(line_words)
            
            # Check if this looks like a heading
            level = self._classify_heading(line_text, avg_size)
            if level:
                headings.append({
                    "level": level,
                    "text": line_text,
                    "page": page_num
                })
        
        return headings

    def _classify_heading(self, text: str, font_size: float) -> Optional[str]:
        """Classify text as heading based on patterns and font size"""
        # Check patterns first
        for pattern, level in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return level
                
        # Fallback to font size if no pattern matches
        if font_size > 14:
            return "H1"
        elif font_size > 12:
            return "H2"
        elif font_size > 10:
            return "H3"
            
        return None

    def _clean_outline(self, outline: List[Dict]) -> List[Dict]:
        """Clean up the outline by removing duplicates and empty entries"""
        seen = set()
        cleaned = []
        
        for item in outline:
            key = (item['level'], item['text'], item['page'])
            if key not in seen and item['text'].strip():
                seen.add(key)
                cleaned.append(item)
        
        return cleaned


def process_pdfs(input_dir: str, output_dir: str):
    """Process all PDFs in input directory and save JSONs to output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    extractor = PDFOutlineExtractor()
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            
            try:
                result = extractor.extract_outline(pdf_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Processed {filename} successfully")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_dir = "input"
    output_dir = "output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    process_pdfs(input_dir, output_dir)