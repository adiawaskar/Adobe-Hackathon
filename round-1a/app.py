# # File: round1a/app.py
# import json
# import os
# from typing import List, Dict, Optional
# import pdfplumber
# import re
# from collections import defaultdict

# class PDFOutlineExtractor:
#     def __init__(self):
#         self.heading_patterns = [
#             (r'^(chapter|part|section)\s+\d+', 'H1'),
#             (r'^\d+\.\d+', 'H2'),
#             (r'^\d+\.\d+\.\d+', 'H3'),
#             (r'^[A-Z][A-Z0-9\s]{15,}', 'H1'),  # All caps with length > 15
#             (r'^[A-Z][a-z0-9\s]{10,}', 'H1'),  # Title case with length > 10
#         ]
#         self.min_font_size = 10  # Minimum font size to consider as heading
#         self.max_font_size_diff = 2  # Maximum font size difference between levels

#     def extract_outline(self, pdf_path: str) -> Dict:
#         """Extract title and headings from PDF"""
#         title = self._guess_title(pdf_path)
#         outline = []
        
#         with pdfplumber.open(pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, start=1):
#                 page_outline = self._process_page(page, page_num)
#                 outline.extend(page_outline)
        
#         # Clean up the outline by removing duplicates and empty entries
#         outline = self._clean_outline(outline)
        
#         return {
#             "title": title,
#             "outline": outline
#         }

#     def _guess_title(self, pdf_path: str) -> str:
#         """Guess the title from the first page"""
#         with pdfplumber.open(pdf_path) as pdf:
#             first_page = pdf.pages[0]
#             text = first_page.extract_text()
#             if text:
#                 first_line = text.split('\n')[0].strip()
#                 return first_line[:200]  # Limit title length
#         return os.path.basename(pdf_path).replace('.pdf', '')

#     def _process_page(self, page, page_num: int) -> List[Dict]:
#         """Process a single page to extract headings"""
#         headings = []
        
#         # Extract text with positioning and font information
#         words = page.extract_words(extra_attrs=["size", "fontname"])
        
#         if not words:
#             return headings
        
#         # Group words by line
#         lines = defaultdict(list)
#         for word in words:
#             # Use y0 (top) coordinate to group words into lines
#             line_key = round(word['top'])
#             lines[line_key].append(word)
        
#         # Process each line
#         for line_words in lines.values():
#             if not line_words:
#                 continue
                
#             line_text = ' '.join(w['text'] for w in line_words)
#             line_text = line_text.strip()
            
#             if not line_text:
#                 continue
                
#             # Get average font size for the line
#             avg_size = sum(w['size'] for w in line_words) / len(line_words)
            
#             # Check if this looks like a heading
#             level = self._classify_heading(line_text, avg_size)
#             if level:
#                 headings.append({
#                     "level": level,
#                     "text": line_text,
#                     "page": page_num
#                 })
        
#         return headings

#     def _classify_heading(self, text: str, font_size: float) -> Optional[str]:
#         """Classify text as heading based on patterns and font size"""
#         # Check patterns first
#         for pattern, level in self.heading_patterns:
#             if re.match(pattern, text, re.IGNORECASE):
#                 return level
                
#         # Fallback to font size if no pattern matches
#         if font_size > 14:
#             return "H1"
#         elif font_size > 12:
#             return "H2"
#         elif font_size > 10:
#             return "H3"
            
#         return None

#     def _clean_outline(self, outline: List[Dict]) -> List[Dict]:
#         """Clean up the outline by removing duplicates and empty entries"""
#         seen = set()
#         cleaned = []
        
#         for item in outline:
#             key = (item['level'], item['text'], item['page'])
#             if key not in seen and item['text'].strip():
#                 seen.add(key)
#                 cleaned.append(item)
        
#         return cleaned


# def process_pdfs(input_dir: str, output_dir: str):
#     """Process all PDFs in input directory and save JSONs to output directory"""
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     extractor = PDFOutlineExtractor()
    
#     for filename in os.listdir(input_dir):
#         if filename.lower().endswith('.pdf'):
#             pdf_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
            
#             try:
#                 result = extractor.extract_outline(pdf_path)
#                 with open(output_path, 'w', encoding='utf-8') as f:
#                     json.dump(result, f, indent=2, ensure_ascii=False)
#                 print(f"Processed {filename} successfully")
#             except Exception as e:
#                 print(f"Error processing {filename}: {str(e)}")


# if __name__ == "__main__":
#     input_dir = "input"
#     output_dir = "output"
#     os.makedirs(input_dir, exist_ok=True)
#     os.makedirs(output_dir, exist_ok=True)
#     process_pdfs(input_dir, output_dir)


import os
import json
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import pdfplumber
import numpy as np
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import cv2
from deskew import determine_skew
import math

class Heading(BaseModel):
    level: str  # H1, H2, H3
    text: str
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    confidence: float = 1.0

class DocumentOutline(BaseModel):
    title: str
    outline: List[Heading]

class PDFOutlineExtractor:
    def __init__(self):
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(stop_words=None, max_features=1000)
        self.heading_embeddings = {}

        # Enhanced heading detection configuration
        self.heading_config = {
            'H1': {
                'min_size': 18,
                'patterns': [
                    r'^(?:[A-Z][A-Z0-9\s\-]+|[IVXLCDM]+\.\s+[A-Z])$',
                    r'^[\u4e00-\u9fff]{2,}[\u4e00-\u9fff、。\s]*$',
                    r'^第[一二三四五六七八九十]+章\s+[\u4e00-\u9fff]+'
                ],
                'position': ('top', 0.2),  # Top 20% of page
                'alignment': ('center', 0.8),  # 80% confidence in center alignment
                'line_spacing': 1.5
            },
            'H2': {
                'min_size': 14,
                'patterns': [
                    r'^(?:[A-Z][a-zA-Z0-9\s\-]+|\d+\.\s+[A-Z][a-z])$',
                    r'^[\u4e00-\u9fff]{2,}[\u4e00-\u9fff、。\s]*$',
                    r'^第[一二三四五六七八九十]+节\s+[\u4e00-\u9fff]+'
                ],
                'position': ('any', 0.5),
                'alignment': ('left', 0.6),
                'line_spacing': 1.2
            },
            'H3': {
                'min_size': 12,
                'patterns': [
                    r'^(?:[a-z][a-z0-9\s\-]+|\d+\.\d+\.\s+[A-Za-z])$',
                    r'^[\u4e00-\u9fff]{2,}[\u4e00-\u9fff、。\s]*$'
                ],
                'position': ('any', 0.3),
                'alignment': ('left', 0.4),
                'line_spacing': 1.0
            }
        }

        # OCR configuration
        self.ocr_config = {
            'lang': 'eng+chi_sim+jpn',
            'config': '--psm 6 --oem 3',
            'preprocess': True
        }

    def extract_outline(self, pdf_path: str) -> DocumentOutline:
        """Enhanced outline extraction with multi-modal approach"""
        title = os.path.basename(pdf_path).replace('.pdf', '')
        outline = []
        title_candidates = []
        all_text = []
        visual_elements = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Extract text and visual elements
                    page_data = self._extract_page_data(page, page_num)
                    all_text.append(page_data['text'])
                    visual_elements.extend(page_data['visual_elements'])
                    
                    # Process extracted content
                    headings = self._find_headings(
                        text=page_data['text'],
                        visual_elements=page_data['visual_elements'],
                        page_num=page_num
                    )
                    outline.extend(headings)
                    
                    # Collect title candidates
                    title_candidates.extend(h.text for h in headings if h.level == 'H1')
                except Exception as e:
                    print(f"Error processing page {page_num}: {str(e)}")
                    continue
        
        # Enhanced semantic analysis
        if all_text:
            self._enhance_with_semantics(outline, all_text, visual_elements)
        
        # Structural analysis and filtering
        outline = self._analyze_structure(outline)
        
        # Select best title
        if title_candidates:
            title = self._select_best_title(title_candidates)
            
        return DocumentOutline(title=title, outline=outline)

    def _extract_page_data(self, page, page_num: int) -> Dict:
        """Multi-modal page data extraction"""
        result = {
            'text': '',
            'visual_elements': [],
            'page_num': page_num
        }
        
        # 1. Text extraction
        result['text'] = self._extract_text(page)
        
        # 2. Visual element extraction
        if len(result['text'].strip()) < 100:  # Threshold for OCR fallback
            img_data = self._process_page_image(page, page_num)
            result['text'] += " " + img_data['text']
            result['visual_elements'] = img_data['visual_elements']
        
        return result

    def _extract_text(self, page) -> str:
        """Extract text with layout preservation"""
        text = page.extract_text(
            layout=True,
            x_density=7.0,
            y_density=7.0
        ) or ""
        
        # Extract from tables
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                text += " ".join(str(cell) for cell in row if cell) + "\n"
        
        return text.strip()

    def _process_page_image(self, page, page_num: int) -> Dict:
        """Advanced image processing for OCR"""
        result = {
            'text': '',
            'visual_elements': []
        }
        
        try:
            # Convert to high-res image
            img = page.to_image(resolution=300).original
            pil_img = Image.fromarray(img)
            
            # Preprocess image
            processed_img = self._preprocess_image(pil_img)
            
            # OCR with layout analysis
            ocr_data = pytesseract.image_to_data(
                processed_img,
                lang=self.ocr_config['lang'],
                config=self.ocr_config['config'],
                output_type=pytesseract.Output.DICT
            )
            
            # Process OCR results
            text_blocks = []
            for i in range(len(ocr_data['text'])):
                if ocr_data['text'][i].strip():
                    block = {
                        'text': ocr_data['text'][i],
                        'bbox': (
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i]
                        ),
                        'conf': ocr_data['conf'][i] / 100.0,
                        'block_num': ocr_data['block_num'][i],
                        'line_num': ocr_data['line_num'][i]
                    }
                    text_blocks.append(block)
            
            # Group by lines and blocks
            grouped = self._group_text_blocks(text_blocks)
            result['visual_elements'] = grouped
            result['text'] = " ".join(b['text'] for b in grouped)
            
        except Exception as e:
            print(f"Image processing failed on page {page_num}: {str(e)}")
        
        return result

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR results"""
        try:
            # Convert to grayscale
            img = np.array(image.convert('L'))
            
            # Deskew
            angle = determine_skew(img)
            if angle:
                img = self._rotate_image(img, angle)
            
            # Enhance contrast
            img = cv2.equalizeHist(img)
            
            # Denoise
            img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            
            # Binarize
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return Image.fromarray(img)
        except:
            return image

    def _rotate_image(self, image: np.array, angle: float) -> np.array:
        """Rotate image to correct skew"""
        h, w = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _group_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Group text blocks into logical elements"""
        # Sort by vertical position
        blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
        
        grouped = []
        current_group = None
        
        for block in blocks:
            if not current_group:
                current_group = {
                    'text': block['text'],
                    'bbox': block['bbox'],
                    'conf': [block['conf']],
                    'lines': 1
                }
            else:
                # Check if same line (similar y-position)
                y_diff = abs(block['bbox'][1] - current_group['bbox'][1])
                if y_diff < 10:  # Same line
                    current_group['text'] += " " + block['text']
                    current_group['bbox'] = (
                        min(current_group['bbox'][0], block['bbox'][0]),
                        min(current_group['bbox'][1], block['bbox'][1]),
                        max(current_group['bbox'][2], block['bbox'][2]),
                        max(current_group['bbox'][3], block['bbox'][3])
                    )
                    current_group['conf'].append(block['conf'])
                else:
                    # New line
                    grouped.append(current_group)
                    current_group = {
                        'text': block['text'],
                        'bbox': block['bbox'],
                        'conf': [block['conf']],
                        'lines': 1
                    }
        
        if current_group:
            grouped.append(current_group)
        
        # Calculate average confidence
        for group in grouped:
            group['conf'] = sum(group['conf']) / len(group['conf'])
        
        return grouped

    def _find_headings(self, text: str, visual_elements: List[Dict], page_num: int) -> List[Heading]:
        """Multi-modal heading detection"""
        headings = []
        
        # 1. Process text-based headings
        text_headings = self._find_text_headings(text, page_num)
        headings.extend(text_headings)
        
        # 2. Process visual elements
        visual_headings = self._find_visual_headings(visual_elements, page_num)
        headings.extend(visual_headings)
        
        return headings

    def _find_text_headings(self, text: str, page_num: int) -> List[Heading]:
        """Find headings in extracted text"""
        headings = []
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            heading_level = self._classify_heading(line)
            if heading_level:
                clean_text = self._clean_text(line)
                if clean_text:
                    headings.append(Heading(
                        level=heading_level,
                        text=clean_text,
                        page=page_num,
                        confidence=0.9  # High confidence for direct text
                    ))
        return headings

    def _find_visual_headings(self, elements: List[Dict], page_num: int) -> List[Heading]:
        """Find headings in visual elements"""
        headings = []
        
        for element in elements:
            if element['conf'] < 0.6:  # Skip low-confidence elements
                continue
                
            text = element['text'].strip()
            if not text:
                continue
                
            # Check visual characteristics
            bbox = element['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height if height > 0 else 1
            
            # Check if element looks like a heading
            heading_level = self._classify_visual_heading(text, element)
            if heading_level:
                headings.append(Heading(
                    level=heading_level,
                    text=self._clean_text(text),
                    page=page_num,
                    bbox=bbox,
                    confidence=element['conf'] * 0.8  # Slightly reduce confidence for visual
                ))
        
        return headings

    def _classify_heading(self, text: str) -> Optional[str]:
        """Classify text as heading using multiple heuristics"""
        # Skip non-heading patterns
        if self._is_non_heading(text):
            return None
            
        # Check each heading level
        for level, config in self.heading_config.items():
            # Pattern matching
            for pattern in config['patterns']:
                if re.fullmatch(pattern, text, re.UNICODE):
                    # Additional linguistic validation
                    if self._validate_heading_content(text, level):
                        return level
        return None

    def _classify_visual_heading(self, text: str, element: Dict) -> Optional[str]:
        """Classify visual element as heading"""
        if self._is_non_heading(text):
            return None
            
        bbox = element['bbox']
        page_width = 600  # Approximate, can be adjusted
        x_center = (bbox[0] + bbox[2]) / 2
        
        for level, config in self.heading_config.items():
            # Check position
            pos_type, pos_thresh = config['position']
            if pos_type == 'top' and bbox[1] > pos_thresh * 1000:  # Assuming page height ~1000
                continue
                
            # Check alignment
            align_type, align_thresh = config['alignment']
            if align_type == 'center':
                center_dist = abs(x_center - page_width/2) / (page_width/2)
                if center_dist > (1 - align_thresh):
                    continue
            elif align_type == 'left' and bbox[0] > page_width * 0.2:  # Too far right
                continue
                
            # Check text patterns
            for pattern in config['patterns']:
                if re.fullmatch(pattern, text, re.UNICODE):
                    if self._validate_heading_content(text, level):
                        return level
        return None

    def _is_non_heading(self, text: str) -> bool:
        """Check if text is definitely not a heading"""
        return any(re.search(pattern, text) for pattern in [
            r'^\d+$',  # Pure numbers
            r'^[\W_]+$',  # Only symbols
            r'^[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF]',  # Non-text
            r'^http[s]?://',  # URLs
            r'^\s*$',  # Empty
            r'^.{0,2}$'  # Too short
        ]) or len(text) > 200  # Too long

    def _validate_heading_content(self, text: str, level: str) -> bool:
        """Validate heading content based on level"""
        # Must contain meaningful text
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / len(text) < 0.3:  # Less than 30% letters
            return False
            
        # Level-specific checks
        if level == 'H1':
            return len(text.split()) >= 1 and len(text) >= 3
        elif level == 'H2':
            return len(text.split()) >= 1 and len(text) >= 2
        return True  # H3 can be shorter

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving multilingual characters"""
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'^\W+|\W+$', '', text)
        return ' '.join(text.split())

    def _enhance_with_semantics(self, outline: List[Heading], all_text: List[str], visual_elements: List[Dict]):
        """Enhanced semantic analysis with visual context"""
        try:
            # Train TF-IDF on all document text
            self.tfidf.fit(all_text)
            
            # Calculate importance scores
            for heading in outline:
                if heading.text not in self.heading_embeddings:
                    # Text embedding
                    text_emb = self.embedding_model.encode(heading.text)
                    
                    # Visual features (if available)
                    vis_feats = np.zeros(5)
                    if heading.bbox:
                        vis_feats = np.array([
                            heading.bbox[0],  # x0
                            heading.bbox[1],  # y0
                            (heading.bbox[2] - heading.bbox[0]),  # width
                            (heading.bbox[3] - heading.bbox[1]),  # height
                            heading.confidence
                        ])
                    
                    # Combine features
                    combined = np.concatenate([text_emb, vis_feats])
                    self.heading_embeddings[heading.text] = combined
        except Exception as e:
            print(f"Enhanced semantic analysis failed: {str(e)}")

    def _analyze_structure(self, outline: List[Heading]) -> List[Heading]:
        """Analyze document structure to validate headings"""
        # Group by page and sort by position
        outline.sort(key=lambda h: (h.page, h.bbox[1] if h.bbox else 0))
        
        # Filter based on structural patterns
        filtered = []
        prev_level = None
        
        for heading in outline:
            # Skip duplicates
            if any(h.text == heading.text for h in filtered):
                continue
                
            # Validate level progression
            if prev_level:
                curr_level_num = int(heading.level[1:])
                prev_level_num = int(prev_level[1:])
                if curr_level_num > prev_level_num + 1:  # Skip invalid jumps
                    continue
                    
            filtered.append(heading)
            prev_level = heading.level
        
        return filtered

    def _select_best_title(self, candidates: List[str]) -> str:
        """Select title using multi-modal features"""
        if not candidates:
            return ""
            
        # Simple fallback
        if not self.heading_embeddings:
            return max(candidates, key=len)
            
        # Score candidates
        scored = []
        for text in candidates:
            if text in self.heading_embeddings:
                features = self.heading_embeddings[text]
                # Score based on position (y-coordinate) and semantic importance
                score = -features[-4] + np.linalg.norm(features[:-5])  # Lower y is better
                scored.append((text, score))
        
        return max(scored, key=lambda x: x[1], default=(candidates[0], 0))[0]

def process_pdfs(input_dir: str = 'input', output_dir: str = 'output'):
    """Process all PDFs with comprehensive error handling"""
    os.makedirs(output_dir, exist_ok=True)
    extractor = PDFOutlineExtractor()
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            try:
                outline = extractor.extract_outline(pdf_path)
                output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'title': outline.title,
                        'outline': [h.dict() for h in outline.outline]
                    }, f, indent=2, ensure_ascii=False)
                    
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")

if __name__ == "__main__":
    input_dir = os.getenv('INPUT_DIR', 'input')
    output_dir = os.getenv('OUTPUT_DIR', 'output')
    process_pdfs(input_dir, output_dir)