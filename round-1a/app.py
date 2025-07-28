import os, json, fitz, re, unicodedata
from typing import List, Tuple

INPUT_DIR  = "input"
OUTPUT_DIR = "output"
NOISE_PATTERNS = [
    r'^[A-Z]\s?RFP:.*',
    r'^\d{1,2}/\d{1,2}/\d{2,4}$',
    r'^Page\s+\d+$',
    r'^Table\s+of\s+Contents$',
    r'^Appendix\s*[A-Z]?$',
]
NOISE_WORDS = {"version", "note", "disclaimer", "contents", "index", "references", "abstract", "introduction", "conclusion"}

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    return re.sub(r'\s+', ' ', s)

def is_noise(s: str) -> bool:
    if not s:
        return True
    sl = s.lower()
    if len(s) < 3 or sl in NOISE_WORDS:
        return True
    for pat in NOISE_PATTERNS:
        if re.match(pat, s, re.IGNORECASE):
            return True
    return False

def cluster_font_sizes(sizes: List[float]) -> List[float]:
    if not sizes:
        return [12.0, 10.0, 8.0]  # default sizes if no text found
    
    # Get unique sizes rounded to 1 decimal place
    uniq = sorted({round(sz, 1) for sz in sizes}, reverse=True)
    
    # If we have less than 3 sizes, pad with smaller sizes
    if len(uniq) < 3:
        if len(uniq) == 1:
            return uniq + [uniq[0]-2.0, uniq[0]-4.0]
        else:
            return uniq + [uniq[-1]-2.0]
    return uniq[:3]

def extract_headings(path: str) -> Tuple[str, List[dict]]:
    doc = fitz.open(path)
    font_sizes = []
    raw = []  # text, size, y0, y1, page, is_bold, is_all_upper

    for p in range(len(doc)):
        page = doc[p]
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans: 
                    continue
                
                # Combine spans with similar properties
                combined_spans = []
                current_span = None
                
                for span in spans:
                    if current_span is None:
                        current_span = span.copy()
                    else:
                        # Check if we can merge with previous span
                        if (abs(span['size'] - current_span['size']) < 0.5 and 
                            span['flags'] == current_span['flags'] and 
                            abs(span['origin'][1] - current_span['origin'][1]) < 2):
                            current_span['text'] += span['text']
                            current_span['bbox'] = (
                                min(current_span['bbox'][0], span['bbox'][0]),
                                min(current_span['bbox'][1], span['bbox'][1]),
                                max(current_span['bbox'][2], span['bbox'][2]),
                                max(current_span['bbox'][3], span['bbox'][3]),
                            )
                        else:
                            combined_spans.append(current_span)
                            current_span = span.copy()
                
                if current_span is not None:
                    combined_spans.append(current_span)
                
                for span in combined_spans:
                    text = normalize(span["text"])
                    if not text or is_noise(text):
                        continue
                    
                    size = span["size"]
                    y0, y1 = span["bbox"][1], span["bbox"][3]
                    is_bold = bool(span.get("flags", 0) & 2)
                    
                    letters = [c for c in text if c.isalpha()]
                    is_all_upper = bool(letters) and all(c.isupper() for c in letters)
                    
                    font_sizes.append(size)
                    raw.append((text, size, y0, y1, p+1, is_bold, is_all_upper))

    if not raw:
        return "", []

    levels = cluster_font_sizes(font_sizes)
    
    # Merge nearby lines with similar properties
    raw.sort(key=lambda x: (x[4], x[2]))  # sort by page then y0
    merged = []
    
    if raw:
        current = list(raw[0])
        
        for next_item in raw[1:]:
            # Check if we should merge with current:
            # Same page, similar size, vertical proximity, and similar styling
            if (next_item[4] == current[4] and 
                abs(next_item[2] - current[3]) < 10 and  # increased from 3 to 10
                abs(next_item[1] - current[1]) < 1.0 and
                next_item[5] == current[5] and 
                next_item[6] == current[6]):
                
                # Merge them
                current[0] += " " + next_item[0]
                current[3] = next_item[3]  # update y1
            else:
                merged.append(tuple(current))
                current = list(next_item)
        
        merged.append(tuple(current))
    
    # Process merged items to identify headings
    title = ""
    outline = []
    first_heading = True
    
    for text, sz, _, _, page, is_bold, is_all_upper in merged:
        if is_noise(text):
            continue
        
        # Determine level based on multiple factors
        level_score = 0
        
        # Size is the primary factor
        if abs(sz - levels[0]) < 1.0:
            level_score += 3
        elif abs(sz - levels[1]) < 1.0:
            level_score += 2
        else:
            level_score += 1
        
        # Bold adds weight
        if is_bold:
            level_score += 1
        
        # All uppercase adds weight
        if is_all_upper:
            level_score += 1
        
        # Title case adds some weight
        if text.istitle():
            level_score += 0.5
        
        # Determine level based on score
        if level_score >= 3.5:
            lvl = "H1"
        elif level_score >= 2.5:
            lvl = "H2"
        else:
            lvl = "H3"
        
        # The first significant H1 becomes the title
        if first_heading and lvl == "H1" and len(text) > 5:
            title = text
            first_heading = False
            continue
        
        # Skip very short headings unless they're H1/H2 and bold
        if len(text.split()) < 2 and len(text) < 8 and not (lvl in ("H1", "H2") and is_bold):
            continue
        
        outline.append({
            "level": lvl,
            "text": text,
            "page": page,
            "size": round(sz, 1),
            "bold": is_bold,
            "all_upper": is_all_upper
        })
    
    # If we didn't find a title, use the first heading
    if not title and outline:
        for item in outline:
            if item["level"] == "H1":
                title = item["text"]
                outline.remove(item)
                break
        else:
            title = outline[0]["text"]
            outline = outline[1:]
    
    return title, outline

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fn in os.listdir(INPUT_DIR):
        if not fn.lower().endswith(".pdf"): 
            continue
        inpath = os.path.join(INPUT_DIR, fn)
        outpath = os.path.join(OUTPUT_DIR, fn[:-4] + ".json")
        try:
            title, outline = extract_headings(inpath)
            with open(outpath, "w", encoding="utf-8") as f:
                json.dump({
                    "title": title,
                    "outline": outline,
                    "source": fn
                }, f, indent=2, ensure_ascii=False)
            print("✅", fn)
        except Exception as e:
            print("❌", fn, str(e))

if __name__ == "__main__":
    main()



# import os
# import json
# import re
# import unicodedata
# from typing import List, Dict, Optional, Tuple
# from pydantic import BaseModel
# import pdfplumber
# import numpy as np
# from PIL import Image
# import pytesseract
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sentence_transformers import SentenceTransformer
# import cv2
# from deskew import determine_skew
# import math

# class Heading(BaseModel):
#     level: str  # H1, H2, H3
#     text: str
#     page: int
#     bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
#     confidence: float = 1.0

# class DocumentOutline(BaseModel):
#     title: str
#     outline: List[Heading]

# class PDFOutlineExtractor:
#     def __init__(self):
#         # Initialize models
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.tfidf = TfidfVectorizer(stop_words=None, max_features=1000)
#         self.heading_embeddings = {}

#         # Enhanced heading detection configuration
#         self.heading_config = {
#             'H1': {
#                 'min_size': 18,
#                 'patterns': [
#                     r'^(?:[A-Z][A-Z0-9\s\-]+|[IVXLCDM]+\.\s+[A-Z])$',
#                     r'^[\u4e00-\u9fff]{2,}[\u4e00-\u9fff、。\s]*$',
#                     r'^第[一二三四五六七八九十]+章\s+[\u4e00-\u9fff]+'
#                 ],
#                 'position': ('top', 0.2),  # Top 20% of page
#                 'alignment': ('center', 0.8),  # 80% confidence in center alignment
#                 'line_spacing': 1.5
#             },
#             'H2': {
#                 'min_size': 14,
#                 'patterns': [
#                     r'^(?:[A-Z][a-zA-Z0-9\s\-]+|\d+\.\s+[A-Z][a-z])$',
#                     r'^[\u4e00-\u9fff]{2,}[\u4e00-\u9fff、。\s]*$',
#                     r'^第[一二三四五六七八九十]+节\s+[\u4e00-\u9fff]+'
#                 ],
#                 'position': ('any', 0.5),
#                 'alignment': ('left', 0.6),
#                 'line_spacing': 1.2
#             },
#             'H3': {
#                 'min_size': 12,
#                 'patterns': [
#                     r'^(?:[a-z][a-z0-9\s\-]+|\d+\.\d+\.\s+[A-Za-z])$',
#                     r'^[\u4e00-\u9fff]{2,}[\u4e00-\u9fff、。\s]*$'
#                 ],
#                 'position': ('any', 0.3),
#                 'alignment': ('left', 0.4),
#                 'line_spacing': 1.0
#             }
#         }

#         # OCR configuration
#         self.ocr_config = {
#             'lang': 'eng+chi_sim+jpn',
#             'config': '--psm 6 --oem 3',
#             'preprocess': True
#         }

#     def extract_outline(self, pdf_path: str) -> DocumentOutline:
#         """Enhanced outline extraction with multi-modal approach"""
#         title = os.path.basename(pdf_path).replace('.pdf', '')
#         outline = []
#         title_candidates = []
#         all_text = []
#         visual_elements = []
        
#         with pdfplumber.open(pdf_path) as pdf:
#             for page_num, page in enumerate(pdf.pages, 1):
#                 try:
#                     # Extract text and visual elements
#                     page_data = self._extract_page_data(page, page_num)
#                     all_text.append(page_data['text'])
#                     visual_elements.extend(page_data['visual_elements'])
                    
#                     # Process extracted content
#                     headings = self._find_headings(
#                         text=page_data['text'],
#                         visual_elements=page_data['visual_elements'],
#                         page_num=page_num
#                     )
#                     outline.extend(headings)
                    
#                     # Collect title candidates
#                     title_candidates.extend(h.text for h in headings if h.level == 'H1')
#                 except Exception as e:
#                     print(f"Error processing page {page_num}: {str(e)}")
#                     continue
        
#         # Enhanced semantic analysis
#         if all_text:
#             self._enhance_with_semantics(outline, all_text, visual_elements)
        
#         # Structural analysis and filtering
#         outline = self._analyze_structure(outline)
        
#         # Select best title
#         if title_candidates:
#             title = self._select_best_title(title_candidates)
            
#         return DocumentOutline(title=title, outline=outline)

#     def _extract_page_data(self, page, page_num: int) -> Dict:
#         """Multi-modal page data extraction"""
#         result = {
#             'text': '',
#             'visual_elements': [],
#             'page_num': page_num
#         }
        
#         # 1. Text extraction
#         result['text'] = self._extract_text(page)
        
#         # 2. Visual element extraction
#         if len(result['text'].strip()) < 100:  # Threshold for OCR fallback
#             img_data = self._process_page_image(page, page_num)
#             result['text'] += " " + img_data['text']
#             result['visual_elements'] = img_data['visual_elements']
        
#         return result

#     def _extract_text(self, page) -> str:
#         """Extract text with layout preservation"""
#         text = page.extract_text(
#             layout=True,
#             x_density=7.0,
#             y_density=7.0
#         ) or ""
        
#         # Extract from tables
#         tables = page.extract_tables()
#         for table in tables:
#             for row in table:
#                 text += " ".join(str(cell) for cell in row if cell) + "\n"
        
#         return text.strip()

#     def _process_page_image(self, page, page_num: int) -> Dict:
#         """Advanced image processing for OCR"""
#         result = {
#             'text': '',
#             'visual_elements': []
#         }
        
#         try:
#             # Convert to high-res image
#             img = page.to_image(resolution=300).original
#             pil_img = Image.fromarray(img)
            
#             # Preprocess image
#             processed_img = self._preprocess_image(pil_img)
            
#             # OCR with layout analysis
#             ocr_data = pytesseract.image_to_data(
#                 processed_img,
#                 lang=self.ocr_config['lang'],
#                 config=self.ocr_config['config'],
#                 output_type=pytesseract.Output.DICT
#             )
            
#             # Process OCR results
#             text_blocks = []
#             for i in range(len(ocr_data['text'])):
#                 if ocr_data['text'][i].strip():
#                     block = {
#                         'text': ocr_data['text'][i],
#                         'bbox': (
#                             ocr_data['left'][i],
#                             ocr_data['top'][i],
#                             ocr_data['left'][i] + ocr_data['width'][i],
#                             ocr_data['top'][i] + ocr_data['height'][i]
#                         ),
#                         'conf': ocr_data['conf'][i] / 100.0,
#                         'block_num': ocr_data['block_num'][i],
#                         'line_num': ocr_data['line_num'][i]
#                     }
#                     text_blocks.append(block)
            
#             # Group by lines and blocks
#             grouped = self._group_text_blocks(text_blocks)
#             result['visual_elements'] = grouped
#             result['text'] = " ".join(b['text'] for b in grouped)
            
#         except Exception as e:
#             print(f"Image processing failed on page {page_num}: {str(e)}")
        
#         return result

#     def _preprocess_image(self, image: Image.Image) -> Image.Image:
#         """Enhance image for better OCR results"""
#         try:
#             # Convert to grayscale
#             img = np.array(image.convert('L'))
            
#             # Deskew
#             angle = determine_skew(img)
#             if angle:
#                 img = self._rotate_image(img, angle)
            
#             # Enhance contrast
#             img = cv2.equalizeHist(img)
            
#             # Denoise
#             img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
            
#             # Binarize
#             _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
#             return Image.fromarray(img)
#         except:
#             return image

#     def _rotate_image(self, image: np.array, angle: float) -> np.array:
#         """Rotate image to correct skew"""
#         h, w = image.shape
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     def _group_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
#         """Group text blocks into logical elements"""
#         # Sort by vertical position
#         blocks.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))
        
#         grouped = []
#         current_group = None
        
#         for block in blocks:
#             if not current_group:
#                 current_group = {
#                     'text': block['text'],
#                     'bbox': block['bbox'],
#                     'conf': [block['conf']],
#                     'lines': 1
#                 }
#             else:
#                 # Check if same line (similar y-position)
#                 y_diff = abs(block['bbox'][1] - current_group['bbox'][1])
#                 if y_diff < 10:  # Same line
#                     current_group['text'] += " " + block['text']
#                     current_group['bbox'] = (
#                         min(current_group['bbox'][0], block['bbox'][0]),
#                         min(current_group['bbox'][1], block['bbox'][1]),
#                         max(current_group['bbox'][2], block['bbox'][2]),
#                         max(current_group['bbox'][3], block['bbox'][3])
#                     )
#                     current_group['conf'].append(block['conf'])
#                 else:
#                     # New line
#                     grouped.append(current_group)
#                     current_group = {
#                         'text': block['text'],
#                         'bbox': block['bbox'],
#                         'conf': [block['conf']],
#                         'lines': 1
#                     }
        
#         if current_group:
#             grouped.append(current_group)
        
#         # Calculate average confidence
#         for group in grouped:
#             group['conf'] = sum(group['conf']) / len(group['conf'])
        
#         return grouped

#     def _find_headings(self, text: str, visual_elements: List[Dict], page_num: int) -> List[Heading]:
#         """Multi-modal heading detection"""
#         headings = []
        
#         # 1. Process text-based headings
#         text_headings = self._find_text_headings(text, page_num)
#         headings.extend(text_headings)
        
#         # 2. Process visual elements
#         visual_headings = self._find_visual_headings(visual_elements, page_num)
#         headings.extend(visual_headings)
        
#         return headings

#     def _find_text_headings(self, text: str, page_num: int) -> List[Heading]:
#         """Find headings in extracted text"""
#         headings = []
#         lines = [line.strip() for line in text.split('\n') if line.strip()]
        
#         for line in lines:
#             heading_level = self._classify_heading(line)
#             if heading_level:
#                 clean_text = self._clean_text(line)
#                 if clean_text:
#                     headings.append(Heading(
#                         level=heading_level,
#                         text=clean_text,
#                         page=page_num,
#                         confidence=0.9  # High confidence for direct text
#                     ))
#         return headings

#     def _find_visual_headings(self, elements: List[Dict], page_num: int) -> List[Heading]:
#         """Find headings in visual elements"""
#         headings = []
        
#         for element in elements:
#             if element['conf'] < 0.6:  # Skip low-confidence elements
#                 continue
                
#             text = element['text'].strip()
#             if not text:
#                 continue
                
#             # Check visual characteristics
#             bbox = element['bbox']
#             width = bbox[2] - bbox[0]
#             height = bbox[3] - bbox[1]
#             aspect_ratio = width / height if height > 0 else 1
            
#             # Check if element looks like a heading
#             heading_level = self._classify_visual_heading(text, element)
#             if heading_level:
#                 headings.append(Heading(
#                     level=heading_level,
#                     text=self._clean_text(text),
#                     page=page_num,
#                     bbox=bbox,
#                     confidence=element['conf'] * 0.8  # Slightly reduce confidence for visual
#                 ))
        
#         return headings

#     def _classify_heading(self, text: str) -> Optional[str]:
#         """Classify text as heading using multiple heuristics"""
#         # Skip non-heading patterns
#         if self._is_non_heading(text):
#             return None
            
#         # Check each heading level
#         for level, config in self.heading_config.items():
#             # Pattern matching
#             for pattern in config['patterns']:
#                 if re.fullmatch(pattern, text, re.UNICODE):
#                     # Additional linguistic validation
#                     if self._validate_heading_content(text, level):
#                         return level
#         return None

#     def _classify_visual_heading(self, text: str, element: Dict) -> Optional[str]:
#         """Classify visual element as heading"""
#         if self._is_non_heading(text):
#             return None
            
#         bbox = element['bbox']
#         page_width = 600  # Approximate, can be adjusted
#         x_center = (bbox[0] + bbox[2]) / 2
        
#         for level, config in self.heading_config.items():
#             # Check position
#             pos_type, pos_thresh = config['position']
#             if pos_type == 'top' and bbox[1] > pos_thresh * 1000:  # Assuming page height ~1000
#                 continue
                
#             # Check alignment
#             align_type, align_thresh = config['alignment']
#             if align_type == 'center':
#                 center_dist = abs(x_center - page_width/2) / (page_width/2)
#                 if center_dist > (1 - align_thresh):
#                     continue
#             elif align_type == 'left' and bbox[0] > page_width * 0.2:  # Too far right
#                 continue
                
#             # Check text patterns
#             for pattern in config['patterns']:
#                 if re.fullmatch(pattern, text, re.UNICODE):
#                     if self._validate_heading_content(text, level):
#                         return level
#         return None

#     def _is_non_heading(self, text: str) -> bool:
#         """Check if text is definitely not a heading"""
#         return any(re.search(pattern, text) for pattern in [
#             r'^\d+$',  # Pure numbers
#             r'^[\W_]+$',  # Only symbols
#             r'^[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u309F\u30A0-\u30FF]',  # Non-text
#             r'^http[s]?://',  # URLs
#             r'^\s*$',  # Empty
#             r'^.{0,2}$'  # Too short
#         ]) or len(text) > 200  # Too long

#     def _validate_heading_content(self, text: str, level: str) -> bool:
#         """Validate heading content based on level"""
#         # Must contain meaningful text
#         alpha_count = sum(1 for c in text if c.isalpha())
#         if alpha_count / len(text) < 0.3:  # Less than 30% letters
#             return False
            
#         # Level-specific checks
#         if level == 'H1':
#             return len(text.split()) >= 1 and len(text) >= 3
#         elif level == 'H2':
#             return len(text.split()) >= 1 and len(text) >= 2
#         return True  # H3 can be shorter

#     def _clean_text(self, text: str) -> str:
#         """Clean text while preserving multilingual characters"""
#         text = unicodedata.normalize('NFKC', text)
#         text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
#         text = re.sub(r'^\W+|\W+$', '', text)
#         return ' '.join(text.split())

#     def _enhance_with_semantics(self, outline: List[Heading], all_text: List[str], visual_elements: List[Dict]):
#         """Enhanced semantic analysis with visual context"""
#         try:
#             # Train TF-IDF on all document text
#             self.tfidf.fit(all_text)
            
#             # Calculate importance scores
#             for heading in outline:
#                 if heading.text not in self.heading_embeddings:
#                     # Text embedding
#                     text_emb = self.embedding_model.encode(heading.text)
                    
#                     # Visual features (if available)
#                     vis_feats = np.zeros(5)
#                     if heading.bbox:
#                         vis_feats = np.array([
#                             heading.bbox[0],  # x0
#                             heading.bbox[1],  # y0
#                             (heading.bbox[2] - heading.bbox[0]),  # width
#                             (heading.bbox[3] - heading.bbox[1]),  # height
#                             heading.confidence
#                         ])
                    
#                     # Combine features
#                     combined = np.concatenate([text_emb, vis_feats])
#                     self.heading_embeddings[heading.text] = combined
#         except Exception as e:
#             print(f"Enhanced semantic analysis failed: {str(e)}")

#     def _analyze_structure(self, outline: List[Heading]) -> List[Heading]:
#         """Analyze document structure to validate headings"""
#         # Group by page and sort by position
#         outline.sort(key=lambda h: (h.page, h.bbox[1] if h.bbox else 0))
        
#         # Filter based on structural patterns
#         filtered = []
#         prev_level = None
        
#         for heading in outline:
#             # Skip duplicates
#             if any(h.text == heading.text for h in filtered):
#                 continue
                
#             # Validate level progression
#             if prev_level:
#                 curr_level_num = int(heading.level[1:])
#                 prev_level_num = int(prev_level[1:])
#                 if curr_level_num > prev_level_num + 1:  # Skip invalid jumps
#                     continue
                    
#             filtered.append(heading)
#             prev_level = heading.level
        
#         return filtered

#     def _select_best_title(self, candidates: List[str]) -> str:
#         """Select title using multi-modal features"""
#         if not candidates:
#             return ""
            
#         # Simple fallback
#         if not self.heading_embeddings:
#             return max(candidates, key=len)
            
#         # Score candidates
#         scored = []
#         for text in candidates:
#             if text in self.heading_embeddings:
#                 features = self.heading_embeddings[text]
#                 # Score based on position (y-coordinate) and semantic importance
#                 score = -features[-4] + np.linalg.norm(features[:-5])  # Lower y is better
#                 scored.append((text, score))
        
#         return max(scored, key=lambda x: x[1], default=(candidates[0], 0))[0]

# def process_pdfs(input_dir: str = 'input', output_dir: str = 'output'):
#     """Process all PDFs with comprehensive error handling"""
#     os.makedirs(output_dir, exist_ok=True)
#     extractor = PDFOutlineExtractor()
    
#     for filename in sorted(os.listdir(input_dir)):
#         if filename.lower().endswith('.pdf'):
#             pdf_path = os.path.join(input_dir, filename)
#             try:
#                 outline = extractor.extract_outline(pdf_path)
#                 output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                
#                 with open(output_path, 'w', encoding='utf-8') as f:
#                     json.dump({
#                         'title': outline.title,
#                         'outline': [h.dict() for h in outline.outline]
#                     }, f, indent=2, ensure_ascii=False)
                    
#                 print(f"Successfully processed {filename}")
#             except Exception as e:
#                 print(f"Failed to process {filename}: {str(e)}")

# if __name__ == "__main__":
#     input_dir = os.getenv('INPUT_DIR', 'input')
#     output_dir = os.getenv('OUTPUT_DIR', 'output')
#     process_pdfs(input_dir, output_dir)