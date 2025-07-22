# import json
# from pathlib import Path
# import re
# import fitz  # PyMuPDF
# import numpy as np
# from collections import Counter, defaultdict
# import statistics

# class PDFProcessor:
#     def __init__(self, pdf_path: Path):
#         self.pdf_path = pdf_path
#         self.document = fitz.open(pdf_path)
#         self.blocks = []
#         self.features = []
#         self.is_form_doc = False
#         self.page_dimensions = {}
#         self.title_candidates = []

#     def process(self) -> dict:
#         self._extract_text_blocks()
#         self._engineer_features()
#         outline = self._structure_outline()
#         return outline

#     def _clean_text(self, text: str) -> str:
#         text = re.sub(r'[\u202a-\u202e\u00ad]', '', text)  # Remove special chars
#         text = re.sub(r'\s+', ' ', text)
#         return text.strip()

#     def _normalize_text(self, text: str) -> str:
#         text = text.lower().strip()
#         text = re.sub(r'\s+', ' ', text)
#         text = re.sub(r'[\.:;\-]+$', '', text)
#         return text

#     def _extract_text_blocks(self):
#         for page_num, page in enumerate(self.document, start=1):
#             self.page_dimensions[page_num] = page.rect
#             blocks = page.get_text("dict", sort=True)["blocks"]
            
#             for block in blocks:
#                 if "lines" not in block:
#                     continue
                
#                 # Combine spans in the same line
#                 for line in block["lines"]:
#                     spans = line.get("spans", [])
#                     if not spans:
#                         continue
                    
#                     # Merge spans in the same line
#                     text = " ".join(span["text"] for span in spans)
#                     text = self._clean_text(text)
#                     if not text:
#                         continue
                    
#                     # Collect style information
#                     font_sizes = {round(span["size"], 1) for span in spans}
#                     font_flags = {span["flags"] for span in spans}
#                     fonts = {span["font"] for span in spans}
                    
#                     self.blocks.append({
#                         "page_num": page_num,
#                         "text": text,
#                         "font_sizes": list(font_sizes),
#                         "font_flags": list(font_flags),
#                         "fonts": list(fonts),
#                         "bbox": line["bbox"],
#                         "dir": line["dir"]
#                     })
                    
#                     # Collect title candidates from first page
#                     if page_num == 1:
#                         self.title_candidates.append({
#                             "text": text,
#                             "avg_font_size": statistics.mean(font_sizes) if font_sizes else 0,
#                             "y_pos": line["bbox"][1]
#                         })

#     def _engineer_features(self):
#         if not self.blocks:
#             return
            
#         # Document type detection
#         form_keywords = ['form', 'application', 'questionnaire', 'checklist', 'survey']
#         first_page_text = " ".join(b["text"] for b in self.blocks if b["page_num"] == 1)
#         self.is_form_doc = any(kw in first_page_text.lower() for kw in form_keywords)
        
#         # If not detected by keywords, check numbered line density
#         if not self.is_form_doc:
#             numbered_count = sum(1 for b in self.blocks if re.match(r'^\d+[\.\)]', b["text"]))
#             if numbered_count / max(1, len(self.blocks)) > 0.25:  # 25% threshold
#                 self.is_form_doc = True

#         # Feature engineering
#         all_font_sizes = [size for block in self.blocks for size in block["font_sizes"]]
#         self.median_font_size = np.median(all_font_sizes) if all_font_sizes else 12.0
#         text_counter = Counter(self._normalize_text(block["text"]) for block in self.blocks)
        
#         # Calculate typical line spacing
#         line_spacings = []
#         for i in range(1, len(self.blocks)):
#             if self.blocks[i]["page_num"] == self.blocks[i-1]["page_num"]:
#                 spacing = self.blocks[i]["bbox"][1] - self.blocks[i-1]["bbox"][3]
#                 if spacing > 0:
#                     line_spacings.append(spacing)
#         self.typical_spacing = np.median(line_spacings) if line_spacings else 10
        
#         for i, block in enumerate(self.blocks):
#             text = block["text"]
#             norm_text = self._normalize_text(text)
#             x0, y0, x1, y1 = block["bbox"]
#             page_width = self.page_dimensions[block["page_num"]].width
#             word_count = len(text.split())
            
#             # Vertical spacing calculation
#             prev_gap = 1000  # large default
#             if i > 0 and self.blocks[i-1]["page_num"] == block["page_num"]:
#                 prev_gap = y0 - self.blocks[i-1]["bbox"][3]  # current top - previous bottom
            
#             # Style features
#             is_all_caps = text.isupper() and word_count > 1
#             is_title_case = text.istitle() and word_count > 1 and not is_all_caps
#             numbering_match = re.match(r'^(\d+(?:\.\d+)*[\.\)]?)\s*', text.strip())
#             starts_with_numbering = bool(numbering_match)
            
#             # Numbering depth calculation
#             numbering_depth = 0
#             if starts_with_numbering:
#                 numbering_str = numbering_match.group(1).strip()
#                 numbering_depth = numbering_str.count('.') + 1 if '.' in numbering_str else 1
            
#             ends_with_period = text.strip().endswith('.')
#             avg_font_size = np.mean(block["font_sizes"]) if block["font_sizes"] else self.median_font_size
#             is_bold = any(f & (1 << 4) for f in block["font_flags"])  # 1 << 4 is bold flag
#             size_ratio = avg_font_size / self.median_font_size if self.median_font_size > 0 else 1.0
#             is_centered = abs((x0 + x1) / 2 - page_width / 2) < (0.15 * page_width)
#             is_repeated = text_counter[norm_text] > 2
#             is_short = 2 <= word_count <= 12  # Reasonable heading length
            
#             self.features.append({
#                 **block,
#                 "word_count": word_count,
#                 "is_all_caps": is_all_caps,
#                 "is_title_case": is_title_case,
#                 "starts_with_numbering": starts_with_numbering,
#                 "numbering_depth": numbering_depth,
#                 "ends_with_period": ends_with_period,
#                 "avg_font_size": avg_font_size,
#                 "is_bold": is_bold,
#                 "size_ratio": size_ratio,
#                 "is_centered": is_centered,
#                 "is_repeated": is_repeated,
#                 "norm_text": norm_text,
#                 "prev_gap": prev_gap,
#                 "is_short": is_short,
#                 "gap_ratio": prev_gap / self.typical_spacing if self.typical_spacing > 0 else 1
#             })

#     def _structure_outline(self) -> dict:
#         # Detect dense numbered blocks
#         in_dense_block = [False] * len(self.features)
#         current_streak = 0
#         for i, block in enumerate(self.features):
#             if block["starts_with_numbering"]:
#                 current_streak += 1
#                 if current_streak > 2:  # Mark as dense after 2 consecutive
#                     for j in range(max(0, i - current_streak + 1), i + 1):
#                         in_dense_block[j] = True
#             else:
#                 current_streak = 0

#         # Filter candidates
#         candidates = []
#         for i, block in enumerate(self.features):
#             if in_dense_block[i]:
#                 continue
#             if block["is_repeated"]:
#                 continue
#             if block["word_count"] > 15 or block["word_count"] < 2:  # Reasonable heading length
#                 continue
#             if block["is_all_caps"] and block["word_count"] > 6:  # Long ALLCAPS are usually not headings
#                 continue
#             if self.is_form_doc and block["starts_with_numbering"]:
#                 continue  # Skip numbered lines in forms
#             candidates.append(block)

#         # Remove near-duplicates
#         unique_blocks = []
#         seen = set()
#         for b in candidates:
#             key = (b['norm_text'], b['page_num'])
#             if key in seen:
#                 continue
#             seen.add(key)
#             unique_blocks.append(b)
            
#         if not unique_blocks:
#             return {"title": "No Title Found", "outline": []}

#         # Font size clustering with adaptive levels
#         font_sizes = [b['avg_font_size'] for b in unique_blocks]
#         if len(font_sizes) > 10:
#             # Adaptive clustering based on document
#             clusters = np.percentile(font_sizes, [90, 70, 40])
#         elif len(font_sizes) > 3:
#             clusters = np.percentile(font_sizes, [80, 60, 30])
#         else:
#             clusters = [16, 14, 12]  # Fallback values

#         def assign_level(h):
#             size = h['avg_font_size']
#             if size >= clusters[0]:
#                 return "H1"
#             elif size >= clusters[1]:
#                 return "H2"
#             else:
#                 return "H3"

#         # Score and select headings
#         headings = []
#         for h in unique_blocks:
#             score = 0
            
#             # Visual prominence features
#             if h["size_ratio"] > 1.3: score += 4
#             elif h["size_ratio"] > 1.15: score += 2
#             if h["is_bold"]: score += 2
#             if h["gap_ratio"] > 2.5: score += 3  # Significant whitespace
            
#             # Structural features
#             if h["is_title_case"]: score += 1
#             if h["is_centered"]: score += 1
#             if h["is_short"]: score += 1
            
#             # Numbering handling (only for non-forms)
#             if not self.is_form_doc and h["starts_with_numbering"]:
#                 score += min(2, h["numbering_depth"])  # Limit numbering boost
            
#             # Penalties
#             if h["ends_with_period"]: score -= 3  # Strong penalty for periods
#             if h["word_count"] > 12: score -= 1
#             if h["norm_text"] in ['page', 'continued', 'section']: score -= 5

#             # Form documents require higher threshold
#             min_score = 5 if self.is_form_doc else 4
#             if score >= min_score:
#                 headings.append(h)

#         if not headings:
#             return {"title": "No Title Found", "outline": []}

#         # Title detection - use first page candidates
#         title = "No Title Found"
#         if self.title_candidates:
#             # Filter to top 30% of first page
#             page_height = self.page_dimensions[1].height
#             top_candidates = [b for b in self.title_candidates if b["y_pos"] < page_height * 0.3]
            
#             if not top_candidates:
#                 top_candidates = self.title_candidates
                
#             # Prefer large, centered text
#             title_block = max(top_candidates, 
#                              key=lambda b: (b["avg_font_size"], 
#                                            -abs(b["y_pos"] - page_height * 0.1)))
#             title = self._clean_text(title_block["text"])

#         # Assign levels with improved hierarchy
#         outline_blocks = []
#         for h in headings:
#             # Skip text that matches title
#             if self._normalize_text(h["text"]) == self._normalize_text(title):
#                 continue
                
#             # Use visual hierarchy unless numbering is significant in non-forms
#             if not self.is_form_doc and h["starts_with_numbering"]:
#                 level_map = {1: "H1", 2: "H2", 3: "H3"}
#                 level = level_map.get(min(h["numbering_depth"], 3), "H3")
#             else:
#                 level = assign_level(h)
                
#             outline_blocks.append({
#                 "level": level,
#                 "text": self._clean_text(h["text"]),
#                 "page": h["page_num"]
#             })

#         # Post-process: Merge fragmented headings
#         merged_outline = []
#         prev_block = None
#         for block in outline_blocks:
#             if prev_block and block["page"] == prev_block["page"]:
#                 # Merge if same page and similar style
#                 if abs(len(block["text"]) - len(prev_block["text"])) < 10:
#                     prev_block["text"] += " " + block["text"]
#                     continue
#             merged_outline.append(block)
#             prev_block = block

#         return {"title": title, "outline": merged_outline}


# if __name__ == '__main__':
#     PROJECT_ROOT = Path(__file__).resolve().parent.parent
#     INPUT_DIR = PROJECT_ROOT / "input"
#     OUTPUT_DIR = PROJECT_ROOT / "output"
#     OUTPUT_DIR.mkdir(exist_ok=True)
    
#     pdf_files = list(INPUT_DIR.glob("*.pdf"))
#     if not pdf_files:
#         print(f"No PDF files found in {INPUT_DIR}")
    
#     for pdf_path in pdf_files:
#         print(f"Processing {pdf_path.name}...")
#         processor = PDFProcessor(pdf_path)
#         structured_outline = processor.process()
        
#         output_filename = OUTPUT_DIR / f"{pdf_path.stem}.json"
#         with open(output_filename, "w", encoding="utf-8") as f:
#             json.dump(structured_outline, f, indent=2, ensure_ascii=False)
            
#         print(f"Created outline: {output_filename}")

# print("Processing complete.")

import json
from pathlib import Path
import re
import fitz  # PyMuPDF
import numpy as np
from collections import Counter
import statistics

class PDFProcessor:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.document = fitz.open(pdf_path)
        self.blocks = []
        self.features = []
        self.is_form_doc = False
        self.page_dimensions = {}
        self.title_candidates = []

    def process(self) -> dict:
        self._extract_text_blocks()
        self._engineer_features()
        return self._structure_outline()

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[\u202a-\u202e\u00ad]', '', text)).strip()

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        return re.sub(r'[\.:;\-]+$', '', re.sub(r'\s+', ' ', text))

    def _extract_text_blocks(self):
        for page_num, page in enumerate(self.document, start=1):
            self.page_dimensions[page_num] = page.rect
            for block in page.get_text("dict", sort=True)["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = self._clean_text(" ".join(span["text"] for span in spans))
                    if not text:
                        continue
                    font_sizes = {round(span["size"], 1) for span in spans}
                    font_flags = {span["flags"] for span in spans}
                    fonts = {span["font"] for span in spans}
                    self.blocks.append({
                        "page_num": page_num,
                        "text": text,
                        "font_sizes": list(font_sizes),
                        "font_flags": list(font_flags),
                        "fonts": list(fonts),
                        "bbox": line["bbox"],
                        "dir": line["dir"]
                    })
                    if page_num == 1:
                        self.title_candidates.append({
                            "text": text,
                            "avg_font_size": statistics.mean(font_sizes) if font_sizes else 0,
                            "y_pos": line["bbox"][1]
                        })

    def _engineer_features(self):
        if not self.blocks:
            return

        form_keywords = ['form', 'application', 'questionnaire', 'checklist', 'survey']
        first_page_text = " ".join(b["text"] for b in self.blocks if b["page_num"] == 1).lower()
        self.is_form_doc = any(kw in first_page_text for kw in form_keywords)
        if not self.is_form_doc:
            numbered = sum(1 for b in self.blocks if re.match(r'^\d+[\.\)]', b["text"]))
            self.is_form_doc = numbered / max(1, len(self.blocks)) > 0.25

        all_sizes = [s for b in self.blocks for s in b["font_sizes"]]
        self.median_font_size = np.median(all_sizes) if all_sizes else 12
        text_counts = Counter(self._normalize_text(b["text"]) for b in self.blocks)

        spacings = [
            self.blocks[i]["bbox"][1] - self.blocks[i - 1]["bbox"][3]
            for i in range(1, len(self.blocks))
            if self.blocks[i]["page_num"] == self.blocks[i - 1]["page_num"] and
            self.blocks[i]["bbox"][1] - self.blocks[i - 1]["bbox"][3] > 0
        ]
        self.typical_spacing = np.median(spacings) if spacings else 10

        for i, block in enumerate(self.blocks):
            x0, y0, x1, y1 = block["bbox"]
            text = block["text"]
            norm_text = self._normalize_text(text)
            word_count = len(text.split())
            prev_gap = 1000
            if i > 0 and self.blocks[i - 1]["page_num"] == block["page_num"]:
                prev_gap = y0 - self.blocks[i - 1]["bbox"][3]
            avg_font = np.mean(block["font_sizes"]) if block["font_sizes"] else self.median_font_size
            size_ratio = avg_font / self.median_font_size if self.median_font_size > 0 else 1.0
            self.features.append({
                **block,
                "word_count": word_count,
                "is_all_caps": text.isupper() and word_count > 1,
                "is_title_case": text.istitle() and word_count > 1 and not text.isupper(),
                "starts_with_numbering": bool(re.match(r'^(\d+(?:\.\d+)*[\.)]?)\s*', text)),
                "numbering_depth": len(re.findall(r'\.', text.split()[0])) + 1 if '.' in text.split()[0] else 1,
                "ends_with_period": text.endswith('.'),
                "avg_font_size": avg_font,
                "is_bold": any(f & (1 << 4) for f in block["font_flags"]),
                "size_ratio": size_ratio,
                "is_centered": abs((x0 + x1) / 2 - self.page_dimensions[block["page_num"]].width / 2) < 0.15 * self.page_dimensions[block["page_num"]].width,
                "is_repeated": text_counts[norm_text] > 2,
                "norm_text": norm_text,
                "prev_gap": prev_gap,
                "is_short": 2 <= word_count <= 12,
                "gap_ratio": prev_gap / self.typical_spacing if self.typical_spacing else 1
            })

    def _structure_outline(self) -> dict:
        in_dense_block = [False] * len(self.features)
        streak = 0
        for i, f in enumerate(self.features):
            if f["starts_with_numbering"]:
                streak += 1
                if streak > 2:
                    for j in range(i - streak + 1, i + 1):
                        in_dense_block[j] = True
            else:
                streak = 0

        candidates = [
            f for i, f in enumerate(self.features)
            if not in_dense_block[i] and not f["is_repeated"] and 2 <= f["word_count"] <= 15 and
               not (f["is_all_caps"] and f["word_count"] > 6) and
               not (self.is_form_doc and f["starts_with_numbering"])
        ]

        seen = set()
        unique_blocks = []
        for b in candidates:
            key = (b['norm_text'], b['page_num'])
            if key not in seen:
                seen.add(key)
                unique_blocks.append(b)

        if not unique_blocks:
            return {"title": "No Title Found", "outline": []}

        sizes = [b['avg_font_size'] for b in unique_blocks]
        clusters = np.percentile(sizes, [90, 70, 40]) if len(sizes) > 10 else np.percentile(sizes, [80, 60, 30]) if len(sizes) > 3 else [16, 14, 12]

        def level(size):
            return "H1" if size >= clusters[0] else "H2" if size >= clusters[1] else "H3"

        headings = []
        for h in unique_blocks:
            score = 0
            score += 4 if h["size_ratio"] > 1.3 else 2 if h["size_ratio"] > 1.15 else 0
            score += 2 if h["is_bold"] else 0
            score += 3 if h["gap_ratio"] > 2.5 else 0
            score += 1 if h["is_title_case"] else 0
            score += 1 if h["is_centered"] else 0
            score += 1 if h["is_short"] else 0
            if not self.is_form_doc and h["starts_with_numbering"]:
                score += min(2, h["numbering_depth"])
            score -= 3 if h["ends_with_period"] else 0
            score -= 1 if h["word_count"] > 12 else 0
            score -= 5 if h["norm_text"] in ['page', 'continued', 'section'] else 0
            if score >= (5 if self.is_form_doc else 4):
                headings.append(h)

        if not headings:
            return {"title": "No Title Found", "outline": []}

        page_height = self.page_dimensions[1].height
        title = "No Title Found"
        tops = [c for c in self.title_candidates if c["y_pos"] < page_height * 0.3]
        top_choice = max(tops or self.title_candidates, key=lambda b: (b["avg_font_size"], -abs(b["y_pos"] - page_height * 0.1)))
        title = self._clean_text(top_choice["text"])

        outline = []
        for h in headings:
            if self._normalize_text(h["text"]) == self._normalize_text(title):
                continue
            lvl = ("H1" if h["numbering_depth"] == 1 else "H2" if h["numbering_depth"] == 2 else "H3") if not self.is_form_doc and h["starts_with_numbering"] else level(h["avg_font_size"])
            outline.append({"level": lvl, "text": self._clean_text(h["text"]), "page": h["page_num"]})

        merged = []
        prev = None
        for h in outline:
            if prev and h["page"] == prev["page"] and abs(len(h["text"]) - len(prev["text"])) < 10:
                prev["text"] += " " + h["text"]
            else:
                merged.append(h)
                prev = h

        return {"title": title, "outline": merged}

if __name__ == '__main__':
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INPUT_DIR = PROJECT_ROOT / "input"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    for pdf_path in INPUT_DIR.glob("*.pdf"):
        print(f"Processing {pdf_path.name}...")
        output = PDFProcessor(pdf_path).process()
        with open(OUTPUT_DIR / f"{pdf_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved outline to: {pdf_path.stem}.json")
