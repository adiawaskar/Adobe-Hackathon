#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import os
import sys
import fitz  # PyMuPDF
import json
import pytesseract
import easyocr
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import re  # 📌 Add this if not already imported

# Utility to remove invisible Unicode directional formatting characters
def clean_text(text):
    return re.sub(r'[\u202a-\u202e]', '', text)


# Define project root and make sure it's in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import language settings
from config import LANGUAGES

# Tesseract path for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Determine PDF path based on whether it's a script or notebook
if "__file__" in globals():
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    PDF_PATH = PROJECT_ROOT / "input" / pdf_arg
else:
    # Likely inside Jupyter
    PDF_PATH = Path().resolve() / "input" / "sample.pdf"

# Define output path
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pdf_text_blocks.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# In[ ]:


def extract_text_blocks(pdf_path: Path):
    doc = fitz.open(pdf_path)
    blocks = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = clean_text(" ".join([span["text"] for span in spans]).strip())
                if not text:
                    continue
                font_sizes = list(set([round(span["size"], 1) for span in spans]))
                font_flags = list(set([span["flags"] for span in spans]))
                blocks.append({
                    "page": page_num,
                    "text": text,
                    "font_sizes": font_sizes,
                    "font_flags": font_flags,
                    "bbox": block.get("bbox")
                })
    print(f"Extracted {len(blocks)} text blocks from {pdf_path.name}")
    return blocks


# In[ ]:


from config import LANGUAGES
reader = easyocr.Reader(LANGUAGES, gpu=False)

def extract_text_ocr(pdf_path: Path):
    ocr_blocks = []
    images = convert_from_path(pdf_path)

    for page_num, img in enumerate(images, start=1):
        img_np = np.array(img)
        result = reader.readtext(img_np)

        for (bbox, text, conf) in result:
            ocr_blocks.append({
                "page": page_num,
                "text": clean_text(text.strip()),
                "bbox": bbox,
                "confidence": round(conf, 2)
            })

    return ocr_blocks


# In[87]:


# Try PyMuPDF first
text_blocks = extract_text_blocks(PDF_PATH)

# Fallback to OCR if no blocks were extracted
if not text_blocks:
    print("No blocks found with PyMuPDF. Falling back to OCR...")
    text_blocks = extract_text_ocr(PDF_PATH)

# Save output
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(text_blocks, f, indent=2, ensure_ascii=False)

print(f"✅ Extracted {len(text_blocks)} blocks from: {PDF_PATH.name}")
print(f"📁 Saved to: {OUTPUT_PATH}")


# In[88]:


for i, block in enumerate(text_blocks[:10], 1):
    print(f"[{i}] Page {block['page']}:\n{block['text']}\n{'-'*40}")


# In[ ]:




