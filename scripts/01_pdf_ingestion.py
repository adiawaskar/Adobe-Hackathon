#!/usr/bin/env python
# coding: utf-8

# In[84]:


from pathlib import Path
import os
import sys
import fitz  # PyMuPDF
import json
import pytesseract
import easyocr  # ✅ Add this here
import numpy as np
from pdf2image import convert_from_path
from PIL import Image

# (Windows fix)
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Determine if script is run via command line or in a notebook
if "__file__" in globals():
    # Running as a script (e.g., from main.py)
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PDF_PATH = PROJECT_ROOT / "input" / pdf_arg
else:
    # Running inside a Jupyter notebook
    PROJECT_ROOT = Path().resolve()
    PDF_PATH = PROJECT_ROOT / "input" / "sample.pdf"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pdf_text_blocks.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# In[85]:


def extract_text_blocks(pdf_path: Path):
    doc = fitz.open(pdf_path)
    blocks = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = " ".join([span["text"] for span in spans]).strip()
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


# In[86]:


reader = easyocr.Reader(['en'], gpu=False)

def extract_text_ocr(pdf_path: Path):
    ocr_blocks = []
    images = convert_from_path(pdf_path)

    for page_num, img in enumerate(images, start=1):
        img_np = np.array(img)
        result = reader.readtext(img_np)

        for (bbox, text, conf) in result:
            ocr_blocks.append({
                "page": page_num,
                "text": text.strip(),
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




