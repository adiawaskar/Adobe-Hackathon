#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
import easyocr
from pdf2image import convert_from_path
from PIL import Image

# Setting up paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tesseract path for Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# OCR reader setup
LANGUAGES = ["en"]
reader = easyocr.Reader(LANGUAGES, gpu=False)

# Cleans unicode formatting characters from text
def clean_text(text):
    return re.sub(r'[\u202a-\u202e]', '', text)

# Extracts text blocks using PyMuPDF
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
    return blocks

# Extracts text using EasyOCR as fallback
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

# Extracts layout and formatting features
def extract_features(blocks):
    all_font_sizes = [fs for blk in blocks for fs in blk.get("font_sizes", [])]
    median_font_size = np.median(all_font_sizes) if all_font_sizes else 12
    features = []
    for block in blocks:
        text = block["text"]
        font_sizes = block.get("font_sizes", [])
        font_flags = block.get("font_flags", [])
        bbox = block.get("bbox", [0, 0, 0, 0])
        x0, y0, x1, y1 = bbox
        avg_font_size = round(np.mean(font_sizes), 1) if font_sizes else 0
        norm_font = round(avg_font_size / median_font_size, 2) if median_font_size else 1.0
        features.append({
            "page": block["page"],
            "text": text,
            "font_size_avg": avg_font_size,
            "norm_font": norm_font,
            "is_bold": any(flag in [1, 20, 21] for flag in font_flags),
            "is_all_caps": text.isupper(),
            "y_position_norm": round(y0 / 1000, 3),
            "word_count": len(text.split()),
            "char_count": len(text),
            "starts_with_number": text.strip()[0].isdigit() if text.strip() else False,
            "ends_with_colon": text.strip().endswith(":"),
            "is_centered": 0.4 < (x0 / 600) < 0.6,
            "bbox": bbox
        })
    return features

# Rules to detect headings
def is_heading(block):
    return (
        block.get("norm_font", 0) >= 0.9 and
        block["word_count"] <= 12 and
        block["is_bold"] and
        not block["text"].endswith(".") and
        not block["text"][0].islower()
    )

# Merges nearby heading-like blocks into one
def merge_headings_flex(df):
    merged_blocks = []
    i = 0
    while i < len(df):
        current = df.iloc[i].to_dict()
        merged_text = current["text"]
        j = i + 1
        x0_curr = current.get("bbox", [0])[0]
        while j < len(df):
            nxt = df.iloc[j].to_dict()
            x0_next = nxt.get("bbox", [0])[0]
            close_y = abs(nxt.get("y_position_norm", 0) - current.get("y_position_norm", 0)) < 0.03
            same_font = abs(nxt.get("font_size_avg", 0) - current.get("font_size_avg", 0)) < 1
            starts_lower = nxt.get("text", "").strip().startswith(tuple("abcdefghijklmnopqrstuvwxyz"))
            same_indent = abs(x0_next - x0_curr) < 20
            if close_y and same_font and (starts_lower or same_indent):
                merged_text += " " + nxt.get("text", "")
                j += 1
            else:
                break
        current["text"] = merged_text
        merged_blocks.append(current)
        i = j
    return merged_blocks

# Assigns heading level based on font size
def get_level(block):
    size = block.get("font_size_avg", 0)
    if size >= 18:
        return "H1"
    elif size >= 14:
        return "H2"
    else:
        return "H3"

# Detects title from H1 block on first page
def is_title(block):
    return (
        block.get("page") == 1 and
        block.get("font_size_avg", 0) >= 18 and
        block.get("level") == "H1" and
        block.get("word_count", 0) <= 12 and
        len(block.get("text", "")) > 5
    )

# Main function to run full pipeline on a PDF
def process_pdf(pdf_path: Path):
    pdf_stem = pdf_path.stem
    text_blocks = extract_text_blocks(pdf_path)
    if not text_blocks:
        text_blocks = extract_text_ocr(pdf_path)
    with open(DATA_DIR / f"{pdf_stem}_blocks.json", "w", encoding="utf-8") as f:
        json.dump(text_blocks, f, indent=2, ensure_ascii=False)

    features = extract_features(text_blocks)
    with open(DATA_DIR / f"{pdf_stem}_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(features)
    df["is_heading"] = df.apply(is_heading, axis=1)
    merged = merge_headings_flex(df)
    with open(OUTPUT_DIR / f"{pdf_stem}_merged_headings.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    for block in merged:
        block["level"] = get_level(block)
    title_block = next((b for b in merged if is_title(b)), None)
    title = title_block["text"] if title_block else "Untitled"
    structured = {
        "title": title,
        "outline": [
            {"level": b["level"], "text": b["text"], "page": b["page"]} for b in merged
        ]
    }
    with open(OUTPUT_DIR / f"{pdf_stem}.json", "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)
    print(f"✅ {pdf_path.name} processed. Title: {title}")

# Entry point to process all PDFs in input directory
if __name__ == "__main__":
    input_pdfs = list(INPUT_DIR.glob("*.pdf"))
    if not input_pdfs:
        print("No PDFs found in /input directory.")
        sys.exit(1)
    for pdf_path in input_pdfs:
        process_pdf(pdf_path)
        print(f"Processed: {pdf_path.name}")
    print("All PDFs processed successfully!")