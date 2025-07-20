#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

# Determine context and set paths
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path().resolve()

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pdf_text_blocks.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pdf_blocks_features.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load original blocks
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    blocks = json.load(f)


# In[ ]:


def extract_features(blocks):
    # Collect all font sizes to compute median
    all_font_sizes = [fs for blk in blocks for fs in blk.get("font_sizes", [])]
    median_font_size = np.median(all_font_sizes) if all_font_sizes else 12

    features = []

    for block in blocks:
        text = block["text"]
        font_sizes = block.get("font_sizes", [])
        font_flags = block.get("font_flags", [])
        bbox = block.get("bbox", [0, 0, 0, 0])

        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0

        avg_font_size = round(np.mean(font_sizes), 1) if font_sizes else 0
        norm_font = round(avg_font_size / median_font_size, 2) if median_font_size else 1.0

        features.append({
            "page": block["page"],
            "text": text,
            "font_size_avg": avg_font_size,
            "norm_font": norm_font,  # ✅ Added
            "is_bold": any(flag in [1, 20, 21] for flag in font_flags),
            "is_all_caps": text.isupper(),
            "y_position_norm": round(y0 / 1000, 3),  # Normalize for comparison
            "word_count": len(text.split()),
            "char_count": len(text),
            "starts_with_number": text.strip()[0].isdigit() if text.strip() else False,
            "ends_with_colon": text.strip().endswith(":"),
            "is_centered": 0.4 < (x0 / 600) < 0.6  # rough center range (assuming width ~600)
        })

    return features


# In[14]:


block_features = extract_features(blocks)
df = pd.DataFrame(block_features)
df.head()


# In[15]:


# Save enhanced features as JSON
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(block_features, f, indent=2, ensure_ascii=False)

# Also save CSV for easier inspection
df.to_csv(OUTPUT_PATH.with_suffix(".csv"), index=False)


print(f"✅ Saved {len(block_features)} blocks with features.")


# In[16]:


# Show top candidates by font size and position
df.sort_values(by=["font_size_avg", "y_position_norm"], ascending=[False, True]).head(10)[
    ["text", "font_size_avg", "is_bold", "is_all_caps", "word_count"]
]

