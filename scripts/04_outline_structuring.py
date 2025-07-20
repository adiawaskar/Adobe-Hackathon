#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell 1: Imports and Path Setup
import json
import pandas as pd
from pathlib import Path
import os

# Resolve project root safely
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJECT_ROOT = Path(os.getcwd()).resolve().parent

# Define input/output paths
# MERGED_PATH = PROJECT_ROOT / "output" / "03_merged_headings.json"
# OUTPUT_PATH = PROJECT_ROOT / "output" / "structured_outline.json"

# Define input/output paths
MERGED_PATH = PROJECT_ROOT / "output" / "03_merged_headings.json"

# Dynamically derive JSON name based on input PDF name stored temporarily
pdf_name = os.environ.get("PDF_NAME", "structured_outline")
pdf_stem = Path(pdf_name).stem
OUTPUT_PATH = PROJECT_ROOT / "output" / f"{pdf_stem}.json"


# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Debug info (optional)
print("📁 Project Root:", PROJECT_ROOT)
print("📥 Merged Headings Path:", MERGED_PATH)
print("📤 Structured Outline Path:", OUTPUT_PATH)


# In[43]:


with open(MERGED_PATH, "r", encoding="utf-8") as f:
    merged = json.load(f)

df = pd.DataFrame(merged)
print("✅ Loaded", len(df), "merged heading candidates")
df.head(2)


# In[ ]:


def is_title(text_block):
    return (
        text_block["page"] == 1 and
        text_block["font_size_avg"] >= 20 and
        text_block["word_count"] <= 10
    )

def get_level(block):
    size = block.get("font_size_avg", 0)
    if size >= 18:
        return "H1"
    elif size >= 14:
        return "H2"
    else:
        return "H3"


# In[ ]:


def is_title(block):
    return (
        block.get("page") == 1 and
        block.get("font_size_avg", 0) >= 18 and
        block.get("level") == "H1" and
        block.get("word_count", 0) <= 12 and
        len(block.get("text", "")) > 5
    )



# In[ ]:


# In[45]:

# Assign heading levels before identifying title
for block in merged:
    block["level"] = get_level(block)

title_block = next((b for b in merged if is_title(b)), None)
title = title_block["text"] if title_block else "Untitled"
print("📘 Detected Title:", title)


# In[47]:


structured = {
    "title": title,
    "outline": [
        {
            "level": block["level"],
            "text": block["text"],
            "page": block["page"]
        }
        for block in merged
    ]
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(structured, f, indent=2, ensure_ascii=False)

print(f"✅ Structured JSON saved to:\n{OUTPUT_PATH}")


# In[48]:


# Preview first 1000 characters (optional full print)
print(json.dumps(structured, indent=2, ensure_ascii=False)[:1000])

