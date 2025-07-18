#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell 1: Imports and Path Setup
import json
import pandas as pd
from pathlib import Path

# Define root and output paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MERGED_PATH = PROJECT_ROOT / "output" / "03_merged_headings.json"
OUTPUT_PATH = PROJECT_ROOT / "output" / "structured_outline.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# In[21]:


with open(MERGED_PATH, "r", encoding="utf-8") as f:
    merged = json.load(f)

df = pd.DataFrame(merged)
print("✅ Loaded", len(df), "merged heading candidates")
df.head(2)


# In[22]:


def is_title(text_block):
    return (
        text_block["page"] == 1 and
        text_block["font_size_avg"] >= 20 and
        text_block["word_count"] <= 10
    )

def get_level(block):
    size = block["font_size_avg"]
    if size >= 18:
        return "H1"
    elif size >= 14:
        return "H2"
    else:
        return "H3"


# In[23]:


title_block = next((b for b in merged if is_title(b)), None)
title = title_block["text"] if title_block else "Untitled"
print("📘 Detected Title:", title)


# In[24]:


for block in merged:
    block["level"] = get_level(block)

# Preview first few levels
for i, b in enumerate(merged[:5], 1):
    print(f"[{i}] {b['level']} | Page {b['page']}: {b['text']}")


# In[25]:


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


# In[26]:


# Preview first 1000 characters (optional full print)
print(json.dumps(structured, indent=2, ensure_ascii=False)[:1000])

