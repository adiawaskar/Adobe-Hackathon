#!/usr/bin/env python
# coding: utf-8

# In[24]:


import json
from pathlib import Path
import pandas as pd
import os

# Handle path resolution for both script and notebook
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
else:
    PROJECT_ROOT = Path().resolve()

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

INPUT_PATH = DATA_DIR / "pdf_blocks_features.json"
OUTPUT_PATH = OUTPUT_DIR / "03_merged_headings.json"

# Load enriched blocks
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    blocks = json.load(f)

df = pd.DataFrame(blocks)
print(f"Loaded {len(df)} blocks")
df.head()


# In[25]:


def is_heading(block):
    return (
        block["font_size_avg"] >= 13.0 and
        block["word_count"] <= 10 and
        block["is_bold"] and
        not block["text"].endswith(".") and
        not block["text"][0].islower()
    )


# In[26]:


df["is_heading"] = df.apply(is_heading, axis=1)
df[["text", "is_heading"]].head(10)


# In[27]:


def merge_headings_flex(df):
    merged_blocks = []
    i = 0
    while i < len(df):
        current = df.iloc[i].to_dict()
        merged_text = current["text"]
        j = i + 1

        while j < len(df):
            nxt = df.iloc[j].to_dict()

            close_y = abs(nxt["y_position_norm"] - current["y_position_norm"]) < 0.025
            same_font = abs(nxt["font_size_avg"] - current["font_size_avg"]) < 1
            starts_lower = nxt["text"][0].islower()

            if close_y and same_font and starts_lower:
                merged_text += " " + nxt["text"]
                j += 1
            else:
                break

        current["text"] = merged_text
        merged_blocks.append(current)
        i = j

    return merged_blocks


# In[28]:


merged_headings = merge_headings_flex(df)  # Use all blocks
print(f"Detected and merged {len(merged_headings)} headings")


# In[29]:


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save final headings
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(merged_headings, f, indent=2, ensure_ascii=False)

# Preview
for i, h in enumerate(merged_headings[:10], 1):
    print(f"[{i}] Page {h['page']}: {h['text']}")


# In[30]:


# import os
# import json

# # Go to root directory no matter where notebook is running from
# ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
# OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# OUTFILE = os.path.join(OUTPUT_DIR, "03_merged_headings.json")

# with open(OUTFILE, "w", encoding="utf-8") as f:
#     json.dump(merged_headings, f, indent=2, ensure_ascii=False)

# print("✅ Saved to:", OUTFILE)


