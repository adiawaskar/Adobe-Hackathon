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


# In[ ]:


def is_heading(block):
    return (
        block.get("norm_font", 0) >= 0.9 and
        block["word_count"] <= 12 and
        block["is_bold"] and
        not block["text"].endswith(".") and
        not block["text"][0].islower()
    )


# In[26]:


df["is_heading"] = df.apply(is_heading, axis=1)
df[["text", "is_heading"]].head(10)


# In[ ]:


def merge_headings_flex(df):
    merged_blocks = []
    i = 0
    while i < len(df):
        current = df.iloc[i].to_dict()
        merged_text = current["text"]
        j = i + 1

        current_bbox = current.get("bbox", [0, 0, 0, 0])
        x0_curr = current_bbox[0]

        while j < len(df):
            nxt = df.iloc[j].to_dict()
            next_bbox = nxt.get("bbox", [0, 0, 0, 0])
            x0_next = next_bbox[0]

            # Calculate merge conditions safely
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


