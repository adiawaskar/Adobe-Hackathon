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
            # "size": round(sz, 1),
            # "bold": is_bold,
            # "all_upper": is_all_upper
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
