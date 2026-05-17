import fitz # PyMuPDF
import io
import easyocr
import re
from PIL import Image
import numpy as np

# Initialize EasyOCR reader (can take a little time to load to memory initially)
reader = easyocr.Reader(['id', 'en'], gpu=False)

def process_document(file_bytes: bytes, filename: str, doc_type: str = "cv") -> tuple[str, float]:
    ext = filename.lower().split('.')[-1]
    
    text = ""
    confidence = 0.8 # Default heuristic 
    
    if ext == "pdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_blocks = []
        for page in doc:
            try:
                blocks = page.get_text("blocks")
            except Exception:
                blocks = []
            
            page_text = ""
            if blocks:
                # Split page into left and right columns
                left_col = []
                right_col = []
                for b in blocks:
                    if len(b) >= 5:
                        x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                        page_mid = page.rect.width / 2.0
                        if x0 < page_mid:
                            left_col.append({'y0': y0, 'text': btext})
                        else:
                            right_col.append({'y0': y0, 'text': btext})
                
                left_col.sort(key=lambda x: x['y0'])
                right_col.sort(key=lambda x: x['y0'])
                
                for blk in left_col:
                    page_text += blk['text'] + "\n"
                for blk in right_col:
                    page_text += blk['text'] + "\n"
                
                for b in blocks:
                    if len(b) >= 5:
                        x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                        all_blocks.append({'page': page.number, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'text': btext, 'width': page.rect.width})
            else:
                page_text = page.get_text("text")

            # If standard text extraction yields very little, fallback to OCR on the page
            if len(page_text.strip()) < 50:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_result = reader.readtext(np.array(img), detail=0)
                text += "\n".join(ocr_result)
                confidence = 0.6
            else:
                text += page_text + "\n"
                confidence = 0.95

        # Use block-level grouping to attempt to extract hard skills strictly from the 'Hard Skills' section
        def extract_hard_skills_from_blocks(blocks_list):
            # Group blocks by approximate column (left/right) using page width midline
            cols = {}
            for b in blocks_list:
                mid = (b['x0'] + b['x1']) / 2.0
                page_mid = b['width'] / 2.0
                col = 0 if mid < page_mid else 1
                cols.setdefault(col, []).append(b)

            header_variants = ['hard skills', 'technical skills', 'keahlian', 'keterampilan', 'skill teknis', 'kompetensi']
            collected = []
            for col, blks in cols.items():
                # sort blocks top-to-bottom
                blks_sorted = sorted(blks, key=lambda x: x['y0'])
                # build line list (approx) preserving order
                lines = []
                for b in blks_sorted:
                    for ln in b['text'].splitlines():
                        lines.append({'text': ln.strip(), 'y': b['y0']})

                # search for header line index
                header_idx = None
                for i, ln in enumerate(lines):
                    low = ln['text'].lower()
                    for hv in header_variants:
                        if hv in low:
                            header_idx = i
                            break
                    if header_idx is not None:
                        break

                if header_idx is None:
                    continue

                # collect subsequent short lines (likely skills) until next header-like token
                stop_collection = False
                for j in range(header_idx + 1, len(lines)):
                    line = lines[j]['text']
                    if not line:
                        continue
                    low = line.lower()
                    if any(k in low for k in ['profil', 'pendidikan', 'portfolio', 'portofolio', 'soft skills', 'projects', 'pengalaman', 'soft skill']):
                        stop_collection = True
                        break
                    # skip contact-like lines
                    if '@' in line or re.search(r'\d', line) or 'http' in line:
                        continue
                    # accept short lines up to 8 words
                    if len(line.split()) <= 8 or re.match(r'^[\-•\*]\s+', line):
                        collected.append(line.strip())
                    else:
                        # long sentence probably not a skill -> stop
                        stop_collection = True
                        break
                if stop_collection:
                    continue

            # dedupe preserving order
            seen = set()
            out = []
            for c in collected:
                lc = c.lower()
                if lc in seen:
                    continue
                seen.add(lc)
                out.append(c)
            return out

        # If we have block data, extract hard skills from blocks and attach to text-level result via metadata
        if all_blocks:
            block_hard_skills = extract_hard_skills_from_blocks(all_blocks)
        else:
            block_hard_skills = []
                
    elif ext in ["jpg", "jpeg", "png"]:
        img = Image.open(io.BytesIO(file_bytes))
        ocr_result = reader.readtext(np.array(img), detail=0)
        text = "\n".join(ocr_result)
        confidence = 0.7
        
    else:
        text = "Unsupported or plain text"
        confidence = 0.5
        
    return text.strip(), confidence


def extract_skills_from_pdf_bytes(file_bytes: bytes, skill_type: str = "hard") -> list:
    """
    Parse PDF bytes and extract skills using block-level layout.
    skill_type: 'hard' or 'soft'
    Returns a list of skill strings or empty list.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception:
        return []

    all_blocks = []
    for page in doc:
        try:
            blocks = page.get_text("blocks")
        except Exception:
            blocks = []
        for b in blocks:
            if len(b) >= 5:
                x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
                all_blocks.append({'page': page.number, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'text': btext, 'width': page.rect.width})

    if not all_blocks:
        return []

    if skill_type == "hard":
        header_variants = ['hard skills', 'technical skills', 'keahlian', 'keterampilan', 'skill teknis', 'kompetensi']
        stop_variants = ['profil', 'pendidikan', 'portfolio', 'portofolio', 'soft skills', 'projects', 'pengalaman', 'soft skill']
    else:
        header_variants = ['soft skills', 'interpersonal skills', 'skill non teknis', 'softskill', 'keterampilan interpersonal', 'keterampilan non teknis']
        stop_variants = ['profil', 'pendidikan', 'portfolio', 'portofolio', 'hard skills', 'projects', 'pengalaman']

    found = []
    for b in all_blocks:
        low = b['text'].lower()
        if any(hv in low for hv in header_variants):
            found.append(b)

    collected = []
    if not found:
        # no explicit header blocks found
        return []

    for header in found:
        header_bottom = header['y1']
        header_left = header['x0']
        
        # collect candidate blocks in same page whose y0 is below header and x0 is at or right of header_left
        # allow some margin (-50) for bullets hanging left
        candidates = [blk for blk in all_blocks if blk['page'] == header['page'] and blk['y0'] >= header_bottom - 10 and blk['y0'] - header_bottom < 600 and blk['x0'] >= header_left - 50]
        
        # sort top-to-bottom, then left-to-right (using bands of 20px)
        candidates = sorted(candidates, key=lambda x: (round(x['y0'] / 20.0) * 20, x['x0']))
        
        for blk in candidates:
            stop_collection = False
            for ln in blk['text'].splitlines():
                line = ln.strip()
                if not line:
                    continue
                low = line.lower()
                # stop if next header-like appears
                if any(k in low for k in stop_variants):
                    stop_collection = True
                    break
                # skip contact-like or sentence-like lines
                if '@' in line or re.search(r'\d', line) or 'http' in line:
                    continue
                # accept short lines (<=8 words) or bullet markers
                if len(line.split()) <= 8 or re.match(r'^[\-•\*]\s+', line):
                    collected.append(line)
                else:
                    # ignore long lines
                    continue
            if stop_collection:
                break

    # dedupe preserving order
    seen = set()
    out = []
    for c in collected:
        lc = c.lower()
        if lc in seen:
            continue
        seen.add(lc)
        # strip any leading bullet chars
        out.append(re.sub(r'^[\-•\*]\s*', '', c))

    return out
