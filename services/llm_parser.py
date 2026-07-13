import re
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 1. Pydantic Schemas for Structured Output
class PortfolioItemSchema(BaseModel):
    title: str = Field(description="Judul proyek, pengalaman praktik, atau karya nyata")
    description: Optional[str] = Field(None, description="Deskripsi singkat mengenai proyek tersebut")

class CVExtractionResponseSchema(BaseModel):
    hard_skills: List[str] = Field(description="Skill teknis spesifik pekerjaan (IT maupun non-IT spt Housekeeping, Tata Boga). JANGAN isi nomor telepon/alamat.")
    soft_skills: List[str] = Field(description="Keterampilan interpersonal dan atribut personal. JANGAN isi nomor telepon/alamat.")
    portfolio: List[PortfolioItemSchema] = Field(description="Daftar proyek atau portofolio nyata")

class SubjectScoreSchema(BaseModel):
    subject: str = Field(description="Nama mata pelajaran")
    score: float = Field(description="Nilai mata pelajaran (skala 0-100)")

class AcademicScoreExtractionSchema(BaseModel):
    subject_scores: List[SubjectScoreSchema] = Field(description="Daftar mata pelajaran beserta nilainya")
    average_score: float = Field(description="Nilai rata-rata keseluruhan (skala 0-100)")

# 2. System Prompt
SYSTEM_PROMPT_CV = """Kamu adalah ahli HR dan Ekstraksi Data CV profesional.
Tugasmu HANYA mengekstrak 3 kategori data dari teks CV pelamar, khususnya lulusan SMK berbagai jurusan (IT, Pariwisata, Tata Boga, Perhotelan, dll).

ATURAN KETAT EKSTRAKSI:
1. 'hard_skills': HANYA masukkan keterampilan teknis. Contoh: Programming, Jaringan, Front Office, Housekeeping, Tata Boga, Akuntansi, Merakit Mesin. JANGAN masukkan lokasi, nama, atau kontak.
2. 'soft_skills': HANYA masukkan keterampilan interpersonal. Contoh: Komunikasi, Kerja Tim, Kepemimpinan. JANGAN masukkan hal-hal teknis atau profil.
3. 'portfolio': Ekstrak pengalaman nyata/proyek yang pernah dibuat/dikerjakan.
4. LARANGAN KERAS: JANGAN PERNAH memasukkan Nomor Telepon, Email, Alamat (seperti 'Denpasar', 'Bali'), atau ringkasan profil/biodata ke dalam kolom skill. Jika ragu, LEWATI.
5. Hasilkan output dalam format JSON yang valid.
"""

SYSTEM_PROMPT_SCORE = """Kamu adalah ahli Ekstraksi Data Akademik profesional.
Tugasmu adalah mengekstrak data nilai dari teks transkrip/rapor siswa SMK Indonesia.

ATURAN KETAT EKSTRAKSI:
1. 'subject_scores': Ekstrak setiap mata pelajaran beserta nilainya. Contoh mata pelajaran SMK: Pendidikan Agama, PKN, Bahasa Indonesia, Matematika, Bahasa Inggris, Informatika, PJOK, Seni Budaya, Sejarah, dan mata pelajaran kejuruan (Dasar Desain Grafis, Pemrograman Web, Komputer dan Jaringan, dll).
2. 'average_score': Hitung rata-rata dari semua nilai mata pelajaran yang ditemukan. Gunakan skala 0-100.
3. Nilai yang valid berada dalam rentang 0-100. Abaikan angka yang merupakan nomor urut, tahun ajaran, NIS/NISN, atau kode lainnya.
4. Jika ada label 'Rata-rata' atau 'Jumlah' atau 'Rerata' di teks, gunakan nilai tersebut sebagai average_score.
5. Hasilkan output dalam format JSON yang valid.
"""

def _extract_section_lines(raw_text: str, header_variants: List[str]) -> List[str]:
    lines = [l.rstrip() for l in raw_text.splitlines()]
    lower_lines = [l.lower() for l in lines]

    # find header index
    header_idx = None
    for i, l in enumerate(lower_lines):
        clean_l = l.strip()
        # A header should be short (usually 1-4 words)
        if len(clean_l.split()) > 5:
            continue
            
        for hv in header_variants:
            # Match if line is exactly the header or starts with it followed by space/colon
            if clean_l == hv or clean_l.startswith(hv + ":") or clean_l.startswith(hv + " "):
                header_idx = i
                break
        if header_idx is not None:
            break

    if header_idx is None:
        return []

    # collect until next known header-like line or two consecutive empty lines
    results = []
    j = header_idx + 1
    known_headers = [
        'profil', 'profile', 'summary', 'pendidikan', 'education', 'portofolio', 'portfolio',
        'hard skills', 'soft skills', 'work', 'projects', 'pengalaman', 'certifications',
        'pengalaman kerja', 'pengalaman organisasi', 'keahlian', 'keterampilan teknis', 'skill', 'skills',
        'sertifikasi', 'bahasa', 'languages', 'contact', 'kontak', 'referensi'
    ]
    blank_count = 0
    while j < len(lines):
        line = lines[j].strip()
        if line == '':
            blank_count += 1
            if blank_count >= 2:
                break
            # single blank line: treat as possible separator but continue
            j += 1
            continue
        blank_count = 0
        low = line.lower()
        if any(k in low for k in known_headers):
            break
        results.append(line)
        j += 1

    return results


def parse_structured_data(raw_text: str, doc_type: str = "cv") -> dict:
    """
    Mock implementation of LLM parser. 
    In production, this would call OpenAI (gpt-4o-mini / gpt-3.5-turbo)
    or a local model (Llama-3 via Ollama) with a JSON schema constraint.
    """
    
    # We return a dict that matches ExtractionResponseData shape
    parsed = {
        "full_name": "",
        "email": "",
        "phone_number": "",
        "address": "",
        "linkedin": "",
        "github": "",
        "portfolio": [],
        "education": [],
        "work_experience": [],
        "organization_experience": [],
        "certifications": [],
        "languages": [],
        "hard_skills": [],
        "soft_skills": [],
        "projects": [],
        "subject_scores": [],
        "average_score": 0.0
    }
    
    # Optional: If OPENAI_API_KEY is available, use real LLM for CV extraction
    if doc_type == "cv" and OpenAI and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CV},
                    {"role": "user", "content": raw_text}
                ],
                response_format=CVExtractionResponseSchema,
            )
            extracted = completion.choices[0].message.parsed
            
            parsed['hard_skills'] = extracted.hard_skills
            parsed['soft_skills'] = extracted.soft_skills
            
            # Map PortfolioItemSchema to dict format expected by the app
            for item in extracted.portfolio:
                parsed['portfolio'].append({
                    'judul': item.title,
                    'poin': [item.description] if item.description else [],
                    'links': []
                })
            
            return parsed
        except Exception as e:
            print(f"OpenAI Extraction failed: {e}. Falling back to regex parser.")
            # Fallback to regex parser below if OpenAI fails

    # Simple regex/line-based parser constrained to specific CV sections only.
    # IMPORTANT: we only extract from explicit sections: 'Portofolio/Portfolio',
    # 'Hard Skills' and 'Soft Skills'. We DO NOT extract names, addresses, phones,
    # or profile/summary text into the skills or portfolio fields.
    if doc_type == "cv":
        # Hard skills: only from explicit hard skills section headers (English + Indonesian variants)
        hs_lines = _extract_section_lines(raw_text, [
            'hard skills', 'technical skills', 'tech stack', 'skill teknis',
            'keahlian', 'keahlian teknis', 'keterampilan teknis', 'keterampilan', 'kompetensi'
        ])
        hard_skills = []
        for l in hs_lines:
            if ',' in l:
                hard_skills.extend([s.strip() for s in l.split(',') if s.strip()])
            else:
                # split by common separators or treat whole line as one skill
                parts = re.split(r'\s{2,}|\|', l)
                if len(parts) > 1:
                    hard_skills.extend([p.strip() for p in parts if p.strip()])
                else:
                    hard_skills.append(l.strip())
        # Filter hard skills: remove any item containing emails/phones/urls or
        # obviously personal/profile text (long sentences). Keep short skill-like tokens.
        def _is_skill_token(s: str) -> bool:
            """Validator for hard skills — strict: no digits, short tokens only."""
            if not s:
                return False
            low = s.lower()
            if '@' in s or re.search(r'https?://', s) or re.search(r'www\.', s):
                return False
            if re.search(r'\d', s):
                return False
            # drop very long lines (likely profile sentences)
            if len(s.split()) > 8:
                return False
            # drop common profile keywords
            if any(k in low for k in ('profil', 'alamat', 'nama', 'ttl', 'tanggal lahir', 'tempat', 'telepon', 'email')):
                return False
            return True

        def _is_soft_skill_token(s: str) -> bool:
            """Validator for soft skills — relaxed: allow multi-word interpersonal phrases."""
            if not s:
                return False
            low = s.lower()
            # Reject contact-info patterns
            if '@' in s or re.search(r'https?://', s) or re.search(r'www\.', s):
                return False
            # Reject phone numbers (5+ consecutive digits)
            if re.search(r'\d{5,}', s):
                return False
            # Drop very long lines (likely full sentences/profile text, not a skill label)
            if len(s.split()) > 10:
                return False
            # drop common profile/bio keywords
            if any(k in low for k in ('profil', 'alamat', 'nama', 'ttl', 'tanggal lahir', 'tempat', 'telepon', 'email')):
                return False
            return True

        parsed['hard_skills'] = [s for s in (x for x in hard_skills) if _is_skill_token(s)]

        # Soft skills: only from explicit soft skills section (including Indonesian variants)
        ss_lines = _extract_section_lines(raw_text, [
            'soft skills', 'interpersonal skills', 'skill non teknis', 'softskill',
            'keterampilan interpersonal', 'keterampilan non teknis', 'keterampilan lunak'
        ])
        valid_ss_lines = []
        for l in ss_lines:
            if re.search(r'[@]', l) or re.search(r'https?://', l) or 'www.' in l: continue
            if re.search(r'\d{5,}', l): continue
            valid_ss_lines.append(l.strip())
            
        soft_skills = []
        if valid_ss_lines:
            has_comma = any(',' in l for l in valid_ss_lines)
            if has_comma:
                curr = ""
                for l in valid_ss_lines:
                    if re.match(r'^[\-•\*]\s+', l):
                        if curr:
                            parts = [s.strip().rstrip('.;') for s in curr.split(',') if s.strip()]
                            soft_skills.extend(parts)
                        curr = l
                    else:
                        if curr: curr += " " + l
                        else: curr = l
                if curr:
                    parts = [s.strip().rstrip('.;') for s in curr.split(',') if s.strip()]
                    soft_skills.extend(parts)
            else:
                for l in valid_ss_lines:
                    clean = l.strip().rstrip('.;')
                    if clean: soft_skills.append(clean)
        # Filter soft skills with a dedicated, less aggressive validator
        parsed['soft_skills'] = [s for s in soft_skills if _is_soft_skill_token(s)]

        # Portofolio / Portfolio (group contiguous non-empty lines into items)
        # Portfolio: only from explicit portfolio/portofolio/projects section (include 'project' variants)
        p_lines = _extract_section_lines(raw_text, [
            'portofolio', 'portfolio', 'projects', 'project', 'projek', 'proyek'
        ])
        portfolio = []
        if p_lines:
            # group contiguous lines into entries separated by blank lines (already bounded)
            # assume each non-empty line is either a title or description; create simple structure
            i = 0
            while i < len(p_lines):
                line = p_lines[i].strip()
                # detect URLs in line
                urls = re.findall(r'https?://\S+|www\.\S+', line)
                # Determine if line is a title or a bullet
                # A title usually starts with 'Proyek'/'Project' or contains a year like (2024)
                is_bullet = bool(re.match(r'^[\-•\*\xb7]', line))
                looks_like_title = bool(re.match(r'^(proyek|project|projek|pembangunan|pengembangan|aplikasi|sistem|website|ui/ux)\b', line, re.IGNORECASE)) or bool(re.search(r'\(20\d{2}\)', line))
                
                # Title must be relatively short and not just a common word
                if (not is_bullet and looks_like_title and len(line.split()) <= 12) or (not is_bullet and i == 0 and len(line.split()) <= 8):
                    title = line
                    j = i + 1
                    bullets = []
                    while j < len(p_lines):
                        next_line = p_lines[j].strip()
                        if not next_line:
                            j += 1
                            continue
                        # If next_line looks like a NEW title, stop
                        is_next_bullet = bool(re.match(r'^[\-•\*\xb7]', next_line))
                        next_looks_title = bool(re.match(r'^(proyek|project|projek|pembangunan|pengembangan|aplikasi|sistem|website|ui/ux)\b', next_line, re.IGNORECASE)) or bool(re.search(r'\(20\d{2}\)', next_line))
                        
                        if not is_next_bullet and next_looks_title:
                            break
                        bullets.append(next_line)
                        j += 1
                    portfolio.append({'judul': title, 'poin': bullets, 'links': urls})
                    i = j
                else:
                    # long line or bullet without a preceding title
                    portfolio.append({'judul': line[:60] + "...", 'poin': [line], 'links': urls})
                    i += 1

        parsed['portfolio'] = portfolio

        # Do not fallback-scan the whole document for hard skills. Only use explicit section.

        # If portfolio still empty, search for lines that explicitly mention 'Proyek'/'Project' etc
        if not parsed['portfolio']:
            lines = [l.rstrip() for l in raw_text.splitlines()]
            portfolio_fallback = []
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                low = line.lower()
                # Fallback titles must strictly start with keyword and be short
                if re.match(r'^(proyek|project|projek)\b', low) and len(line.split()) <= 10:
                    title = line
                    j = i + 1
                    bullets = []
                    # collect following lines until blank or next header-like word
                    while j < len(lines):
                        nl = lines[j].strip()
                        if nl == '':
                            break
                        if any(k in nl.lower() for k in ['hard skills', 'soft skills', 'portofolio', 'portfolio', 'pendidikan', 'profil', 'pengalaman']):
                            break
                        # skip contact lines
                        if re.search(r'[@\d]|https?://|www\.', nl):
                            j += 1
                            continue
                        bullets.append(nl)
                        j += 1
                    portfolio_fallback.append({'judul': title, 'poin': bullets, 'links': re.findall(r'https?://\S+|www\.\S+', title)})
                    i = j
                    continue
                i += 1

            if portfolio_fallback:
                parsed['portfolio'] = portfolio_fallback

        # Do NOT populate personal/contact fields (full_name, email, phone_number, address)
        parsed['full_name'] = ''
        parsed['email'] = ''
        parsed['phone_number'] = ''
        parsed['address'] = ''

    elif doc_type == "score":
        result = _parse_academic_scores(raw_text)
        parsed['subject_scores'] = result['subject_scores']
        parsed['average_score'] = result['average_score']
    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# Academic Score Parsing — 3-Layer Strategy
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_ocr_text(raw_text: str) -> str:
    """
    Preprocess OCR text to fix common artifacts before parsing.

    Key fixes:
    1. Rejoin split decimal numbers: "84, 24" → "84,24", "81. 93" → "81.93"
       OCR engines often insert a space after comma/period in numbers.
    2. Rejoin decimals split across adjacent lines:
       Line N:   "Rata-rata"
       Line N+1: "81"
       Line N+2: "93"
       → Merge "81" + "93" into "81,93" when they look like a split decimal.
    3. Normalize various dash/space combos in "RATA-RATA" labels.
    """
    if not raw_text:
        return raw_text

    # Fix 1: Rejoin "NN, DD" or "NN. DD" where NN is 2-3 digits and DD is 1-2 digits
    # This catches OCR reading "84,24" as "84, 24" (space after comma/period)
    # Pattern: 2-3 digit number, comma/period, optional space(s), 1-2 digit number
    text = re.sub(
        r'(\d{2,3})([.,])\s+(\d{1,2})(?!\d)',
        r'\1\2\3',
        raw_text
    )

    # Fix 2: Handle lines where a decimal is split across lines
    # e.g., Line "81" followed by line "93" should become "81,93"
    # But ONLY do this adjacent to a "rata-rata" label line, to avoid false merges
    lines = text.splitlines()
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if the NEXT line is a lone 1-2 digit number that could be a decimal part
        if i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            # Current line ends with 2-3 digit number, next line is 1-2 digit standalone number
            curr_match = re.search(r'(\d{2,3})\s*$', stripped)
            next_match = re.match(r'^(\d{1,2})$', next_stripped)
            if curr_match and next_match:
                curr_num = int(curr_match.group(1))
                next_num = int(next_match.group(1))
                # Plausible if curr_num is 30-100 and next_num is 0-99 (decimal part)
                combined = float(f"{curr_num}.{next_num:02d}" if next_num < 10 else f"{curr_num}.{next_num}")
                if 30 <= curr_num <= 100 and next_num <= 99 and 30 <= combined <= 100:
                    # Merge: replace the trailing number on current line with combined
                    merged_line = re.sub(r'(\d{2,3})\s*$', f'{curr_num},{next_match.group(1)}', stripped)
                    merged_lines.append(merged_line)
                    i += 2  # Skip the next line (it's been merged)
                    continue

        merged_lines.append(line)
        i += 1

    return '\n'.join(merged_lines)


def _parse_academic_scores(raw_text: str) -> dict:
    """
    3-layer strategy to extract academic scores from OCR text of SMK report cards.

    Layer 1: Regex pattern matching for explicit 'Rata-rata' labels.
    Layer 2: Statistical extraction — collect all plausible scores, filter outliers, average.
    Layer 3: OpenAI GPT-4o-mini fallback for non-standard formats.
    """
    result = {'subject_scores': [], 'average_score': 0.0}

    if not raw_text or not raw_text.strip():
        return result

    # ── Step 0: Normalize OCR artifacts (rejoin split decimals) ────────────
    raw_text = _normalize_ocr_text(raw_text)

    # ── Layer 1: Regex — look for explicit average labels ──────────────────────
    layer1 = _layer1_regex_average(raw_text)
    layer1_subjects = _layer1_extract_subjects(raw_text)

    if layer1_subjects:
        result['subject_scores'] = layer1_subjects

    if layer1 > 0:
        result['average_score'] = layer1
        if not result['subject_scores']:
            # Try to also get subjects even if avg was found via label
            result['subject_scores'] = _layer2_extract_subject_scores(raw_text)

        # ── Cross-validation: compare label average vs calculated average ──────
        # OCR decimal digits are highly error-prone (e.g., "27" misread as "76").
        # When we have >= 5 subject scores, the calculated average is MORE reliable
        # than the OCR-read label value. Use calculated avg if they differ by > 0.3.
        # (Normal rounding differences will be << 0.3; OCR misread diffs are usually > 0.4)
        if result['subject_scores'] and len(result['subject_scores']) >= 5:
            scores = [s['score'] for s in result['subject_scores']]
            calc_avg = round(sum(scores) / len(scores), 2)
            discrepancy = abs(result['average_score'] - calc_avg)
            print(f"[DEBUG] Cross-validation: label_avg={result['average_score']}, calc_avg={calc_avg}, discrepancy={discrepancy:.4f}")
            if discrepancy > 0.3:
                print(f"[DEBUG] Cross-validation OVERRIDE: decimal likely misread by OCR. Using calc_avg={calc_avg} over label_avg={result['average_score']}")
                result['average_score'] = calc_avg
            else:
                print(f"[DEBUG] Cross-validation OK: difference {discrepancy:.4f} within tolerance, keeping label_avg={result['average_score']}")

        return result

    # ── Layer 2: Statistical extraction ────────────────────────────────────────
    layer2_subjects = _layer2_extract_subject_scores(raw_text)
    if layer2_subjects:
        result['subject_scores'] = layer2_subjects
        scores = [s['score'] for s in layer2_subjects]
        result['average_score'] = round(sum(scores) / len(scores), 2)
        return result

    # If we got subjects from layer1 but no avg, calculate from them
    if layer1_subjects:
        scores = [s['score'] for s in layer1_subjects]
        if scores:
            result['average_score'] = round(sum(scores) / len(scores), 2)
            return result

    # ── Layer 3: OpenAI fallback ───────────────────────────────────────────────
    if OpenAI and os.getenv("OPENAI_API_KEY"):
        layer3 = _layer3_openai_extraction(raw_text)
        if layer3['average_score'] > 0:
            return layer3

    # ── Fallback: brute-force number collection ────────────────────────────────
    fallback_avg = _fallback_brute_average(raw_text)
    if fallback_avg > 0:
        result['average_score'] = fallback_avg

    return result


def _layer1_regex_average(raw_text: str) -> float:
    """
    Layer 1: Look for explicit 'Rata-rata', 'Rerata', 'Average', 'Jumlah / N' labels.
    Returns the average score or 0.0 if not found.

    Handles common OCR artifacts:
    - "RATA- RATA" (space after dash)
    - "RATA -RATA" (space before dash)
    - "RATA - RATA" (spaces around dash)
    - "RATA RATA" (no dash at all)
    - Score on same line or next line
    """
    text = raw_text.lower()

    # ── Strategy A: single-pass regex on full text ──────────────────────────
    avg_patterns = [
        # "Rata-rata" / "Rata rata" / "Rata - rata" variants → number on SAME LINE
        # Allow any combo of spaces/dashes between "rata" and "rata"
        r'(?:nilai\s+)?rata[\s\-]+rata[\s:=.,]*([\d]{2,3}[.,]\d{1,2})',
        # Handle OCR artifact: space inside decimal → "81, 93" or "81. 93"
        r'(?:nilai\s+)?rata[\s\-]+rata[\s:=.,]*(\d{2,3})[.,]\s*(\d{1,2})(?!\d)',
        r'(?:nilai\s+)?rata[\s\-]+rata[\s:=.,]*([\d]{2,3})\b',
        r'rerata[\s:=.,]*([\d]{2,3}[.,]\d{1,2})',
        r'rerata[\s:=.,]*([\d]{2,3})\b',
        r'average[\s:=.,]*([\d]{2,3}[.,]\d{1,2})',
        r'mean[\s:=.,]*([\d]{2,3}[.,]\d{1,2})',
        # Number BEFORE the label
        r'([\d]{2,3}[.,]\d{1,2})\s*rata[\s\-]+rata',
        # "Jumlah" followed by number → might be sum, handle carefully
        r'jumlah[\s:=]+([\d]+[.,]?\d*)',
    ]

    for pattern in avg_patterns:
        match = re.search(pattern, text)
        if match:
            # Handle the special 2-group pattern for split decimals
            if match.lastindex == 2:
                val_str = f"{match.group(1)}.{match.group(2)}"
            else:
                val_str = match.group(1).replace(',', '.')
            try:
                val = float(val_str)
                if 0 < val <= 100:
                    return round(val, 2)
                if 'jumlah' in pattern and val > 100:
                    continue
            except ValueError:
                continue

    # ── Strategy B: line-by-line scan ───────────────────────────────────────
    # OCR often puts "RATA-RATA" on one line and the score on the next line.
    # Key fix: when an integer is found on the rata-rata line, we MUST check the
    # next line for a 1-2 digit decimal part BEFORE returning the integer alone.
    # This prevents incorrect combinations like 89+76=89.76 when actual is 89.27
    lines = raw_text.splitlines()
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        # Check if this line contains a "rata-rata" label (with various OCR artifacts)
        if re.search(r'rata[\s\-]*rata', line_lower):
            print(f"[DEBUG] Found rata-rata at line {i}: {repr(line.strip())}")

            # First, try to extract a proper decimal (with comma/dot) from THIS line
            nums_on_line = re.findall(r'(\d{2,3}[.,]\d{1,2})', line)
            if nums_on_line:
                for ns in nums_on_line:
                    try:
                        v = float(ns.replace(',', '.'))
                        if 30 <= v <= 100:
                            print(f"[DEBUG] Strategy B - Decimal on same line: {v}")
                            return round(v, 2)
                    except ValueError:
                        continue

            # Try split decimal on same line: "81, 93" or "81. 93"
            split_match = re.search(r'(\d{2,3})[.,]\s+(\d{1,2})(?!\d)', line)
            if split_match:
                try:
                    v = float(f"{split_match.group(1)}.{split_match.group(2)}")
                    if 30 <= v <= 100:
                        print(f"[DEBUG] Strategy B - Split decimal same line: {v}")
                        return round(v, 2)
                except ValueError:
                    pass

            # Check standalone integer on THIS line — but FIRST peek at the next
            # non-empty line to see if it's a 1-2 digit decimal part (e.g. "27")
            # CRITICAL FIX: do NOT return integer alone before checking next line
            int_nums_on_line = re.findall(r'\b(\d{2,3})\b', line)
            candidate_int = None
            for ns in int_nums_on_line:
                try:
                    v = int(ns)
                    if 30 <= v <= 100:
                        candidate_int = v
                        break
                except ValueError:
                    continue

            # Look at next 1-3 lines for the decimal part or a full decimal value
            found_in_next = False
            for j in range(i + 1, min(i + 4, len(lines))):
                next_line = lines[j].strip()
                if not next_line:
                    continue

                print(f"[DEBUG] Strategy B - Checking next line {j}: {repr(next_line)}")

                # Priority 1: full decimal value on next line
                nums_next = re.findall(r'(\d{2,3}[.,]\d{1,2})', next_line)
                if nums_next:
                    for ns in nums_next:
                        try:
                            v = float(ns.replace(',', '.'))
                            if 30 <= v <= 100:
                                print(f"[DEBUG] Strategy B - Full decimal on next line: {v}")
                                return round(v, 2)
                        except ValueError:
                            continue

                # Priority 2: split decimal on next line "89, 27"
                split_next = re.search(r'(\d{2,3})[.,]\s+(\d{1,2})(?!\d)', next_line)
                if split_next:
                    try:
                        v = float(f"{split_next.group(1)}.{split_next.group(2)}")
                        if 30 <= v <= 100:
                            print(f"[DEBUG] Strategy B - Split decimal on next line: {v}")
                            return round(v, 2)
                    except ValueError:
                        pass

                # Priority 3: combine candidate_int from THIS line with lone 1-2 digit on NEXT line
                # e.g., this line has "89", next line has "27" → 89.27
                if candidate_int is not None:
                    lone_decimal = re.match(r'^(\d{1,2})$', next_line)
                    if lone_decimal:
                        try:
                            dec_part = lone_decimal.group(1)
                            v = float(f"{candidate_int}.{dec_part}")
                            if 30 <= v <= 100:
                                print(f"[DEBUG] Strategy B - Combined int+decimal (cross-line): {candidate_int}+{dec_part} = {v}")
                                return round(v, 2)
                        except ValueError:
                            pass

                # Priority 4: next line is standalone int that might combine with ITS next line
                standalone_int = re.match(r'^(\d{2,3})$', next_line)
                if standalone_int and j + 1 < len(lines):
                    follow_line = lines[j + 1].strip()
                    follow_match = re.match(r'^(\d{1,2})$', follow_line)
                    if follow_match:
                        try:
                            v = float(f"{standalone_int.group(1)}.{follow_match.group(1)}")
                            if 30 <= v <= 100:
                                print(f"[DEBUG] Strategy B - Combined across 2 next lines: {v}")
                                return round(v, 2)
                        except ValueError:
                            pass

                # Priority 5: standalone integer on next line (no decimal found anywhere)
                # Only use this as LAST resort — grab standalone int from next line
                int_nums_next = re.findall(r'\b(\d{2,3})\b', next_line)
                for ns in int_nums_next:
                    try:
                        v = float(ns)
                        if 30 <= v <= 100:
                            # But before returning, peek at line after this to see if decimal part
                            if j + 1 < len(lines):
                                after_line = lines[j + 1].strip()
                                lone_dec = re.match(r'^(\d{1,2})$', after_line)
                                if lone_dec:
                                    combined_v = float(f"{int(v)}.{lone_dec.group(1)}")
                                    if 30 <= combined_v <= 100:
                                        print(f"[DEBUG] Strategy B - Int+peek next: {combined_v}")
                                        return round(combined_v, 2)
                            print(f"[DEBUG] Strategy B - Standalone int on next line: {v}")
                            found_in_next = True
                            return round(v, 2)
                    except ValueError:
                        continue

                # Stop after first non-empty next line processed
                break

            # If we have a candidate_int from this line and nothing better found in next lines
            if candidate_int is not None and not found_in_next:
                print(f"[DEBUG] Strategy B - Fallback to integer on rata-rata line: {candidate_int}")
                return float(candidate_int)

    return 0.0


def _layer1_extract_subjects(raw_text: str) -> List[dict]:
    """
    Layer 1: Extract subject-score pairs from table-like structures in OCR text.
    Looks for patterns like:
      - "Matematika    85"
      - "1. Bahasa Indonesia  80"
      - "Bahasa Inggris : 75"
    """
    subjects = []
    lines = raw_text.splitlines()

    # Common SMK subjects for validation
    subject_keywords = [
        'agama', 'pkn', 'pancasila', 'kewarganegaraan', 'bahasa indonesia',
        'matematika', 'bahasa inggris', 'english', 'informatika', 'komputer',
        'pjok', 'penjas', 'olahraga', 'seni', 'budaya', 'sejarah',
        'fisika', 'kimia', 'biologi', 'ekonomi', 'geografi', 'sosiologi',
        # Kejuruan SMK
        'pemrograman', 'jaringan', 'desain', 'grafis', 'web', 'multimedia',
        'akuntansi', 'keuangan', 'administrasi', 'perkantoran', 'pemasaran',
        'pariwisata', 'perhotelan', 'tata boga', 'busana', 'kecantikan',
        'teknik', 'mesin', 'otomotif', 'elektro', 'listrik', 'elektronika',
        'konstruksi', 'bangunan', 'pertanian', 'agribisnis',
        'basis data', 'database', 'sistem operasi', 'animasi',
        'produk kreatif', 'kewirausahaan', 'simulasi', 'komunikasi', 'digital',
        'dasar program', 'dasar desain', 'teknologi', 'mapel',
        'prakarya', 'bimbingan', 'konseling', 'muatan lokal',
    ]

    # Pattern: subject name (with possible numbering) followed by a score
    # e.g., "1. Matematika  85" or "Bahasa Indonesia : 78.5" or "Pendidikan Agama Islam    82"
    line_pattern = re.compile(
        r'^\s*'
        r'(?:\d{1,2}[.)]\s*)?'            # Optional numbering: "1." or "2)"
        r'([A-Za-z][A-Za-z\s./&\-]{2,50}?)' # Subject name (at least 3 chars)
        r'\s*[:\s]\s*'                      # Separator (colon, spaces, tabs)
        r'(\d{2,3}(?:[.,]\d{1,2})?)\s*$',    # Score (2-3 digits, optional decimal)
        re.IGNORECASE
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = line_pattern.match(line)
        if m:
            subj_name = m.group(1).strip()
            score_str = m.group(2).replace(',', '.')

            # Skip lines that look like headers, dates, or IDs
            subj_lower = subj_name.lower()
            skip_words = ['nomor', 'nis', 'nisn', 'kelas', 'semester', 'tahun',
                          'tanggal', 'nama', 'wali', 'kepala', 'guru', 'alamat',
                          'tempat', 'lahir', 'jumlah', 'rata', 'halaman', 'page',
                          'no', 'keterangan', 'kkm', 'predikat']
            if any(sw in subj_lower for sw in skip_words):
                continue

            try:
                score = float(score_str)
                if 0 < score <= 100:
                    subjects.append({
                        'subject': subj_name,
                        'score': score
                    })
            except ValueError:
                continue

    return subjects


def _layer2_extract_subject_scores(raw_text: str) -> List[dict]:
    """
    Layer 2: Statistical extraction — find clusters of numbers in 0-100 range
    that are likely academic scores. Filters out years, phone numbers, IDs, etc.
    """
    lines = raw_text.splitlines()
    candidate_scores = []

    # Words that indicate a line contains a score (Indonesian rapor context)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Skip obvious non-score lines
        lower = stripped.lower()
        skip_indicators = ['nis', 'nisn', 'telepon', 'telp', 'hp', 'tahun pelajaran',
                           'tahun ajaran', 'semester', 'kelas', 'alamat', 'tempat lahir',
                           'tanggal lahir', 'nama siswa', 'wali kelas', 'kepala sekolah',
                           'nip', 'halaman', 'page', 'nomor induk']
        if any(si in lower for si in skip_indicators):
            continue

        # Extract all numbers from the line
        numbers = re.findall(r'\b(\d{2,3}(?:[.,]\d{1,2})?)\b', stripped)

        for num_str in numbers:
            try:
                val = float(num_str.replace(',', '.'))
            except ValueError:
                continue

            # Filter criteria:
            # 1. Must be in plausible score range (typically 40-100 for SMK, allow 0-100)
            if val < 10 or val > 100:
                continue

            # 2. Skip year-like numbers (2020-2030)
            if 2000 <= val <= 2099:
                continue
            # Integer check: also skip if it matches YYYY pattern in the line context
            if re.search(r'20[12]\d', num_str):
                continue

            # 3. Skip small integers that are likely sequence numbers (1-15)
            if val == int(val) and val <= 15:
                # Check if this looks like a sequence number (preceded by "No" or is alone)
                if re.search(r'(?:^|no|nomor)[.\s]*' + re.escape(num_str) + r'\b', lower):
                    continue
                # If the number is isolated (not part of a subject-score pair), skip small numbers
                if val <= 15 and len(numbers) <= 1:
                    continue

            # 4. This looks like a plausible score
            # Try to extract a subject name from the same line
            subj = re.sub(r'\d+[.,]?\d*', '', stripped).strip()
            subj = re.sub(r'^[\d.):]+\s*', '', subj).strip()  # Remove leading numbering
            subj = re.sub(r'[:\-=]+$', '', subj).strip()       # Remove trailing separators

            if len(subj) >= 3 and not subj.isnumeric():
                candidate_scores.append({
                    'subject': subj[:60],  # Cap subject name length
                    'score': val
                })

    # Deduplicate: if same subject appears multiple times, keep the last occurrence
    seen = {}
    for item in candidate_scores:
        key = item['subject'].lower().strip()
        seen[key] = item

    # Only return if we have a reasonable number of scores (at least 3 subjects)
    unique_scores = list(seen.values())
    if len(unique_scores) >= 3:
        return unique_scores

    return []


def _layer3_openai_extraction(raw_text: str) -> dict:
    """
    Layer 3: Use OpenAI GPT-4o-mini with structured output to extract scores
    from non-standard transcript formats.
    """
    result = {'subject_scores': [], 'average_score': 0.0}

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SCORE},
                {"role": "user", "content": f"Berikut adalah teks hasil OCR dari rapor/transkrip siswa SMK. Ekstrak semua nilai mata pelajaran dan hitung rata-ratanya:\n\n{raw_text}"}
            ],
            response_format=AcademicScoreExtractionSchema,
        )
        extracted = completion.choices[0].message.parsed

        result['subject_scores'] = [
            {'subject': s.subject, 'score': s.score}
            for s in extracted.subject_scores
            if 0 < s.score <= 100
        ]
        result['average_score'] = round(extracted.average_score, 2)

        # Validate: if OpenAI average seems off, recalculate from subjects
        if result['subject_scores'] and (result['average_score'] <= 0 or result['average_score'] > 100):
            scores = [s['score'] for s in result['subject_scores']]
            result['average_score'] = round(sum(scores) / len(scores), 2)

        return result

    except Exception as e:
        print(f"OpenAI academic score extraction failed: {e}")
        return result


def _fallback_brute_average(raw_text: str) -> float:
    """
    Last resort: collect ALL numbers in the 40-100 range from the text,
    filter aggressively, and compute their average.
    This is used only when all other layers fail.
    """
    # Find all numbers
    all_numbers = re.findall(r'\b(\d{2,3}(?:[.,]\d{1,2})?)\b', raw_text)
    plausible_scores = []

    for num_str in all_numbers:
        try:
            val = float(num_str.replace(',', '.'))
        except ValueError:
            continue

        # Only accept values that look like scores (40-100 range, typical for SMK)
        if 40 <= val <= 100:
            # Skip years
            if 2000 <= val <= 2099:
                continue
            plausible_scores.append(val)

    # Need at least 3 plausible scores to compute a meaningful average
    if len(plausible_scores) >= 3:
        return round(sum(plausible_scores) / len(plausible_scores), 2)

    return 0.0
