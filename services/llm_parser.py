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
        'pengalaman kerja', 'pengalaman organisasi', 'keahlian', 'keterampilan', 'skill', 'skills',
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

        parsed['hard_skills'] = [s for s in (x for x in hard_skills) if _is_skill_token(s)]

        # Soft skills
        # Soft skills: only from explicit soft skills section (including Indonesian variants)
        ss_lines = _extract_section_lines(raw_text, [
            'soft skills', 'interpersonal skills', 'skill non teknis', 'softskill',
            'keterampilan interpersonal', 'keterampilan non teknis', 'keterampilan lunak'
        ])
        soft_skills = []
        for l in ss_lines:
            # If the line appears to contain contact info (emails/phones), skip the whole line
            if re.search(r'[@]', l) or re.search(r'\d{2,}', l) or re.search(r'https?://', l) or 'www.' in l:
                continue
            if ',' in l:
                # If any comma-separated token looks like contact info, skip the entire line
                parts = [s.strip() for s in l.split(',') if s.strip()]
                if any(re.search(r'[@\d]|https?://|www\.', p) for p in parts):
                    continue
                soft_skills.extend(parts)
            else:
                soft_skills.append(l.strip())
        # Filter soft skills similarly to avoid contact/profile noise
        parsed['soft_skills'] = [s for s in soft_skills if _is_skill_token(s)]

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
        # Score parsing ...
        parsed['average_score'] = 3.5 # dummy
    return parsed
