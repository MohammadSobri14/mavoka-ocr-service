from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
from services.ocr_service import process_document, extract_skills_from_pdf_bytes
from services.llm_parser import parse_structured_data

app = FastAPI(title="Mavoka OCR & AI Extraction Service")

class ExtractionResponseData(BaseModel):
    full_name: str = ""
    email: str = ""
    phone_number: str = ""
    address: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: List[Any] = []
    education: List[Any] = []
    work_experience: List[Any] = []
    organization_experience: List[Any] = []
    certifications: List[Any] = []
    languages: List[str] = []
    hard_skills: List[str] = []
    soft_skills: List[str] = []
    projects: List[Any] = []
    subject_scores: List[Any] = []
    average_score: float = 0.0

class ExtractionResponse(BaseModel):
    success: bool
    message: str
    data: ExtractionResponseData
    extracted_text: str = ""
    extraction_confidence: float = 0.0

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Mavoka OCR & AI"}

@app.post("/extract/cv", response_model=ExtractionResponse)
async def extract_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.docx', '.jpg', '.jpeg', '.png')):
         raise HTTPException(status_code=400, detail="Unsupported file format")

    try:
        content = await file.read()
        
        # 1. Extract raw text with PyMuPDF or OCR
        raw_text, confidence = process_document(content, file.filename, doc_type="cv")

        # 2. Parse using AI/NLP (OpenAI / Local Models)
        parsed_data = parse_structured_data(raw_text, doc_type="cv")
        # 3. Attempt to extract skills using block-level PDF layout (strictly from exact sections)
        try:
            block_hard_skills = extract_skills_from_pdf_bytes(content, skill_type="hard")
            if block_hard_skills:
                parsed_data['hard_skills'] = block_hard_skills
                
            block_soft_skills = extract_skills_from_pdf_bytes(content, skill_type="soft")
            if block_soft_skills:
                parsed_data['soft_skills'] = block_soft_skills
        except Exception:
            pass
        
        return ExtractionResponse(
            success=True,
            message="CV extracted successfully",
            data=parsed_data,
            extracted_text=raw_text,
            extraction_confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/academic-score", response_model=ExtractionResponse)
async def extract_score(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        # 1. Image OCR specifically handling tables
        raw_text, confidence = process_document(content, file.filename, doc_type="score")

        # 2. Parse Subject Scores & Average Score using AI
        parsed_data = parse_structured_data(raw_text, doc_type="score")
        
        return ExtractionResponse(
            success=True,
            message="Academic score extracted successfully",
            data=parsed_data,
            extracted_text=raw_text,
            extraction_confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
