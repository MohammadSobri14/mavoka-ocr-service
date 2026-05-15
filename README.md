# Mavoka OCR & AI Extraction Service

Layanan microservice untuk mengekstrak data dari CV dan Transkrip Nilai (KHS/Transkrip SMK) menggunakan teknologi OCR (Optical Character Recognition) dan AI (LLM).

## 🚀 Fitur Utama

- **CV Extraction**: Mengekstrak hard skills, soft skills, dan portofolio dari file PDF/Gambar.
- **Academic Score Extraction**: Mengekstrak nilai mata pelajaran dan nilai rata-rata dari transkrip.
- **Hybrid Parser**: Menggunakan OpenAI GPT-4o-mini (jika tersedia) atau fallback ke Regex Parser yang cerdas.
- **Multi-OCR Engine**: Mendukung PaddleOCR, EasyOCR, dan PyMuPDF untuk akurasi maksimal.

## 🛠 Panduan Instalasi (Untuk Junior Developer)

### 1. Prasyarat

Pastikan Anda sudah menginstal:

- [Python 3.9 atau lebih baru](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

### 2. Persiapan Lingkungan (Virtual Environment)

Sangat disarankan menggunakan virtual environment agar library tidak bentrok dengan proyek lain.

```powershell
# Masuk ke folder proyek
cd mavoka-ocr-service

# Buat virtual environment
python -m venv .venv

# Aktifkan virtual environment (Windows)
.\venv\Scripts\activate
```

### 3. Instalasi Library

Instal semua dependensi yang dibutuhkan:

```powershell
pip install -r requirements.txt
```

> **Note:** Jika Anda mengalami error saat instalasi `paddleocr` atau `paddlepaddle`, pastikan Anda sudah menginstal [Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe).

### 4. Konfigurasi Environment (Opsional tapi Disarankan)

Untuk hasil ekstraksi AI yang lebih akurat, gunakan OpenAI API Key. Buat file `.env` di root folder:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🏃 Cara Menjalankan

Jalankan server menggunakan Uvicorn:

```powershell
uvicorn main:app --reload
```

Server akan berjalan di: `http://127.0.0.1:8000`

## 📑 Dokumentasi API (Swagger)

Setelah server berjalan, Anda bisa mencoba API secara langsung melalui:
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoint Utama

- `POST /extract/cv`: Kirim file CV (PDF/JPG/PNG).
- `POST /extract/academic-score`: Kirim file transkrip nilai.

## 📁 Struktur Folder

- `main.py`: Entry point aplikasi FastAPI.
- `services/`: Logika utama ekstraksi.
  - `ocr_service.py`: Menangani pembacaan teks dari gambar/PDF.
  - `llm_parser.py`: Menangani pembersihan dan strukturisasi data menggunakan AI.
- `requirements.txt`: Daftar library yang dibutuhkan.

## 💡 Tips untuk Junior

1. **Logs**: Perhatikan terminal saat menjalankan aplikasi untuk melihat jika ada error OCR atau API Key.
2. **File Testing**: Gunakan file `test_parse.py` untuk mencoba logika parsing tanpa menjalankan server.
3. **Branching**: Selalu buat branch baru jika ingin menambah fitur: `git checkout -b fitur-baru`.

---

Dibuat dengan ❤️ untuk Tim Mavoka.
