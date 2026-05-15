from services.llm_parser import parse_structured_data, _extract_section_lines
raw = '''Siti Aulia
Profil Singkat
Pendidikan
Portofolio
Hard Skills
Soft Skills
+123-456-7890, example@gmail.com, Semarang
Siswa SMK jurusan Akuntansi dan Keuangan Lembaga yang memiliki minat dalam pengelolaan
keuangan dan administrasi bisnis. Memiliki pengalaman dalam membuat laporan keuangan
sederhana, melakukan pencatatan transaksi, serta mengoperasikan aplikasi spreadsheet untuk
pengolahan data. Terbiasa bekerja dengan teliti, disiplin, dan memiliki kemampuan analisis dasar
dalam bidang akuntansi.
SMKN 7 Semarang, Akuntansi dan Keuangan Lembaga (2024 - Sekarang)
Proyek Laporan Keuangan Sederhana (2024)
Microsoft Excel
Accurate
Data Entry
Pembukuan
Teliti
Disiplin
Tanggung Jawab
Komunikasi
Proyek Administrasi Kas Organisasi (2025)
Membuat laporan laba rugi dan neraca sederhana.
Melakukan pencatatan transaksi harian menggunakan Microsoft Excel.
Menyesuaikan format laporan sesuai standar dasar akuntansi.
Mengelola pemasukan dan pengeluaran kas organisasi sekolah.
Membuat rekap transaksi bulanan secara terstruktur.
Melakukan pengecekan data untuk meminimalisir kesalahan input.
Problem Solving
Manajemen Waktu
Akuntansi Dasar
Administrasi Keuangan
'''
print('--- extract hard skills lines ---')
hs_lines = _extract_section_lines(raw, ['hard skills','technical skills','keahlian','keterampilan'])
print(hs_lines)
print('--- parsed ---')
print(parse_structured_data(raw))

# Debug: show raw lines
print('\n--- raw lines debug ---')
for i,l in enumerate(raw.splitlines()):
	print(i, repr(l), l.lower())
