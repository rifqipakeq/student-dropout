# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Seiring pertumbuhan jumlah mahasiswa, institusi menghadapi tantangan serius: **tingginya angka dropout (32%)**. Dropout mahasiswa berdampak langsung pada:
- Kerugian finansial institusi (kehilangan uang kuliah)
- Menurunnya reputasi dan akreditasi
- Terbuangnya sumber daya pengajaran

### Permasalahan Bisnis
**Bagaimana memprediksi mahasiswa yang berisiko dropout sedini mungkin** agar institusi dapat mengambil tindakan preventif (bimbingan akademik, bantuan finansial, dll) sebelum mahasiswa benar-benar keluar?

### Cakupan Proyek
Proyek ini mencakup:
1. Exploratory Data Analysis (EDA)
2. Feature engineering dan preprocessing data
3. Pembangunan model Machine Learning
4. Evaluasi model
5. Identifikasi feature importance
6. Pembuatan dashboard Tableau
7. Pembuatan prototype solusi machine learning yang siap digunakan
7. Penyusunan rekomendasi action plan


### Persiapan

Sumber data: `data.csv`

Setup environment:

1. Step 1 - Pastikan python sudah terinstall di komputer anda
Lakukan pengecekan dengan memasukan perintah *python --version* di terminal komputer anda.
Jika muncul versi Python, maka lanjut ke tahap berikutnya. Namun, jika belum terinstall, maka install dahulu melalui *https://www.python.org/downloads/*

2. Step 2 - Buka code editor favorit anda. Saya sarankan menggunakan VSCode

3. Step 3 - Masuk ke folder project
submission/
├─ `data.csv`
├─ `link_dashboard.txt`
├─ `features_names.pkl`
├─ `label_encoder.pkl`
├─ `model.pkl`
├─ `scaler.pkl`
├─ `notebook.ipynb`
├─ `app.py`
├─ `README.md`
├─ `requirements.txt`
└─ `Rifqi-Rahardian_Dicoding_Dashboard.png`

4. Step 4 - Buat virtual environment
Dalam folder project. Masukan perintah *python -m venv venv* di terminal. Perintah ini akan membuat folder *venv*

5. Step 5 - Aktifkan virtual environment
Masukan perintah di terminal *venv\Scripts\activate*

6. Step 6 - Install library yang dibutukan
Masukan perintah di terminal *pip install -r requirements.txt*

7. Stepp 7 - Jalankan file python
Bisa dengan memasukan perintah *python hr_attrition_model.py* ataupun menggunakan shortcut *Ctrl+Alt+n*


## Business Dashboard
Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.

## Menjalankan Sistem Machine Learning
Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib
```

1. Jalankan Notebook
Buka dan jalankan semua sel di `notebook.ipynb` untuk melatih model dan menyimpan artefak.

2. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
3. Aplikasi akan berjalan secara lokal di `http://localhost:8501` 

4. Jika ingin mengakses langsung dapat mengunjungi link ``

## Conclusion
Berdasarkan pengolahan dan analisa data, didapatkan beberapa fakta bahwa mahasiswa dengan biaya tunggakan memiliki risiko lebih tinggi untuk dropout yakni **~67%** dibanding dengan yang sudah melunasi yang hanya sebesar **~19%** serta bukan merupakan penerima beasiswa juga memiliki risiko tinggi yakni **~42%** berbanding dengan penerima beasiswa yang hanya **~16%**.
Dengan mengintegrasikan strategi data-driven yang komprehensif seperti Early Warning System, bantuan keuangan tepat sasaran, serta dashboard monitoring bagi dosen wali, institusi dapat bertransformasi dari pendekatan reaktif menjadi preventif, sehingga berpotensi signifikan menurunkan dropout rate, meningkatkan efisiensi intervensi, dan memperkuat kualitas pengambilan keputusan berbasis data dalam jangka panjang.

### Rekomendasi Action Items
**1. Program Early Warning System**
- Implementasikan aplikasi prediksi ini (`app.py`) ke dalam sistem informasi akademik
- Jalankan prediksi otomatis di akhir semester 1 untuk seluruh mahasiswa aktif
- Tetapkan ambang batas: mahasiswa dengan probabilitas dropout >60% masuk daftar intervensi prioritas
- Assign dosen wali/konselor khusus untuk mahasiswa berisiko tinggi dalam 2 minggu sejak identifikasi

**2. Program Bantuan Keuangan Darurat**
- Buat skema cicilan atau penangguhan biaya kuliah bagi mahasiswa dengan hambatan finansial
- Tingkatkan kuota beasiswa internal — fokus pada mahasiswa semester 2-4 yang menunjukkan potensi akademik
- Buat hotline keuangan anonim agar mahasiswa tidak malu melaporkan kesulitan finansial
- Target: turunkan dropout akibat masalah finansial sebesar 40% dalam 1 tahun ajaran

**3. Program Orientasi dan Onboarding yang Lebih Kuat**
- Perkuat program orientasi mahasiswa baru (OSPEK) dengan sesi manajemen akademik dan finansial
- Buat workshop manajemen stres dan time management di awal setiap semester

**4. Dashboard Monitoring untuk Dosen Wali**
- Sediakan dashboard real-time bagi setiap dosen wali yang menampilkan status risiko mahasiswa bimbingannya
- Integrasikan data absensi, nilai UTS, dan pembayaran kuliah dalam satu tampilan
- Wajibkan pertemuan tatap muka minimal 2x per semester bagi mahasiswa dengan risiko sedang/tinggi

**5. Perbaikan Pengalaman Mahasiswa Non-Tradisional**
- Mahasiswa yang mendaftar di usia >25 tahun dan kelas malam memiliki risiko lebih tinggi
- Buat program kelas fleksibel/hybrid untuk mengakomodasi mahasiswa yang juga bekerja
- Sediakan fasilitas child care untuk mahasiswa yang sudah menikah/memiliki anak

**6. Evaluasi dan Perbaikan Berkelanjutan**
- Re-train model prediksi setiap semester dengan data terbaru untuk menjaga akurasi
- Lakukan survei exit interview kepada setiap mahasiswa yang dropout untuk validasi model
- Tetapkan KPI dropout rate tahunan: targetkan penurunan 5% per tahun selama 3 tahun ke depan
- Benchmark dengan institusi lain yang berhasil menurunkan angka dropout
