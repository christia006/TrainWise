# ⚙️ TrainWise — Platform AutoML Lokal dengan PostgreSQL

**TrainWise** adalah platform pembelajaran mesin otomatis (AutoML) yang sepenuhnya berjalan secara lokal. Aplikasi ini memungkinkan pengguna untuk mengunggah dataset `.csv`, memilih algoritma, dan melatih model ML langsung di komputer mereka — tanpa ketergantungan cloud atau API eksternal.

## 🎯 Tujuan Proyek

- Membangun pipeline pelatihan ML yang bersih, reusable, dan dapat digunakan siapa pun.
- Mengintegrasikan penyimpanan metadata dan konfigurasi pelatihan ke dalam PostgreSQL.
- Menyediakan antarmuka interaktif (Streamlit) untuk eksperimen model, evaluasi performa, dan manajemen hasil pelatihan.
- Menyimpan model terlatih dalam format `.pkl` yang dapat digunakan kembali.

## 🧩 Fitur Utama

- Upload dataset lokal (`.csv`) melalui antarmuka.
- Preprocessing otomatis (missing values, encoding, scaling).
- Pilihan algoritma: Random Forest, Decision Tree, SVM.
- Evaluasi model (akurasi, matriks kebingungan, ROC).
- Metadata dan hasil disimpan di PostgreSQL.
- Model dapat diunduh dan digunakan kembali.

## 🛠️ Teknologi yang Digunakan

- Python · scikit-learn · pandas · matplotlib
- PostgreSQL (via pgAdmin 4) · SQLAlchemy
- Streamlit (antarmuka pengguna)

## 📌 Mengapa Proyek Ini Dilirik Google

TrainWise mencerminkan prinsip internal seperti yang digunakan di Vertex AI — pipeline pelatihan, evaluasi model, dan versioning. Menampilkan kemampuan dalam:

- Desain sistem machine learning terstruktur
- Clean Architecture & separation of concerns
- Integrasi database untuk manajemen eksperimen
