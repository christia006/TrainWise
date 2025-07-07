# ⚙️ TrainWise — Platform Latih Model ML Sendiri Secara Lokal

**TrainWise** adalah alat bantu untuk latihan machine learning secara lokal. Kamu bisa upload dataset `.csv`, pilih model, latih langsung dari komputer kamu, dan simpan modelnya buat dipakai ulang. Semuanya tanpa perlu akses internet.

## 🔧 Fitur Utama

- Upload file CSV langsung dari lokal.
- Bersihin data otomatis (missing value, encoding, dsb).
- Pilih algoritma: Decision Tree, Random Forest, atau SVM.
- Lihat hasil evaluasi seperti akurasi & confusion matrix.
- Semua hasil dan info disimpan ke database PostgreSQL.
- Bisa export model jadi file `.pkl`.

## 🛠 Teknologi yang Dipakai

- Python · pandas · scikit-learn
- PostgreSQL (pakai pgAdmin 4)
- Streamlit buat tampilan interaktif

## 💡 Kenapa Menarik

Proyek  ML tanpa ribet setup cloud. Semua dikelola rapi, datanya disimpan dengan PostgreSQL, dan workflow-nya mirip seperti alat internal yang dipakai di perusahaan besar. Intinya: alat sederhana tapi serius.
