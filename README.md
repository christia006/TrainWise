<div align="center">

# ⚙️ **TrainWise — Platform Latih Model ML Sendiri Secara Lokal**

*Platform sederhana namun powerful untuk melatih model Machine Learning langsung dari komputer  
tanpa perlu koneksi internet. Cocok untuk eksperimen, riset pribadi, atau pembelajaran hands-on di bidang AI/ML.*

*Built with the tools and technologies:*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-f7931e?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Handling-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-5790c8?logo=seaborn&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql&logoColor=white)
![Psycopg2](https://img.shields.io/badge/Psycopg2-PostgreSQL%20Adapter-2c5d8e?logo=postgresql&logoColor=white)
![joblib](https://img.shields.io/badge/joblib-Model%20Persistence-339933?logo=python&logoColor=white)

![Offline](https://img.shields.io/badge/Offline%20Training-💻-important)
![Easy Experiment Tracking](https://img.shields.io/badge/Experiment%20Tracking-📊-success)
![Privacy First](https://img.shields.io/badge/Privacy%20First-🔒-blue)

</div>
## 🔄 Alur Program 

Bagaimana sebenarnya TrainWise bekerja di balik layar? Di bawah ini adalah perjalanan lengkap sebuah eksperimen dari awal hingga akhir. Bayangkan ini seperti "dapur rahasia" sistem AutoML kamu — otomatis, rapi, dan bisa diandalkan.

---

### 🛠️ 1. Inisiasi & Persiapan

📁 Sebelum memulai eksperimen:
- Program memastikan struktur folder penting tersedia: `data/`, `models/`, dan `logs/`.
- Koneksi ke **PostgreSQL** dibuka dan tabel-tabel penting diverifikasi (atau dibuat otomatis).
  
> Semua siap!.

---

### 📝 2. Pencatatan Eksperimen Baru

📌 Setiap eksperimen diawali dengan mencatat:
- **Nama eksperimen**, **nama file dataset**, dan **target kolom** ke dalam tabel `experiments`.
- Status awal diatur menjadi `'running'`.

> Bayangkan aja seperti menulis resep baru di buku masak eksperimenmulah.

---

### 🧼 3. Pra-Pemrosesan Data Otomatis

📊 Begitu file CSV diunggah:
- Fitur **numerik dan kategorikal** dikenali otomatis.
- Dilakukan langkah preprocessing:
  - **Nilai hilang**? Diimputasi! (mean untuk numerik, modus untuk kategorikal).
  - **Numerik**? Di-scale dengan `StandardScaler`.
  - **Kategorikal**? Diubah ke angka lewat One-Hot Encoding.
- Split data ke **training dan testing** set.

> Data mentah diubah menjadi bahan siap masak untuk model machine learning.

---

### 🤖 4. Pelatihan & Evaluasi Model

Untuk setiap model yang dipilih:

#### ✅ Decision Tree  
#### 🌲 Random Forest  
#### 💠 Support Vector Machine (SVM)

Langkah-langkah yang dilakukan:

1. **Latih model** dengan data training.
2. **Simpan model** ke `.joblib` dan catat path-nya ke database.
3. **Evaluasi model**:
   - Gunakan metrik standar: Akurasi, Presisi, Recall, F1-Score, ROC AUC.
   - Hasil disimpan ke tabel `metrics`.
4. **Visualisasi otomatis**:
   - ✅ Confusion Matrix untuk melihat performa klasifikasi.
   - 🌟 Feature Importance (khusus model berbasis pohon).

> Satu per satu model diuji. Siapa yang tampil terbaik? Semua dicatat dengan rapi.

---

### ✅ 5. Penyelesaian & Penutupan

📌 Setelah semua model selesai:
- Status eksperimen diperbarui ke `'completed'`.
- Jika terjadi error saat proses, status otomatis berubah menjadi `'failed'` agar mudah dilacak.

🔒 Terakhir, koneksi ke database PostgreSQL ditutup.

> Seperti menutup buku resep setelah masakan siap disajikan.

---

## 🚀 Fitur Utama Platform AutoML

Platform **TrainWise** dirancang untuk menyederhanakan proses machine learning secara end-to-end, dengan fokus utama pada klasifikasi data. Berikut fitur-fitur unggulannya:

### ⚙️ Preprocessing Data Otomatis
- Menangani nilai hilang menggunakan:
  - **Imputasi mean** untuk kolom numerik.
  - **Imputasi modus** untuk kolom kategorikal.
- Scaling fitur numerik menggunakan `StandardScaler`.
- Encoding fitur kategorikal dengan **One-Hot Encoding**.
- Semua transformasi disatukan dalam pipeline `ColumnTransformer` untuk efisiensi.

### 🤖 Pelatihan dan Evaluasi Model Serbaguna
- Mendukung model klasifikasi:  
  ✅ Decision Tree  
  ✅ Random Forest  
  ✅ Support Vector Machine (SVM)
- Hyperparameter dapat disesuaikan.
- Evaluasi model meliputi:  
  `Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC AUC`.

### 🗃️ Manajemen Eksperimen Terpusat
- Semua eksperimen dicatat ke **PostgreSQL** dengan metadata lengkap:
  - Nama eksperimen, dataset, target kolom.
  - Status (berjalan, selesai, gagal).
  - Timestamp mulai dan selesai.
  - Catatan tambahan.

### 🧠 Pencatatan & Pelacakan Model
- Setiap model yang dilatih dicatat dengan informasi berikut:
  - Nama model & path file.
  - Parameter pelatihan & waktu eksekusi.
  - Log file spesifik untuk model tersebut.

### 📊 Penyimpanan Metrik Kinerja
- Skor evaluasi **train & test set** tersimpan otomatis.
- Memungkinkan perbandingan kinerja antar eksperimen dan model.

### 📈 Visualisasi Hasil Otomatis
- Menyimpan hasil visualisasi seperti:
  - **Confusion Matrix** untuk mengevaluasi prediksi.
  - **Feature Importance** untuk model berbasis pohon (Decision Tree, Random Forest).

### 💾 Penyimpanan Artefak Model
- Model tersimpan dalam format `.joblib`:
  - Mudah di-reload untuk inference.
  - Memungkinkan reusabilitas model tanpa retraining.

### 📝 Sistem Logging Ekstensif
- Logging menyeluruh ke:
  - Konsol selama runtime.
  - File log terpisah untuk tiap eksperimen.
- Mempermudah debugging & pelacakan kesalahan.


---

## 📸 Hasil & Dokumentasi Visual

Berikut adalah beberapa hasil eksperimen dan visualisasi yang dihasilkan dari TrainWise:

### 🔍 Evaluasi Model (Confusion Matrix)

| Model | Gambar |
|-------|--------|
| Decision Tree (Exp 3) | ![Decision Tree CM Exp3](logs/plots/DecisionTree_confusion_matrix_exp3.png) |
| Decision Tree (Exp 5) | ![Decision Tree CM Exp5](logs/plots/DecisionTree_confusion_matrix_exp5.png) |
| Random Forest (Exp 3) | ![Random Forest CM Exp3](logs/plots/RandomForest_confusion_matrix_exp3.png) |
| Random Forest (Exp 4) | ![Random Forest CM Exp4](logs/plots/RandomForest_confusion_matrix_exp4.png) |
| Random Forest (Exp 7) | ![Random Forest CM Exp7](logs/plots/RandomForest_confusion_matrix_exp7.png) |
| SVM (Exp 3) | ![SVM CM Exp3](logs/plots/SVM_confusion_matrix_exp3.png) |
| SVM (Exp 6) | ![SVM CM Exp6](logs/plots/SVM_confusion_matrix_exp6.png) |

---

### 🌲 Feature Importance (untuk model pohon)

| Model | Gambar |
|-------|--------|
| Decision Tree (Exp 3) | ![DT Feature Importance Exp3](logs/plots/DecisionTree_feature_importance_exp3.png) |
| Decision Tree (Exp 5) | ![DT Feature Importance Exp5](logs/plots/DecisionTree_feature_importance_exp5.png) |
| Random Forest (Exp 3) | ![RF Feature Importance Exp3](logs/plots/RandomForest_feature_importance_exp3.png) |
| Random Forest (Exp 4) | ![RF Feature Importance Exp4](logs/plots/RandomForest_feature_importance_exp4.png) |
| Random Forest (Exp 7) | ![RF Feature Importance Exp7](logs/plots/RandomForest_feature_importance_exp7.png) |

---
## 🧪 Tambahan: Semua Hasil Eksperimen

Berikut dokumentasi visual dari hasil-hasil eksperimen yang telah dilakukan:

| Judul Visualisasi   | Pratinjau |
|---------------------|-----------|
| Semua Eksperimen    | ![All Experiments](images/all%20experiment.png) |
| Eksperimen 1        | ![Experiment 1](images/experiment%201.png) |
| Eksperimen 2        | ![Experiment 2](images/experiment%202.png) |
| Eksperimen 3        | ![Experiment 3](images/experiment%203.png) |
| Eksperimen 4        | ![Experiment 4](images/experiment%204.png) |
| Eksperimen 5        | ![Experiment 5](images/experiment%205.png) |

---



Proyek ini dikembangkan oleh:

👤 **Christian J. Hutahaean**  

---
> “Build locally. Think globally.” — TrainWise
