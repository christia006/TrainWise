# ⚙️ TrainWise — Platform Latih Model ML Sendiri Secara Lokal

**TrainWise** adalah platform sederhana namun powerful untuk melatih model Machine Learning langsung dari komputer  tanpa perlu koneksi internet. Cocok buat eksperimen, riset pribadi, atau pembelajaran hands-on di bidang AI/ML.

---

## 🧠 Deskripsi Singkat

TrainWise memudahkan  dalam melakukan preprocessing data, memilih algoritma ML, melatih model, mengevaluasi performa, dan menyimpan hasilnya — semua dalam satu antarmuka interaktif. Dengan integrasi database PostgreSQL, setiap eksperimen tercatat dengan baik dan bisa dilacak kembali.

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

## 🛠 Teknologi yang Digunakan

Platform ini dibangun dengan kombinasi teknologi Python dan sistem database relasional untuk memastikan kestabilan dan keterlacakan eksperimen:

| Teknologi | Keterangan |
|-----------|------------|
| **Python** | Bahasa pemrograman utama untuk seluruh platform. |
| **Scikit-learn** | Untuk preprocessing (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`), pembuatan pipeline (`ColumnTransformer`), dan model ML (`RandomForestClassifier`, `DecisionTreeClassifier`, `SVC`). |
| **Pandas** | Manipulasi data dan loading file CSV ke dalam `DataFrame`. |
| **NumPy** | Operasi numerik dan pengelolaan array dengan performa tinggi. |
| **Matplotlib & Seaborn** | Visualisasi hasil eksperimen: Confusion Matrix dan Feature Importance. |
| **PostgreSQL** | Database relasional untuk menyimpan semua metadata eksperimen dan model. |
| **Psycopg2** | Adapter PostgreSQL untuk Python, mengelola koneksi dan query ke database. |
| **joblib** | Untuk menyimpan dan memuat ulang model yang sudah dilatih. |

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
