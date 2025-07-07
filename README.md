# ⚙️ TrainWise — Platform Latih Model ML Sendiri Secara Lokal

**TrainWise** adalah platform sederhana namun powerful untuk melatih model Machine Learning langsung dari komputer kamu tanpa perlu koneksi internet. Cocok buat eksperimen, riset pribadi, atau pembelajaran hands-on di bidang AI/ML.

---

## 🧠 Deskripsi Singkat

TrainWise memudahkan kamu dalam melakukan preprocessing data, memilih algoritma ML, melatih model, mengevaluasi performa, dan menyimpan hasilnya — semua dalam satu antarmuka interaktif. Dengan integrasi database PostgreSQL, setiap eksperimen tercatat dengan baik dan bisa dilacak kembali.

---

## 🔧 Fitur Utama

- ✅ Upload file CSV langsung dari lokal.
- 🧼 Auto-preprocessing data (handle missing value, encoding, dsb).
- 🤖 Dukungan algoritma: Decision Tree, Random Forest, dan SVM.
- 📊 Evaluasi otomatis (akurasi, precision, recall, F1-score).
- 📁 Simpan model jadi file `.pkl` untuk dipakai ulang.
- 🧠 Tracking semua eksperimen ke database PostgreSQL.
- 📈 Visualisasi hasil: confusion matrix & feature importance.

---

## 🛠 Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python
- **Library ML & Analisis**:  
  `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Database**: PostgreSQL + `psycopg2`
- **Framework UI**: Streamlit

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

## 🤝 Kontribusi & Kontak

Proyek ini dikembangkan oleh:

👤 **Christian J. Hutahaean**  
📫 christia006@gmail.com  
🌐 [LinkedIn](https://www.linkedin.com/in/christian-hutahaean)

Jika kamu tertarik untuk berkontribusi atau memiliki saran perbaikan, jangan ragu untuk menghubungi saya. Kontribusi terbuka untuk umum!

---

## 🧪 Tambahan: Semua Hasil Eksperimen

| Gambar Keseluruhan | |
|--------------------|--|
| Semua Eksperimen | ![All Experiments](all experiment.png) |
| Eksperimen 1 | ![Exp 1](experiment 1.png) |
| Eksperimen 2 | ![Exp 2](experiment 2.png) |
| Eksperimen 3 | ![Exp 3](experiment 3.png) |
| Eksperimen 4 | ![Exp 4](experiment 4.png) |
| Eksperimen 5 | ![Exp 5](experiment 5.png) |

---

> “Build locally. Think globally.” — TrainWise
