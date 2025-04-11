# Laporan Proyek Machine Learning - Andre Gregori Sangari

## Domain Proyek

Penyakit jantung adalah penyebab utama kematian di seluruh dunia. Menurut World Health Organization (WHO), pada tahun 2019, sekitar 17,9 juta orang meninggal karena penyakit kardiovaskular, yang mewakili 32% dari semua kematian global [1]. Di Indonesia, berdasarkan data Kementerian Kesehatan RI, penyakit jantung menjadi penyebab kematian nomor satu dengan prevalensi sebesar 1,5% dari total populasi [2].

Penyakit jantung koroner (Coronary Heart Disease/CHD) terjadi ketika pembuluh darah yang memasok jantung dengan darah, oksigen, dan nutrisi (arteri koroner) menjadi sempit akibat penumpukan plak. Kondisi ini dapat menyebabkan angina (nyeri dada), serangan jantung, gagal jantung, dan bahkan kematian mendadak. 

### Mengapa dan bagaimana masalah ini harus diselesaikan

Deteksi dini penyakit jantung sangat penting karena beberapa alasan:

1. **Tingkat Kematian yang Tinggi**: Penyakit jantung adalah penyebab kematian tertinggi di dunia.
2. **Peningkatan Keberhasilan Pengobatan**: Diagnosis dini meningkatkan kemungkinan penanganan yang efektif.
3. **Pengurangan Biaya Kesehatan**: Pencegahan dan intervensi dini lebih hemat biaya dibandingkan pengobatan kondisi lanjut.
4. **Peningkatan Kualitas Hidup**: Deteksi dini memungkinkan modifikasi gaya hidup dan pengobatan yang dapat meningkatkan kualitas hidup pasien.

Machine learning dapat membantu tenaga medis dalam melakukan deteksi dini dan prediksi risiko penyakit jantung berdasarkan parameter kesehatan pasien. Dengan menganalisis pola dalam data pasien sebelumnya, algoritma machine learning dapat mengidentifikasi faktor-faktor yang berkontribusi terhadap penyakit jantung dan memprediksi kemungkinan seorang pasien menderita kondisi tersebut.

### Referensi

[1] World Health Organization. (2021). [Cardiovascular diseases (CVDs)](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))

[2] Kementerian Kesehatan RI. (2018). [Hasil Utama Riskesdas 2018](https://kesmas.kemkes.go.id/assets/upload/dir_519d41d8cd98f00/files/Hasil-riskesdas-2018_1274.pdf). Jakarta: Kementerian Kesehatan RI.

[3] Rajkumar, A., & Reena, G. S. (2010). [Diagnosis of Heart Disease Using Datamining Algorithm](https://globaljournals.org/GJCST_Volume10/10-Diagnosis-of-Heart-Disease-Using-Datamining-Algorithm.pdf). Global Journal of Computer Science and Technology, 10(10), 38-43.

[4] Mohan, S., Thirumalai, C., & Srivastava, G. (2019). [Effective Heart Disease Prediction Using Hybrid Machine Learning Techniques](https://ieeexplore.ieee.org/document/8727774). IEEE Access, 7, 81542-81554.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, berikut adalah rumusan masalah yang akan diselesaikan dalam proyek ini:

1. Bagaimana mengembangkan model machine learning yang dapat memprediksi risiko penyakit jantung pada pasien dengan tingkat akurasi yang tinggi berdasarkan parameter klinis?
2. Apa faktor klinis yang paling berpengaruh dalam memprediksi risiko penyakit jantung pada pasien?
3. Bagaimana cara meningkatkan performa model prediksi penyakit jantung agar dapat menjadi alat bantu yang reliabel bagi tenaga medis?

### Goals

Tujuan dari proyek ini adalah:

1. Mengembangkan model machine learning yang dapat memprediksi risiko penyakit jantung dengan akurasi, presisi, dan recall yang tinggi.
2. Mengidentifikasi faktor-faktor klinis yang memiliki pengaruh signifikan terhadap risiko penyakit jantung.
3. Membandingkan performa berbagai algoritma machine learning dan memilih model terbaik yang dapat digunakan untuk deteksi dini penyakit jantung.

### Solution Statements

Untuk mencapai tujuan tersebut, berikut adalah solusi yang akan diterapkan:

1. Mengembangkan dan membandingkan beberapa model machine learning untuk klasifikasi:
   - Logistic Regression: Algoritma klasifikasi linear yang sederhana dengan interpretabilitas tinggi.
   - Random Forest: Algoritma ensemble berbasis decision tree yang dapat menangani fitur non-linear dan interaksi kompleks antar fitur.
   - Gradient Boosting: Algoritma boosting yang secara sekuensial memperbaiki kesalahan model sebelumnya.
   - Support Vector Machine (SVM): Algoritma yang efektif untuk data dimensi tinggi dan mampu menangani hubungan non-linear melalui fungsi kernel.

2. Melakukan hyperparameter tuning untuk meningkatkan performa model:
   - Menggunakan GridSearchCV dengan cross-validation untuk menemukan kombinasi parameter optimal untuk model terbaik.
   - Parameter yang dioptimalkan akan disesuaikan dengan algoritma yang dipilih.

3. Menggunakan metrik evaluasi yang sesuai untuk masalah medis:
   - Accuracy: Proporsi prediksi yang benar dari total prediksi.
   - Precision: Proporsi positif sejati dari semua yang diprediksi positif, penting untuk menghindari false positive dalam diagnosis.
   - Recall: Proporsi positif sejati dari semua positif aktual, penting untuk mengidentifikasi semua pasien yang berisiko.
   - F1-Score: Rata-rata harmonik dari precision dan recall.
   - ROC-AUC: Mengukur kemampuan model membedakan antara kelas positif dan negatif.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah Heart Disease Dataset yang tersedia secara publik. Dataset ini berisi data medis dan demografis dari individu yang telah menjalani berbagai tes untuk mendiagnosis penyakit jantung. Dataset ini dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

### Informasi Dataset
- **Jumlah Data:** 1.025 sampel
- **Jumlah Fitur:** 13 fitur + 1 target
- **Format:** CSV (Comma Separated Values)

### Variabel-variabel pada Heart Disease Dataset adalah sebagai berikut:

1. **age**: Usia pasien dalam tahun (numerik)
2. **sex**: Jenis kelamin (1 = laki-laki, 0 = perempuan) (kategorikal)
3. **cp**: Tipe nyeri dada (kategorikal)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Tekanan darah istirahat dalam mm Hg saat masuk ke rumah sakit (numerik)
5. **chol**: Kolesterol serum dalam mg/dl (numerik)
6. **fbs**: Gula darah puasa > 120 mg/dl (1 = ya, 0 = tidak) (kategorikal)
7. **restecg**: Hasil elektrokardiografi istirahat (kategorikal)
   - 0: Normal
   - 1: Memiliki kelainan gelombang ST-T
   - 2: Hipertrofi ventrikel kiri
8. **thalach**: Detak jantung maksimum yang dicapai (numerik)
9. **exang**: Angina yang dipicu oleh olahraga (1 = ya, 0 = tidak) (kategorikal)
10. **oldpeak**: Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat (numerik)
11. **slope**: Kemiringan segmen ST pada puncak olahraga (kategorikal)
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca**: Jumlah pembuluh darah utama (0-3) yang diwarnai oleh fluoroskopi (kategorikal)
13. **thal**: Status thalassemia (kategorikal)
    - 1: Normal
    - 2: Fixed defect
    - 3: Reversible defect
14. **target**: Diagnosis penyakit jantung (1 = ya, 0 = tidak) (kategorikal)

### Exploratory Data Analysis (EDA)

Untuk memahami lebih dalam tentang dataset, dilakukan beberapa analisis eksplorasi data.

#### 1. Distribusi Kelas Target

![Distribusi Kelas Target](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/target_distribution.png)

Dari visualisasi di atas, dapat diamati bahwa distribusi kelas dalam dataset cukup seimbang, dengan sekitar 54% pasien didiagnosis dengan penyakit jantung (kelas 1) dan 46% tidak memiliki penyakit jantung (kelas 0). Keseimbangan ini baik untuk model klasifikasi karena mengurangi risiko bias dalam prediksi.

#### 2. Korelasi Antar Fitur

![Korelasi Antar Fitur](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/correlation_matrix.png)

Dari matriks korelasi, beberapa insight penting yang diperoleh:
- Terdapat korelasi negatif yang kuat antara `thalach` (detak jantung maksimum) dan `age` (usia), menunjukkan bahwa detak jantung maksimum cenderung menurun dengan bertambahnya usia.
- `cp` (tipe nyeri dada) memiliki korelasi positif yang cukup kuat dengan `target`, mengindikasikan bahwa tipe nyeri dada tertentu berhubungan dengan peningkatan risiko penyakit jantung.
- `oldpeak` (depresi ST) menunjukkan korelasi negatif dengan `slope` (kemiringan segmen ST), yang konsisten dengan pemahaman medis.

#### 3. Distribusi Fitur Numerik Berdasarkan Target

![Distribusi Fitur Numerik](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/numeric_distributions.png)

Analisis distribusi fitur numerik berdasarkan target menunjukkan beberapa pola:
- Pasien dengan penyakit jantung cenderung memiliki usia yang lebih tinggi.
- Kolesterol serum rata-rata tidak menunjukkan perbedaan yang signifikan antara pasien dengan dan tanpa penyakit jantung, menunjukkan bahwa faktor ini mungkin bukan prediktor utama dalam dataset ini.
- Detak jantung maksimum (`thalach`) cenderung lebih rendah pada pasien dengan penyakit jantung.

#### 4. Analisis Fitur Kategorikal

![Analisis Fitur Kategorikal](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/categorical_distributions.png)

Analisis fitur kategorikal memberikan insight berikut:
- Pria (sex=1) memiliki prevalensi penyakit jantung yang lebih tinggi dibandingkan wanita dalam dataset ini.
- Pasien dengan nyeri dada tipe Asymptomatic (cp=3) memiliki prevalensi penyakit jantung yang sangat tinggi.
- Pasien dengan angina yang dipicu oleh olahraga (exang=1) lebih cenderung memiliki penyakit jantung.
- Jumlah pembuluh darah yang terpengaruh (ca) menunjukkan hubungan positif yang kuat dengan diagnosis penyakit jantung.

## Data Preparation

Beberapa tahapan data preparation yang dilakukan dalam proyek ini adalah:

### 1. Penanganan Missing Values

Setelah pemeriksaan awal, tidak ditemukan nilai yang hilang dalam dataset. Namun, untuk memastikan robustness model, tetap diterapkan strategi imputasi dengan menggunakan:
- `SimpleImputer` dengan strategi `median` untuk fitur numerik
- `SimpleImputer` dengan strategi `most_frequent` untuk fitur kategorikal

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

Strategi ini dipilih karena:
- Median lebih robust terhadap outlier dibandingkan mean untuk fitur numerik
- Most frequent (modus) adalah pendekatan yang umum untuk fitur kategorikal

### 2. Feature Scaling

Standarisasi fitur numerik dilakukan menggunakan `StandardScaler` untuk mengubah fitur-fitur numerik menjadi distribusi dengan mean 0 dan standar deviasi 1. Hal ini penting karena:
- Beberapa algoritma machine learning (seperti SVM dan Logistic Regression) sensitif terhadap skala fitur
- Fitur dengan skala yang berbeda dapat membuat proses optimasi menjadi lebih sulit
- Standarisasi membantu algoritma konvergen lebih cepat selama training

### 3. Encoding Fitur Kategorikal

Fitur kategorikal diubah menjadi format numerik menggunakan `OneHotEncoder`. Proses ini menghasilkan fitur biner (dummy variables) untuk setiap kategori dalam fitur kategorikal. Encoding ini penting karena:
- Algoritma machine learning bekerja dengan input numerik
- One-hot encoding menghindari pengenalan urutan yang tidak ada pada kategori nominal
- Mencegah algoritma memberikan bobot yang tidak proporsional pada kategori dengan nilai numerik yang lebih tinggi

### 4. Feature Engineering

Tidak dilakukan feature engineering yang ekstensif karena fitur-fitur yang tersedia sudah cukup representatif untuk prediksi penyakit jantung. Namun, kita bisa mempertimbangkan untuk menambahkan fitur baru seperti:
- BMI (jika tersedia data tinggi dan berat badan)
- Rasio kolesterol terhadap tekanan darah
- Kategori usia (young, middle-aged, elderly)

### 5. Train-Test Split

Dataset dibagi menjadi data training (80%) dan testing (20%) dengan stratifikasi berdasarkan variabel target untuk memastikan distribusi kelas yang seimbang di kedua subset.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

Stratifikasi penting karena:
- Memastikan proporsi kelas target yang sama di data training dan testing
- Mengurangi variance dalam estimasi performa model
- Sangat penting untuk dataset dengan distribusi kelas yang tidak seimbang

## Modeling

Dalam proyek ini, empat algoritma klasifikasi diterapkan dan dibandingkan untuk memprediksi risiko penyakit jantung: Logistic Regression, Random Forest, Gradient Boosting, dan Support Vector Machine (SVM).

### 1. Logistic Regression

Logistic Regression adalah algoritma klasifikasi linear yang memodelkan probabilitas kelas target menggunakan fungsi logistik.

```python
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
```

**Kelebihan:**
- Interpretabilitas tinggi: koefisien model menunjukkan kontribusi setiap fitur
- Efisien secara komputasional
- Performa baik untuk dataset dengan hubungan linear
- Menghasilkan probabilitas yang terkalibrasi dengan baik

**Kekurangan:**
- Tidak dapat menangkap hubungan non-linear yang kompleks
- Performa suboptimal pada dataset dengan dimensi tinggi
- Rentan terhadap multicollinearity

### 2. Random Forest

Random Forest adalah algoritma ensemble yang membangun multiple decision trees dan menggabungkan prediksi mereka.

```python
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
```

**Kelebihan:**
- Robust terhadap outlier dan noise
- Dapat menangkap hubungan non-linear dan interaksi kompleks
- Feature importance tersedia secara langsung
- Tidak memerlukan scaling fitur
- Kecil risiko overfitting (relatif terhadap decision tree tunggal)

**Kekurangan:**
- Interpretabilitas lebih rendah dibandingkan model linear
- Komputasi lebih intensif
- Dapat memerlukan lebih banyak memory untuk dataset besar
- Hyperparameter tuning dapat memakan waktu

### 3. Gradient Boosting

Gradient Boosting membangun model secara sekuensial, di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya.

```python
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])
```

**Kelebihan:**
- Performa yang sangat baik untuk berbagai masalah klasifikasi
- Dapat menangkap hubungan non-linear yang kompleks
- Robust terhadap outlier (dengan parameter yang tepat)
- Feature importance tersedia

**Kekurangan:**
- Sensitif terhadap hyperparameter
- Rentan terhadap overfitting jika tidak diatur dengan benar
- Lebih sulit diinterpretasi dibandingkan model linear
- Training yang lebih lambat dibandingkan Random Forest

### 4. Support Vector Machine (SVM)

SVM mencari hyperplane optimal yang memisahkan kelas dengan margin maksimal.

```python
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])
```

**Kelebihan:**
- Efektif dalam ruang dimensi tinggi
- Fleksibel melalui pemilihan kernel yang berbeda
- Robust terhadap overfitting pada dataset kecil hingga sedang
- Generalisasi yang baik ke data baru

**Kekurangan:**
- Tidak efisien secara komputasional untuk dataset besar
- Sulit diinterpretasi
- Sensitif terhadap scaling fitur
- Hyperparameter tuning yang menantang

### Hyperparameter Tuning

Berdasarkan performa awal, Random Forest dipilih untuk hyperparameter tuning menggunakan GridSearchCV dengan cross-validation.

```python
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    rf_pipeline, 
    param_grid=param_grid, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1
)
```

Parameter yang dioptimalkan:
- `n_estimators`: Jumlah trees dalam forest
- `max_depth`: Kedalaman maksimum setiap tree
- `min_samples_split`: Jumlah minimum sampel yang diperlukan untuk split internal
- `min_samples_leaf`: Jumlah minimum sampel yang diperlukan pada leaf node

Setelah proses tuning, kombinasi parameter terbaik ditemukan:
- n_estimators: 200
- max_depth: 10
- min_samples_split: 2
- min_samples_leaf: 1

Model Random Forest yang telah dituning ini dipilih sebagai model final karena:
1. Mencapai ROC-AUC tertinggi (0.91) dibandingkan model lainnya
2. Memiliki keseimbangan yang baik antara precision dan recall
3. Menunjukkan stabilitas performa yang baik dalam cross-validation
4. Menyediakan feature importance yang dapat diinterpretasi

## Evaluation

Untuk mengevaluasi performa model dalam memprediksi penyakit jantung, beberapa metrik evaluasi digunakan:

### 1. Confusion Matrix

![Confusion Matrix](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/confusion_matrix_random_forest_(tuned).png)

Confusion matrix memberikan gambaran visual tentang prediksi model dibandingkan dengan nilai sebenarnya:
- True Positive (TP): Jumlah pasien dengan penyakit jantung yang diprediksi benar
- True Negative (TN): Jumlah pasien tanpa penyakit jantung yang diprediksi benar
- False Positive (FP): Jumlah pasien tanpa penyakit jantung yang diprediksi salah (Type I error)
- False Negative (FN): Jumlah pasien dengan penyakit jantung yang diprediksi salah (Type II error)

### 2. Metrik Evaluasi Utama

Berikut adalah hasil evaluasi model Random Forest yang telah dituning:

| Metrik    | Nilai   | Formula                     | Penjelasan                                                   |
|-----------|---------|-----------------------------|------------------------------------------------------------|
| Accuracy  | 0.90    | (TP + TN) / (TP + TN + FP + FN) | Proporsi prediksi yang benar dari total prediksi             |
| Precision | 0.89    | TP / (TP + FP)              | Proporsi positif benar dari semua yang diprediksi positif    |
| Recall    | 0.93    | TP / (TP + FN)              | Proporsi positif benar dari semua yang sebenarnya positif    |
| F1 Score  | 0.91    | 2 * (Precision * Recall) / (Precision + Recall) | Rata-rata harmonik dari precision dan recall |
| ROC AUC   | 0.92    | Area Under ROC Curve        | Kemampuan model membedakan antara kelas positif dan negatif  |

### 3. ROC Curve

![ROC Curve](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/roc_comparison.png)

ROC Curve (Receiver Operating Characteristic Curve) memplot True Positive Rate (Sensitivity) terhadap False Positive Rate (1-Specificity) pada berbagai threshold. Area Under the Curve (AUC) mengukur kemampuan model untuk membedakan antara kelas positif dan negatif, dengan nilai dari 0.5 (tidak lebih baik dari random) hingga 1.0 (sempurna).

Model Random Forest final mencapai AUC 1.0, menunjukkan kemampuan diskriminasi yang sangat baik.

### 4. Feature Importance

![Feature Importance](https://github.com/andregregs/submission_predictive-analytics/blob/main/images/feature_importance.png)

Analisis feature importance dari model Random Forest menunjukkan bahwa lima fitur teratas yang berkontribusi pada prediksi penyakit jantung adalah:
1. **cp** (tipe nyeri dada): 0.25
2. **ca** (jumlah pembuluh darah utama): 0.18
3. **thalach** (detak jantung maksimum): 0.14
4. **thal** (status thalassemia): 0.13
5. **oldpeak** (depresi ST): 0.09

Temuan ini konsisten dengan literatur medis yang menunjukkan bahwa jenis nyeri dada, jumlah pembuluh darah yang terpengaruh, dan detak jantung maksimum adalah prediktor penting untuk penyakit jantung.

### 5. Cross-Validation

Untuk memastikan robustness model, 5-fold cross-validation dilakukan, menghasilkan:
- Mean ROC AUC: 0.89
- Standard Deviation: 0.03

Nilai standar deviasi yang rendah menunjukkan bahwa model memiliki stabilitas yang baik di berbagai subset data.

### Intepretasi Hasil

Model Random Forest yang telah dituning berhasil mencapai performa yang sangat baik dalam memprediksi risiko penyakit jantung, dengan akurasi 90% dan ROC AUC 0.92. Beberapa insight penting dari model:

1. Recall yang tinggi (0.93) sangat penting dalam konteks medis, karena kita ingin meminimalkan false negative (pasien yang berisiko tidak terdeteksi).

2. Precision yang baik (0.89) menunjukkan bahwa model tidak terlalu banyak menghasilkan false positive, sehingga dapat mengurangi kecemasan dan tes lanjutan yang tidak perlu.

3. Fitur klinis yang paling berpengaruh (tipe nyeri dada, jumlah pembuluh darah utama, dan detak jantung maksimum) dapat membantu tenaga medis fokus pada faktor-faktor paling relevan saat melakukan skrining awal.

Sebagai kesimpulan, model ini memiliki potensial yang besar untuk digunakan sebagai alat bantu dalam deteksi dini penyakit jantung, meskipun keputusan final harus tetap berada di tangan profesional medis.