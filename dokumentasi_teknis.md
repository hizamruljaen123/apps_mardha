# Dokumentasi Teknis Implementasi C5.0

Dokumentasi ini menjelaskan implementasi algoritma pohon keputusan C5.0 dalam file `web/app.py`.

## Tahapan Proses

1.  **Pemuatan dan Praproses Data (Data Loading and Preprocessing):**
    *   Data dimuat dari file Excel `data_latih.xlsx` menggunakan `pandas.read_excel()`.
    *   Kolom `Tekanan_Darah` dipisah menjadi dua kolom numerik, `Sistolik` dan `Diastolik`, menggunakan fungsi `str.split()` dan `astype(int)`. Kolom `Tekanan_Darah` kemudian dihapus.
    *   Kolom-kolom kategorikal `Jenis_Kelamin`, `Nyeri_Dada`, `Sesak_Napas`, dan `Kelelahan` dikonversi menjadi nilai numerik menggunakan fungsi `map()`.

2.  **Pembagian Data (Data Splitting):**
    *   Data dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan fungsi `pandas.DataFrame.sample()` untuk memilih sampel acak untuk pelatihan, dan `pandas.DataFrame.drop()` untuk mendapatkan data pengujian.

3.  **Implementasi Algoritma C5.0 (C5.0 Decision Tree Implementation):**
    *   **Fungsi `entropy(target, sample_weights=None)`:** Menghitung entropi dari variabel target. Entropi adalah ukuran ketidakpastian atau keacakan dalam data. Fungsi ini menggunakan bobot sampel jika diberikan.
    *   **Fungsi `information_gain(data, feature, target, sample_weights=None)`:** Menghitung perolehan informasi (information gain) dari sebuah fitur. Perolehan informasi mengukur seberapa banyak sebuah fitur mengurangi ketidakpastian tentang variabel target.
    *   **Fungsi `split_info(data, feature, sample_weights=None)`:** Menghitung informasi pemisahan (split info) dari sebuah fitur. Split info digunakan untuk menormalisasi perolehan informasi dan mencegah fitur dengan banyak nilai unik dari yang dipilih secara tidak adil.
    *   **Fungsi `gain_ratio(data, feature, target, sample_weights=None)`:** Menghitung rasio perolehan (gain ratio) dari sebuah fitur. Gain ratio adalah perolehan informasi yang dinormalisasi oleh informasi pemisahan.
    *   **Kelas `DecisionNode`:** Merepresentasikan sebuah node dalam pohon keputusan. Setiap node memiliki fitur (feature), ambang batas (threshold), nilai (value), cabang kiri (left), dan cabang kanan (right).
    *   **Fungsi `build_tree(data, target, features, depth=0, max_depth=3, min_samples_split=2, sample_weights=None)`:** Membangun pohon keputusan secara rekursif. Fungsi ini memilih fitur terbaik untuk membagi data pada setiap node, berdasarkan rasio perolehan tertinggi. Proses ini berlanjut hingga mencapai kasus dasar (base cases), seperti kedalaman maksimum tercapai, jumlah sampel minimum tidak terpenuhi, atau tidak ada fitur yang tersisa untuk dibagi.

4.  **Prediksi (Prediction):**
    *   **Fungsi `predict_single_tree(node, sample)`:** Memprediksi kelas untuk sampel tunggal menggunakan pohon keputusan tunggal. Fungsi ini menelusuri pohon, mengikuti cabang yang sesuai berdasarkan nilai-nilai fitur sampel. Ketika mencapai node daun (leaf node), fungsi ini mengembalikan nilai prediksi dari node tersebut.
    *   **Fungsi `predict(sample, use_boosting=False)`:** Memprediksi kelas untuk sampel tunggal, menggunakan boosting jika diaktifkan. Jika boosting digunakan, fungsi ini menggabungkan prediksi dari beberapa pohon keputusan yang dilatih secara iteratif.

5.  **Pelatihan Model (Model Training):**
    *   Endpoint `/train_model` menerima parameter pelatihan (seperti kedalaman maksimum pohon, jumlah sampel minimum untuk pemisahan, dan apakah akan menggunakan boosting).
    *   Fungsi `build_tree` digunakan untuk membangun pohon keputusan.
    *   Jika boosting diaktifkan, beberapa pohon keputusan dibangun secara iteratif, dengan bobot sampel disesuaikan pada setiap iterasi untuk fokus pada sampel yang salah diklasifikasikan.

6.  **API Endpoints:**
    *   `/api/predictions`: Mengembalikan hasil prediksi.
    *   `/api/training_data`: Mengembalikan data pelatihan.
    *   `/api/results`: Mengembalikan hasil.
    *   `/api/tree_visualization`: Mengembalikan struktur pohon untuk visualisasi.
    *   `/api/rules`: Mengembalikan aturan-aturan dari pohon keputusan.

## Glosarium Istilah Teknis

*   **Pandas:** Library Python untuk analisis dan manipulasi data.
*   **Numpy:** Library Python untuk komputasi numerik.
*   **Flask:** Framework web mikro Python.
*   **Dataframe:** Struktur data tabular dalam Pandas.
*   **Entropi (Entropy):** Ukuran ketidakpastian atau keacakan dalam suatu dataset.
*   **Fitur (Feature):** Atribut atau karakteristik yang digunakan untuk membuat prediksi.
*   **Variabel Target (Target Variable):** Variabel yang ingin diprediksi.
*   **Information Gain:** Ukuran seberapa banyak sebuah fitur mengurangi ketidakpastian tentang variabel target.
*   **Split Info:** Ukuran kompleksitas pemisahan yang dilakukan oleh sebuah fitur.
*   **Gain Ratio:** Ukuran seberapa baik sebuah fitur memisahkan data, dinormalisasi oleh split info.
*   **Decision Node:** Node dalam pohon keputusan yang mewakili sebuah keputusan berdasarkan nilai fitur.
*   **Leaf Node:** Node akhir dalam pohon keputusan yang mewakili hasil prediksi.
*   **Threshold:** Nilai batas yang digunakan untuk membagi data pada node keputusan.
*   **Overfitting:** Kondisi ketika model belajar data pelatihan terlalu baik dan gagal menggeneralisasi dengan baik ke data baru.
*   **Boosting:** Teknik ensemble learning yang menggabungkan beberapa model lemah untuk menghasilkan model yang lebih kuat.
*   **Sample Weights:** Bobot yang diberikan kepada setiap sampel dalam data pelatihan, digunakan dalam algoritma boosting.
*   **JSON (JavaScript Object Notation):** Format data yang ringan dan mudah dibaca yang digunakan untuk bertukar data antara server dan aplikasi web.
*   **API (Application Programming Interface):** Sekumpulan definisi dan protokol yang digunakan untuk membangun dan mengintegrasikan perangkat lunak aplikasi.
*   **Endpoint:** Titik akhir URL yang menyediakan akses ke sumber daya atau fungsionalitas tertentu dari sebuah API.
*   **Data Training:** Data yang digunakan untuk melatih model machine learning.
*   **Data Testing:** Data yang digunakan untuk mengevaluasi kinerja model machine learning yang telah dilatih.