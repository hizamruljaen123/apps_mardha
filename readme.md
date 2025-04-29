# Implementasi Pohon Keputusan C5.0

Proyek ini mengimplementasikan algoritma pohon keputusan C5.0 untuk memprediksi penyakit jantung. Implementasi ini didasarkan pada langkah-langkah berikut:

1.  **Pemuatan dan Praproses Data:**
    *   Data dimuat dari file Excel (`data_latih.xlsx`).
    *   Kolom `Tekanan_Darah` dipisah menjadi kolom `Sistolik` dan `Diastolik`.
    *   Fitur kategorikal (`Jenis_Kelamin`, `Nyeri_Dada`, `Sesak_Napas`, `Kelelahan`) dipetakan ke nilai numerik.

2.  **Pembagian Data:**
    *   Data dibagi menjadi 80% data pelatihan dan 20% data pengujian.

3.  **Implementasi Algoritma Pohon Keputusan C5.0:**
    *   Fungsi `entropy` menghitung entropi dari variabel target.
    *   Fungsi `information_gain` menghitung perolehan informasi dari sebuah fitur.
    *   Fungsi `split_info` menghitung informasi pemisahan dari sebuah fitur.
    *   Fungsi `gain_ratio` menghitung rasio perolehan dari sebuah fitur.
    *   Kelas `DecisionNode` mewakili sebuah node dalam pohon keputusan.
    *   Fungsi `build_tree` secara rekursif membangun pohon keputusan dengan memilih fitur terbaik untuk membagi pada setiap node.

4.  **Prediksi:**
    *   Fungsi `predict_single_tree` memprediksi kelas untuk sampel tunggal menggunakan pohon keputusan tunggal.
    *   Fungsi `predict` memprediksi kelas untuk sampel tunggal, menggunakan boosting jika diaktifkan.

5.  **Pelatihan:**
    *   Endpoint `/train_model` melatih model pohon keputusan menggunakan data pelatihan dan parameter yang ditentukan.

6.  **Visualisasi:**
    *   Endpoint `/api/tree_visualization` menyediakan representasi JSON dari struktur pohon keputusan untuk visualisasi.
    *   Endpoint `/api/rules` menyediakan daftar aturan keputusan yang diekstraksi dari pohon.

**Penjelasan Sederhana Algoritma C5.0:**

Algoritma C5.0 adalah metode untuk membangun pohon keputusan, yang digunakan untuk memprediksi kategori (seperti "memiliki penyakit jantung" atau "tidak memiliki penyakit jantung") berdasarkan serangkaian fitur input (seperti usia, tekanan darah, dll.). Berikut adalah uraian yang disederhanakan:

1.  **Apa itu Entropi?** Bayangkan Anda memiliki sekantong kelereng, ada yang merah dan ada yang biru. Entropi mengukur seberapa campur aduk warna-warna tersebut. Jika semua kelereng berwarna sama, entropinya rendah (tidak campur). Jika warnanya terbagi rata, entropinya tinggi (sangat campur aduk).

2.  **Perolehan Informasi (Information Gain):** Ini mengukur seberapa besar sebuah fitur membantu untuk memisahkan warna-warna di dalam kantong. Misalnya, jika memisahkan kelereng berdasarkan ukuran menghasilkan dua kantong yang masing-masing kantong memiliki kelereng yang sebagian besar berwarna sama, maka ukuran memiliki perolehan informasi yang tinggi.

3.  **Rasio Perolehan (Gain Ratio):** C5.0 menggunakan rasio perolehan alih-alih hanya perolehan informasi. Rasio perolehan menyesuaikan perolehan informasi untuk memperhitungkan fitur-fitur yang mungkin memiliki banyak nilai.

4.  **Membangun Pohon:**
    *   Mulai dengan seluruh dataset (kantong kelereng).
    *   Hitung rasio perolehan untuk setiap fitur.
    *   Pilih fitur dengan rasio perolehan tertinggi. Fitur ini menjadi "keputusan" di bagian atas pohon.
    *   Bagi dataset berdasarkan nilai-nilai fitur tersebut. Sekarang Anda memiliki kantong kelereng yang lebih kecil.
    *   Ulangi proses untuk setiap kantong yang lebih kecil. Terus lakukan sampai Anda mencapai titik di mana:
        *   Semua kelereng di dalam kantong berwarna sama (entropi rendah).
        *   Anda kehabisan fitur untuk dibagi.
        *   Kantong menjadi terlalu kecil.

5.  **Membuat Prediksi:** Untuk memprediksi apakah seorang pasien baru menderita penyakit jantung, Anda mulai dari bagian atas pohon dan menjawab pertanyaan (fitur) saat Anda menuruni pohon hingga Anda mencapai daun. Daun memberi tahu Anda kategori yang diprediksi (penyakit jantung atau tidak ada penyakit jantung).

6.  **Boosting (Opsional):** C5.0 juga dapat menggunakan boosting, yang berarti membangun beberapa pohon. Setiap pohon mencoba memperbaiki kesalahan pohon-pohon sebelumnya. Hal ini seringkali menghasilkan akurasi yang lebih baik.

Ini adalah penjelasan yang disederhanakan, dan implementasi sebenarnya melibatkan lebih banyak detail, seperti menangani nilai-nilai yang hilang dan memangkas pohon untuk mencegah overfitting.

**Glosarium:**

*   **Entropi:** Ukuran ketidakpastian atau keacakan dalam suatu dataset. Semakin tinggi entropi, semakin tidak pasti hasilnya.
*   **Fitur:** Atribut atau karakteristik yang digunakan untuk membuat prediksi (misalnya, usia, tekanan darah).
*   **Gain Ratio:** Ukuran seberapa baik sebuah fitur memisahkan data, disesuaikan untuk memperhitungkan fitur-fitur dengan banyak nilai.
*   **Node:** Titik dalam pohon keputusan yang mewakili keputusan atau hasil.
*   **Leaf:** Node akhir dalam pohon keputusan yang mewakili prediksi.
*   **Overfitting:** Ketika model belajar data pelatihan terlalu baik dan tidak dapat menggeneralisasi dengan baik ke data baru.
*   **Boosting:** Teknik untuk meningkatkan akurasi model dengan menggabungkan beberapa model yang lebih lemah.
*   **Target:** Variabel yang ingin diprediksi (misalnya, penyakit jantung).