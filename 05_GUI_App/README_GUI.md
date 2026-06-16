# Petunjuk Penggunaan GUI Aplikasi Image Processing & ANN

Aplikasi ini merupakan alat untuk melakukan preprocessing citra rasi bintang (Zodiac) secara batch dan melakukan training/testing menggunakan model Artificial Neural Network (ANN) secara mandiri.

## 1. Persiapan Data
* Pastikan Anda memindahkan/menyalin seluruh file gambar (240 gambar) ke folder `05_GUI_App/input/`.
* Nama file gambar harus mengikuti format seperti: `Aquarius (0).jpg` hingga `Virgo (19).jpg`.

## 2. Cara Menjalankan Aplikasi
Buka terminal/command prompt pada folder `05_GUI_App` lalu jalankan perintah:
```bash
python app_gui.py
```

## 3. Langkah-Langkah Proses (Batch Processing)
Aplikasi memproses seluruh 240 gambar dalam sekali klik.
1. **Pilih Mode Pooling:** `average` (Average Pooling) atau `max` (Max Pooling).
2. **Atur Ukuran Output:** Secara default adalah 28x28.
3. Klik **1. Color-to-Grayscale Conversion** untuk merubah citra menjadi grayscale. Output disimpan pada folder `output/grayscale`.
4. Klik **2. Binarization** untuk mengubah background menjadi hitam dan rasi bintang menjadi putih menggunakan Otsu Thresholding. Output disimpan pada `output/binarization`.
5. Klik **3. Resizing** untuk melakukan tahap akhir yang mencakup crop, square canvas, dan pooling (Average/Max) hingga siap latih. Output antara disimpan pada `output/resizing/` dan hasil final (ready) disimpan di `output/ready/`. Gambar debug divisualisasikan pada folder `preview/`.
6. Klik **4. Create Dataset + Label** untuk mengonversi citra `.jpg` yang ready menjadi `inputs...npy` dan membuat `.npy` label one-hot array. Disimpan di folder `dataset/`.
7. Klik **5. Randomize Dataset** untuk mengacak urutan baris data secara tersinkronisasi antara input dan labelnya.

## 4. Pelatihan & Pengujian ANN
* Klik **6. ANN Training** untuk melatih model berdasarkan citra `random_inputs...npy`. Parameter disimpan dalam bentuk file `.npy` pada folder `ann/`.
* Klik **7. ANN Testing by Index** untuk menguji akurasi model pada satu citra pilihan dari indeks 0-239 pada dataset yang sudah diacak.

## 5. Fitur Preview
Pada panel sebelah kanan, Anda dapat memeriksa hasil tahapan preprocessing pada setiap gambar secara individual. Pilihlah _Class_, _Index_, dan _Stage_ (Original, Grayscale, Binarization, Resizing, Ready) untuk memvisualisasikan bagaimana perubahan pada gambar terjadi setelah tombol pada Panel Kiri dieksekusi.
