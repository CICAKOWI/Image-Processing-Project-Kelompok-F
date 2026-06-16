# Image Processing — Klasifikasi Aksara Sunda dengan ANN

Pipeline **end-to-end** untuk mengenali karakter **Aksara Sunda** (ka, ca, nga, ga, ja) menggunakan **Artificial Neural Network (ANN)** yang diimplementasikan dari nol dengan **NumPy** dan **Matplotlib** (tanpa TensorFlow/PyTorch).

---

## Alur Pipeline

```
00_Dataset_Mentah/          01_ImageProcessing/          02a_Create_Dataset/
  Foto mentah (jpg)           Preprocessing                 Gambar → Dataset
  (kontras manual)            → down-pooling 28x28          → inputs_*.npy
  → hasil/*.jpg               → *_ready.jpg                   │
        │                          │                          │
        └────────────>─────────────┘           ┌──────────────┘
                                                │
                                                ▼
                                        03_Randomize/
                                          Acak dataset
                                          → random_*.npy
                                                │
                                                ▼
                                    04_ANN_Training_Testing_Insample/
                                      Training JST (from scratch)
                                      → Bobot *.npy
                                      → Prediksi interaktif
```

---

## Detail Tiap Tahap

| Tahap | Folder | Input | Proses | Output |
|-------|--------|-------|--------|--------|
| **1** | `00_Dataset_Mentah/` | Foto mentah `.jpg` | Kontras diatur manual | `hasil/*.jpg` |
| **2** | `01_ImageProcessing/` | `hasil/*.jpg` | Otsu threshold → crop → padding → down-pooling 28×28 | `*_ready.jpg` |
| **3a** | `02a_Create_Dataset/` | `*_ready.jpg` | Flatten pixel → simpan ke `.npy` | `inputs_*.npy` |
| **3b** | `02b_Create_Label/` | — | One-hot encoding labels | `labels_*.npy` |
| **4** | `03_Randomize/` | `inputs_*.npy` + `labels_*.npy` | Acak dengan index yang sama | `random_*.npy` |
| **5** | `04_ANN_Training_Testing_Insample/` | `random_*.npy` | Training ANN (backpropagation manual, 100 epoch) → testing interaktif | Bobot `.npy` + prediksi |

---

## Teknologi

- **Python 3**
- **NumPy** — semua operasi numerik & matriks
- **Matplotlib** — baca/tampilkan gambar
- **Tanpa deep learning framework** — ANN ditulis manual

---

## Cara Menjalankan

1. **Preprocessing:** jalankan `01_ImageProcessing/Downs AverPool GS Kontras copy.py`
3. **Buat dataset:** copy `_ready.jpg` ke `02a_Create_Dataset/`, lalu jalankan script di folder `02a/02b`
4. **Acak:** jalankan `03_Randomize/01_randomize_dataset.py`
5. **Training & test:** jalankan `04_ANN_Training_Testing_Insample/01_ann_huruf_training_testing_insample.py`

---

> **Catatan:** Pipeline ini awalnya untuk 10 kelas Hangul, diadaptasi ke 5 kelas Aksara Sunda. Beberapa parameter (ukuran input, jumlah kelas) mungkin perlu disesuaikan.
