# 01_ImageProcessing — Preprocessing & Standarisasi Gambar

## Tujuan
Mengubah gambar bersih menjadi ukuran standar 28×28 piksel biner.

## Alur
1. Input gambar dari `00_Dataset_Mentah/hasil/*.jpg` (sudah diatur kontras manual)
2. Jalankan `Downs AverPool GS Kontras copy.py`
3. Proses: grayscale → Otsu thresholding → crop objek → padding → square canvas → down-pooling ke 28×28

## Script
| File | Fungsi |
|------|--------|
| `Downs AverPool GS Kontras copy.py` | Preprocessing utama (max pooling, 28×28) |
| `Downs AverPool GS Kontras.py` | Versi lama (average pooling, 20×20) |

## Output
- `*_ready.jpg` — gambar 28×28 siap dibuat dataset
- `_debug/` — visualisasi tiap tahap preprocessing
