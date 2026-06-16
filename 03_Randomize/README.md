# 03_Randomize — Pengacakan Dataset

## Tujuan
Mengacak urutan baris input dan label dengan urutan yang sama agar pasangan data-label tetap terjaga.

## Alur
1. Membaca `inputs_*.npy` dari `02a_Create_Dataset/`
2. Membaca `labels_*.npy` dari `02b_Create_Label/`
3. Membuat index acak, lalu mengurutkan ulang kedua dataset
4. Menyimpan hasil ke folder ini

## Script
| File | Fungsi |
|------|--------|
| `01_randomize_dataset.py` | Mengacak dataset input & label |

## Output
- `random_inputs_*.npy`
- `random_labels_*.npy`
