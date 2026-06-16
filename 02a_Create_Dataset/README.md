# 02a_Create_Dataset — Membuat Dataset Input (.npy)

## Tujuan
Mengubah gambar `_ready.jpg` menjadi file dataset numerik format `.npy`.

## Alur
1. Copy/pindahkan file `*_ready.jpg` dari `01_ImageProcessing/` ke folder ini
2. Rename sesuai format: `aksun_ka (0)_ready.jpg`, `aksun_ka (1)_ready.jpg`, ... (20 per kelas)
3. Jalankan `01_create_dataset_(inputs) copy.py`
4. Setiap baris = 1 sampel (kolom 0 = index, kolom 1–784 = pixel)

## Script
| File | Fungsi |
|------|--------|
| `01_create_dataset_(inputs) copy.py` | Untuk 5 kelas Aksara Sunda (100 sampel) |
| `01_create_dataset_(inputs).py` | Versi asli 10 kelas Hangul (200 sampel) |

## Output
- `inputs_100_785.npy` — dataset input siap diacak & dilatih
