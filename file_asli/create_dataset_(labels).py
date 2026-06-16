import numpy as np
import os

dir = r"C:\Users\rende\OneDrive\Dokumen\data kuliah\semester 6\3 image processing\pertemuan 15\M10-M14 Klasif dg ANN Huruf Hangul NAS\M10-M14 Klasif dg ANN Huruf Hangul NAS\02b_Create_Label"

"""
CATATAN TENTANG DATASET LABELS YANG DIBANGUN INI
Kolom ke-0 berisi nomor sampel, 0-199
Kolom ke-1, 2, 3 s.d. 10 berisi label. 
Untuk baris ke-0 s.d 19  labelnya (10000 00000)
Untuk baris ke-20 s.d 39 labelnya (01000 00000)
dst
"""
m = 10   #Banyaknya kelas ada 10
n = 20                                                 #Banyaknya sampel per kelas: 20.
o = 1            #Banyaknya kolom tambahan, dalam hal ini kolom ke-0 untuk nomor sampel.
labels   = np.zeros(shape = (m*n, m+o), dtype = float)

for k in range(0, m*n):
    labels[k, 0] = k              #Kolom pertama dataset label diisi dengan nomor sampel.
    posisi = int(k/n) + 1 #Untuk menentukan posisi angka "1" di kolom mana (100, 010, atau 001).
    labels[k, posisi] = 1

print(labels)
print(labels.shape)

np.save(os.path.join(dir, "labels_" + str(m*n) + "_" + str(m+o)),  labels)
print("Selamat! Dataset labels untuk " + str(m) + " kelas, masing-masing terdiri dari "\
      + str(n) + " sampel berhasil dibuat." )
print("Kolom ke-0 telah ditambahkan untuk nomor sampel. Ukuran dataset labels menjadi", str(m+o), ",", str(m*n), ".")
print("File dataset labels ini telah disimpan di folder yang sama dengan program ini.")
