import numpy as np
import os

#USER ENTRY
print("Pilih dataset berdasarkan mode pooling yang digunakan.")
print("Ketik: average atau max")
mode_pooling = input("Mode pooling (average/max): ")
mode_pooling = mode_pooling.lower().strip()

if mode_pooling not in ["average", "max"]:
    print("Mode pooling salah.")
    print("Program dihentikan. Ketik hanya: average atau max")
    exit()

dir_base = r"d:\tugas proyek img process\02b_Create_Label"
dir = os.path.join(dir_base, mode_pooling)

"""
Urutan kelas:

0-9   = Aquarius
10-19  = Aries
...
"""

m = 12      #jumlah kelas
n = 10     #sampel per kelas
o = 1      #kolom nomor sampel

labels = np.zeros(shape=(m*n, m+o), dtype=float)

for k in range(0, m*n):
    labels[k, 0] = k

    posisi = int(k/n) + 1
    labels[k, posisi] = 1

print(labels)
print(labels.shape)

if not os.path.exists(dir):
    os.makedirs(dir)

nama_file = "labels_" + str(m*n) + "_" + str(m+o) + ".npy"

np.save(os.path.join(dir, nama_file), labels)

print("")
print("Dataset labels berhasil dibuat.")
print("Ukuran dataset :", labels.shape)
print("Nama file      :", nama_file)