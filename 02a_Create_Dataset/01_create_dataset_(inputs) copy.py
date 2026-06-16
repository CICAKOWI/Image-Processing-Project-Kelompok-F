

import numpy as np
import matplotlib.pyplot as plt
import os

#USER ENTRIES
print("Pilih dataset berdasarkan mode pooling yang digunakan.")
print("Ketik: average atau max")
mode_pooling = input("Mode pooling (average/max): ")
mode_pooling = mode_pooling.lower().strip()

if mode_pooling not in ["average", "max"]:
    print("Mode pooling salah.")
    print("Program dihentikan. Ketik hanya: average atau max")
    exit()

dir_in_base = r"d:\tugas proyek img process\01_ImageProcessing"
dir_out_base = r"d:\tugas proyek img process\02a_Create_Dataset"

dir_in = os.path.join(dir_in_base, mode_pooling)
dir_out = os.path.join(dir_out_base, mode_pooling)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

file0 = "Aquarius ("
file1 = "Aries ("
file2 = "Cancer ("
file3 = "Capricorn ("
file4 = "Gemini ("
file5 = "Leo ("
file6 = "Libra ("
file7 = "Pisces ("
file8 = "Sagitarius ("
file9 = "Scorpio ("
file10 = "Taurus ("
file11 = "Virgo ("

ext = ")_ready.jpg"

juml_kelas = 12
juml_sampel_per_huruf = 10
juml_sampel = juml_kelas * juml_sampel_per_huruf


#MENCARI UKURAN ROW DAN COL DARI TIAP GAMBAR
i = 0
pic = plt.imread(os.path.join(dir_in, file0 + str(i) + ext))

if pic.dtype != np.uint8:
    pic = (pic * 255).astype(np.uint8)

row, col = pic.shape[0], pic.shape[1]
print("row, col =", row, ",", col, ".")

juml_kolom = row * col + 1
print("Jumlah kolom dataset =", juml_kolom)


#BUAT TEMPLATE UNTUK DATASET (.npy)
dataset = np.zeros(shape=(juml_sampel, juml_kolom), dtype=np.uint16)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ISI KOLOM PERTAMA DATASET DENGAN NOMOR URUT SAMPEL
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for k in range(0, juml_sampel):
    dataset[k, 0] = k


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#MEMASUKKAN NILAI PIKSEL TIAP GAMBAR KE TEMPLATE DATASET KOLOM KE-1 s.d. KOLOM TERAKHIR
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
temp = np.zeros(shape=(row, col), dtype=np.uint16)

k = 0
for i in range(0, juml_kelas):
    for n in range(0, juml_sampel_per_huruf):

        nama_file = globals()["file" + str(i)]

        path_gambar = os.path.join(dir_in, nama_file + str(n) + ext)

        pic = plt.imread(path_gambar)

        if pic.dtype != np.uint8:
            pic = (pic * 255).astype(np.uint8)

        #Jika gambar 3 channel, ambil channel pertama.
        #Jika grayscale 2D, langsung pakai gambar itu.
        if len(pic.shape) == 3:
            temp = pic[:, :, 0]
        else:
            temp = pic[:, :]

        data = temp.reshape(row * col)

        dataset[k, 1:juml_kolom] = data

        print("Baris", k, "diisi dari file:", nama_file + str(n) + ext)

        k += 1


print("")
print("Ukuran dataset:", dataset.shape)
print("Baris pertama:")
print(dataset[0, :])
print("Baris terakhir:")
print(dataset[juml_sampel - 1, :])


nama_file_dataset = "inputs_" + str(juml_sampel) + "_" + str(juml_kolom) + ".npy"

np.save(os.path.join(dir_out, nama_file_dataset), dataset)

print("")
print("Dataset telah di-create dan disimpan pada folder yang sama dengan program ini.")
print("Nama file dataset:", nama_file_dataset)


#CEK ISI DATASET DAN TAMPILKAN GAMBARNYA
plt.figure()

i = 0
while i < juml_sampel:
    isi = np.load(os.path.join(dir_out, nama_file_dataset))

    isi_baris = isi[i, 1:]
    isi_baris = isi_baris.reshape(row, col)

    gambar = np.zeros(shape=(row, col, 3), dtype=np.uint16)
    gambar[:, :, 0] = isi_baris
    gambar[:, :, 1] = isi_baris
    gambar[:, :, 2] = isi_baris

    plt.title(str(i))
    plt.imshow(gambar)
    plt.ion()
    plt.show()
    plt.pause(1)

    i += juml_sampel_per_huruf

    