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

dir_in_inputs = os.path.join(r"d:\tugas proyek img process\02a_Create_Dataset", mode_pooling)
dir_in_labels = os.path.join(r"d:\tugas proyek img process\02b_Create_Label", mode_pooling)
dir = os.path.join(r"d:\tugas proyek img process\03_Randomize", mode_pooling)

file_input = "inputs_120_785.npy"
file_label = "labels_120_13.npy"

if not os.path.exists(dir):
    os.makedirs(dir)

inputs = np.load(os.path.join(dir_in_inputs, file_input))
labels = np.load(os.path.join(dir_in_labels, file_label))

m, n = np.shape(inputs)
r, s = np.shape(labels)

print("Ukuran inputs:", np.shape(inputs))
print("Ukuran labels:", np.shape(labels))

if m != r:
    print("ERROR: Jumlah baris inputs dan labels tidak sama.")
    print("Baris inputs:", m)
    print("Baris labels:", r)
    exit()

#Creating seeds for randomization.
a = []

for i in range(0, m):
    a.append(i)

print("Jumlah indeks sebelum acak:", np.shape(a))
print("Indeks pertama sebelum acak:", a[0])
print("Indeks terakhir sebelum acak:", a[m - 1])

np.random.shuffle(a)

print("Jumlah indeks setelah acak:", np.shape(a))
print("Indeks pertama setelah acak:", a[0])
print("Indeks terakhir setelah acak:", a[m - 1])

#Now randomize inputs and labels using the same seeds.
inputs_random = np.zeros(shape=(m, n), dtype=float)
labels_random = np.zeros(shape=(r, s), dtype=float)

for i in range(0, m):
    inputs_random[i, :] = inputs[a[i], :]
    labels_random[i, :] = labels[a[i], :]

np.save(os.path.join(dir, "random_" + file_input), inputs_random)
np.save(os.path.join(dir, "random_" + file_label), labels_random)

print("")
print("Randomisasi selesai.")
print("File inputs random :", "random_" + file_input)
print("File labels random :", "random_" + file_label)
print("Folder output      :", dir)