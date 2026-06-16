"""
CATATAN TENTANG DATASET INPUT YANG DIBANGUN INI
Kolom ke-0 berisi nomor sampel.
Kolom ke-1 s.d. ke-200 berisi nilai piksel dari gambar input.
Baris-baris dataset harus sudah diacak. Jika dataset diatur dengan urutan yang tertib,justru akan
menghasilkan prediksi yang salah.
Pada ekperimen ini, input tidak dinormalisasi (0-255). Sedangkan label dibuat dengan nilai
  normalisasi, yaitu 0 s.d. 1. Ternyata ANN tetap bekerja.
Dataset yang terdiri dari 200 baris berikut ini digunakan untuk melatih ANN.
Setelah ANN dilatih, dibutuhkan 1 data saja untuk pengujian. Data ini bisa 
diambilkan dari dataset (insample) ataupun dari luar dataset (outsample)."""

import numpy as np
import matplotlib.pyplot as plt
import os

#USER ENTRIES
dir_in = r"C:\Users\rende\OneDrive\Dokumen\data kuliah\semester 6\3 image processing\pertemuan 15\M10-M14 Klasif dg ANN Huruf Hangul NAS\M10-M14 Klasif dg ANN Huruf Hangul NAS\03_Randomize"
dir = r"C:\Users\rende\OneDrive\Dokumen\data kuliah\semester 6\3 image processing\pertemuan 15\M10-M14 Klasif dg ANN Huruf Hangul NAS\M10-M14 Klasif dg ANN Huruf Hangul NAS\04_ANN_Training_Testing_Insample"
nama_file_latih_input = "random_inputs_200_401.npy" #Kolom ke-0 berisi nomor gambar asal, jangan diproses.
nama_file_latih_label = "random_labels_200_11.npy" #Kolom ke-0 berisi nomor gambar asal, jangan diproses.
inp_row, inp_col = 20, 20                             #Ukuran asli data input di dataset tanpa kolom ke-0.
number_of_hidden_neurons = 20
epochs = 100
act = "sigmoid"                                   #ReLU                    #Please choose ReLU or sigmoid.

#DEFINING ACTIVATION FUNCTION; dalam model ANN ini a berupa array 2 dimensi.
def sigmoid(a):
    hasil = 1 / (1 + np.exp(-a))
    return hasil
def ReLU(a):                                                  #Rectified Linear Unit
    hasil = a               #Nilai awal untuk hasil agar ukurannya sama dengan nilai.
    e, f = a.shape
    for i in range(0, e):
        for j in range(0, f):
            if a[i,j] < 0 : hasil[i,j] = 0
            if a[i,j] >= 0: hasil[i,j] = a[i,j]
    return hasil

#PREPARATION
inputs = np.load(os.path.join(dir_in, nama_file_latih_input))[:, 1:] #Kolom ke-0 berisi nomor gambar asal, jangan diproses.
inputs_nomor_gambar_asal = np.load(os.path.join(dir_in, nama_file_latih_input))[:, 0]
labels = np.load(os.path.join(dir_in, nama_file_latih_label))[:, 1:] #Kolom ke-0 berisi nomor gambar asal, jangan diproses.

m, n = inputs.shape; print("Data latih inputs:", m,",", n) #Atau m, n = np.shape(inputs).
o, p = labels.shape; print("Data latih label:", o, "'", p) #Atau 0, p = np.shape(labels).
i_n = n                                                #the number of input neurons
o_n = p                                                #the number of output neurons
h_n = number_of_hidden_neurons                         #the number of hidden neurons
print('Hidden neuron:', h_n)

w_i_h = np.random.uniform(-0.5, 0.5, (h_n, i_n))   #weights input - hidden
w_h_o = np.random.uniform(-0.5, 0.5, (o_n, h_n))  #weights hidden - output
b_i_h = np.zeros((h_n, 1))                                           #bias input - hidden
b_h_o = np.zeros((o_n, 1))                                           #bias hidden - output

#ANN TRAINING
learn_rate = 0.001
nr_correct = 0
for epoch in range(1, epochs+1):
    for inp, label in zip(inputs, labels):
        inp.shape += (1,) #Untuk menambahkan dimensi pada array ini agar cocok dengan array lain.
        label.shape += (1,)   #Untuk menambahkan dimensi pada array ini agar cocok dengan array lain.

        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ inp
        if act == "ReLU": h = ReLU(h_pre)
        if act == "sigmoid": h = sigmoid(h_pre)

        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h
        if act == "ReLU": o = ReLU(o_pre)
        if act == "sigmoid": o = sigmoid(o_pre)

        # Cost / Error calculation
        e = 1 / len(o) * np.sum((o - label) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(label))

        # Backpropagation output -> hidden (cost function derivative)
        delta_o = o - label
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backpropagation hidden -> input (activation function derivative)
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(inp)
        b_i_h += -learn_rate * delta_h

    # Show accuracy for this epoch
    print(f"epoch = {epoch}, " + f"Accuracy: {round((nr_correct / inputs.shape[0]) * 100, 2)}%.")
    nr_correct = 0

#SAVING THE TRAINING RESULTS INTO THE HDD
np.save(os.path.join(dir, "a3_b_i_h.npy"), b_i_h)
np.save(os.path.join(dir, "a3_w_i_h.npy"), w_i_h)
np.save(os.path.join(dir, "a3_b_h_o.npy"), b_h_o)
np.save(os.path.join(dir, "a3_w_h_o.npy"), w_h_o)

#ANN TESTING
def konversi(array):
    posisi_dg_nilai_tertinggi = array.argmax()
    if posisi_dg_nilai_tertinggi == 0: label_hasil = "hangul_a"
    if posisi_dg_nilai_tertinggi == 1: label_hasil = "hangul_eo"
    if posisi_dg_nilai_tertinggi == 2: label_hasil = "hangul_eu"
    if posisi_dg_nilai_tertinggi == 3: label_hasil = "hangul_i"
    if posisi_dg_nilai_tertinggi == 4: label_hasil = "hangul_o"
    if posisi_dg_nilai_tertinggi == 5: label_hasil = "hangul_u"
    if posisi_dg_nilai_tertinggi == 6: label_hasil = "hangul_ya"
    if posisi_dg_nilai_tertinggi == 7: label_hasil = "hangul_yeo"
    if posisi_dg_nilai_tertinggi == 8: label_hasil = "hangul_yo"
    if posisi_dg_nilai_tertinggi == 9: label_hasil = "hangul_yu"
    return label_hasil

#READING THE TRAINING RESULTS FROM THE HDD
b_i_h = np.load(os.path.join(dir, "a3_b_i_h.npy"))
w_i_h = np.load(os.path.join(dir, "a3_w_i_h.npy"))
b_h_o = np.load(os.path.join(dir, "a3_b_h_o.npy"))
w_h_o = np.load(os.path.join(dir, "a3_w_h_o.npy"))

while True:
    print("")
    # Pilih satu gambar pisang yang ingin dikenali kelasnya.
    index1 = int(input("Pilih gambar yang ingin dikenali (0 - " + str(m-1) + "): "))  #Insample sudah ternormalisasi

    # PREDIKSI / KLASIFIKASI DIMULAI
    #Ambil satu baris data dari dataset dan dari label sesuai pilihan pengguna dan tampilkan di layar.
    inp = inputs[index1]
    #inp.shape += (1,)
    plt.figure()
    plt.imshow(inp.reshape(inp_row, inp_col), cmap="Greys") #Reshape berguna jika data uji berupa gambar (punya banyak baris).
    print("Label di dataset adalah:", labels[index1])

    # Forward propagation input -> hidden
    h_pre = b_i_h + w_i_h @ inp.reshape(inp_row*inp_col, 1) #Reshape berguna jika data uji berupa gambar (punya banyak baris).
    if act == "ReLU": h = ReLU(h_pre)
    if act == "sigmoid": h = sigmoid(h_pre)

    # Forward propagation hidden -> output
    o_pre = b_h_o + w_h_o @ h
    if act == "ReLU": o = ReLU(o_pre)
    if act == "sigmoid": o = sigmoid(o_pre)
    o = sigmoid(o_pre)
    output = o
    print("Output:", output)

    #Hasil klasifikasi akan dimunculkan sebagai judul gambar
    label_hasil = konversi(output)
    print("Label hasil:", label_hasil)
    plt.title(f"Klasifikasi menurut model ini:" + label_hasil + ".")
    plt.show()

    """
    # Dengan index 1 kita cari nomor gambar asal dengan mapping di inputs_nomor_gambar_asal
    index2 = int(inputs_nomor_gambar_asal[index1])
    print("Data uji adalah gambar di dataset asal yang bernomor:", str(index2))
    plt.imshow(plt.imread("pisang ori (" + str(index2) + ").jpg")) #Menampilkan gambar uji sesuai index yang dipilih di atas.
    """
