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

dir_in = os.path.join(r"d:\tugas proyek img process\03_Randomize", mode_pooling)
dir = os.path.join(r"d:\tugas proyek img process\04_ANN_Training_Testing_Insample", mode_pooling)

nama_file_latih_input = "random_inputs_120_785.npy"
nama_file_latih_label = "random_labels_120_13.npy"

inp_row, inp_col = 28, 28

number_of_hidden_neurons = 20
epochs = 100
act = "sigmoid"


#DEFINING ACTIVATION FUNCTION
def sigmoid(a):
    hasil = 1 / (1 + np.exp(-a))
    return hasil


def ReLU(a):
    hasil = a
    e, f = a.shape

    for i in range(0, e):
        for j in range(0, f):
            if a[i, j] < 0:
                hasil[i, j] = 0

            if a[i, j] >= 0:
                hasil[i, j] = a[i, j]

    return hasil


#PREPARATION
if not os.path.exists(dir):
    os.makedirs(dir)

inputs_full = np.load(os.path.join(dir_in, nama_file_latih_input))
labels_full = np.load(os.path.join(dir_in, nama_file_latih_label))

inputs = inputs_full[:, 1:]
inputs_nomor_gambar_asal = inputs_full[:, 0]
labels = labels_full[:, 1:]

# Normalisasi supaya sigmoid tidak cepat jenuh.
inputs = inputs / 255.0

m, n = inputs.shape
o, p = labels.shape

print("Data latih inputs:", m, ",", n)
print("Data latih label:", o, ",", p)

if m != o:
    print("ERROR: Jumlah baris input dan label tidak sama.")
    exit()

if n != inp_row * inp_col:
    print("ERROR: Ukuran input tidak sesuai dengan inp_row dan inp_col.")
    print("Jumlah fitur input:", n)
    print("inp_row * inp_col:", inp_row * inp_col)
    exit()

i_n = n
o_n = p
h_n = number_of_hidden_neurons

print("Input neuron:", i_n)
print("Hidden neuron:", h_n)
print("Output neuron:", o_n)


w_i_h = np.random.uniform(-0.5, 0.5, (h_n, i_n))
w_h_o = np.random.uniform(-0.5, 0.5, (o_n, h_n))
b_i_h = np.zeros((h_n, 1))
b_h_o = np.zeros((o_n, 1))


#ANN TRAINING
learn_rate = 0.01

for epoch in range(1, epochs + 1):
    nr_correct = 0

    for inp, label in zip(inputs, labels):
        inp = inp.reshape(inp_row * inp_col, 1)
        label = label.reshape(o_n, 1)

        # Forward propagation input -> hidden
        h_pre = b_i_h + w_i_h @ inp

        if act == "ReLU":
            h = ReLU(h_pre)

        if act == "sigmoid":
            h = sigmoid(h_pre)

        # Forward propagation hidden -> output
        o_pre = b_h_o + w_h_o @ h

        if act == "ReLU":
            output = ReLU(o_pre)

        if act == "sigmoid":
            output = sigmoid(o_pre)

        nr_correct += int(np.argmax(output) == np.argmax(label))

        # Backpropagation output -> hidden
        delta_o = output - label
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        # Backpropagation hidden -> input
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(inp)
        b_i_h += -learn_rate * delta_h

    print(f"epoch = {epoch}, Accuracy: {round((nr_correct / inputs.shape[0]) * 100, 2)}%.")


#SAVING THE TRAINING RESULTS INTO THE HDD
np.save(os.path.join(dir, "a3_b_i_h.npy"), b_i_h)
np.save(os.path.join(dir, "a3_w_i_h.npy"), w_i_h)
np.save(os.path.join(dir, "a3_b_h_o.npy"), b_h_o)
np.save(os.path.join(dir, "a3_w_h_o.npy"), w_h_o)


#ANN TESTING
ZODIAC_CLASSES = ["Aquarius", "Aries", "Cancer", "Capricorn", "Gemini", "Leo", "Libra", "Pisces", "Sagitarius", "Scorpio", "Taurus", "Virgo"]

def konversi(array):
    posisi_dg_nilai_tertinggi = array.argmax()
    if 0 <= posisi_dg_nilai_tertinggi < len(ZODIAC_CLASSES):
        return ZODIAC_CLASSES[posisi_dg_nilai_tertinggi]
    return "Unknown"


def konversi_label_asli(array):
    posisi = array.argmax()
    if 0 <= posisi < len(ZODIAC_CLASSES):
        return ZODIAC_CLASSES[posisi]
    return "Unknown"


#READING THE TRAINING RESULTS FROM THE HDD
b_i_h = np.load(os.path.join(dir, "a3_b_i_h.npy"))
w_i_h = np.load(os.path.join(dir, "a3_w_i_h.npy"))
b_h_o = np.load(os.path.join(dir, "a3_b_h_o.npy"))
w_h_o = np.load(os.path.join(dir, "a3_w_h_o.npy"))


while True:
    print("")

    user_in = input("Pilih gambar yang ingin dikenali (0 - " + str(m - 1) + ", ketik -1 untuk keluar): ")
    if user_in.strip() == "-1" or user_in.strip() == "":
        print("Keluar dari program.")
        break
    index1 = int(user_in)

    if index1 < 0 or index1 >= m:
        print("Index di luar batas.")
        continue

    inp = inputs[index1]
    label_asli = labels[index1]

    plt.figure()
    plt.imshow(inp.reshape(inp_row, inp_col), cmap="Greys")

    print("Label di dataset adalah:", label_asli)
    print("Label asli:", konversi_label_asli(label_asli))

    h_pre = b_i_h + w_i_h @ inp.reshape(inp_row * inp_col, 1)

    if act == "ReLU":
        h = ReLU(h_pre)

    if act == "sigmoid":
        h = sigmoid(h_pre)

    o_pre = b_h_o + w_h_o @ h

    if act == "ReLU":
        output = ReLU(o_pre)

    if act == "sigmoid":
        output = sigmoid(o_pre)

    print("Output:", output)

    label_hasil = konversi(output)

    print("Label hasil:", label_hasil)

    plt.title("Klasifikasi menurut model ini: " + label_hasil)
    plt.show()