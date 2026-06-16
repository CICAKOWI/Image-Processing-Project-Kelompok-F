import numpy as np
import os

dir_in_inputs = r"C:\Users\rende\OneDrive\Dokumen\data kuliah\semester 6\3 image processing\pertemuan 15\M10-M14 Klasif dg ANN Huruf Hangul NAS\M10-M14 Klasif dg ANN Huruf Hangul NAS\02a_Create_Dataset"
dir_in_labels = r"C:\Users\rende\OneDrive\Dokumen\data kuliah\semester 6\3 image processing\pertemuan 15\M10-M14 Klasif dg ANN Huruf Hangul NAS\M10-M14 Klasif dg ANN Huruf Hangul NAS\02b_Create_Label"
dir = r"C:\Users\rende\OneDrive\Dokumen\data kuliah\semester 6\3 image processing\pertemuan 15\M10-M14 Klasif dg ANN Huruf Hangul NAS\M10-M14 Klasif dg ANN Huruf Hangul NAS\03_Randomize"

file_input = "inputs_200_401.npy"
file_label = "labels_200_11.npy"

inputs = np.load(os.path.join(dir_in_inputs, file_input))   #Please type the file name accordingly.
labels = np.load(os.path.join(dir_in_labels, file_label))       #Please type the file name accordingly.
m, n = np.shape(inputs)
r, s = np.shape(labels)
print(np.shape(inputs))
print(np.shape(labels))

#Creating seeds for randomization.
a = []
for i in range(0, m):
    a.append(i)
print(np.shape(a))
print(a[0])
print(a[m-1])
np.random.shuffle(a)
print(np.shape(a))
print(a[0])
print(a[m-1])

#Now randomize inputs and labels using the seeds.
inputs_random = np.zeros(shape = (m, n), dtype = float)   #Template for the randomizes dataset (inputs).
labels_random = np.zeros(shape = (r, s), dtype = float)   #Template for the randomizes dataset (labels).

for i in range(0,m):
    inputs_random[i,:] = inputs[a[i],:]            #Realizing the randomization (inputs).
    labels_random[i,:] = labels[a[i],:]            #Realizing the randomization (labels).

np.save(os.path.join(dir, "random_" + file_input), inputs_random)
np.save(os.path.join(dir, "random_" + file_label), labels_random)
