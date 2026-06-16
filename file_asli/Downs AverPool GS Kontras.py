import numpy as np
import matplotlib.pyplot as plt
import os

#USER ENTRY
dir = r"F:\tugas proyek img process\00_Dataset_Mentah\hasil"
file_input = []
file_input.append ("aksun_ca")
file_input.append ("hangul_a")
file_input.append ("hangul_eo")
file_input.append ("hangul_eu")
file_input.append ("hangul_i")
file_input.append ("hangul_o")
file_input.append ("hangul_u")
file_input.append ("hangul_ya")
file_input.append ("hangul_yeo")
file_input.append ("hangul_yo")
file_input.append ("hangul_yu")
file_input.append ("nul")         #Nilai "nul" kita gunakan sebagai tanda berhenti pada iterasi pencarian nama file.
ext = ".jpg"
j = 5                             #banyaknya sampel per huruf

row_crop = 0.05                    #5% of rows at left and right will be cut.
col_crop = 0.05                    #5% of rows at top and bottom will be cut.
m, n, ch = 20, 20, 3                       #Ukuran gambar setelah didownsize.
th = 105

#FUNCTION / LIBRARY
def preprocessing (dir, file_input, ext, th):
    #PREPARATION
    pic_ori = plt.imread(os.path.join(dir, file_input + ext))                   #np.uint8.
    row, col  = np.shape(pic_ori)[0], np.shape(pic_ori)[1]

    #CROP (POTONG BAGIAN PINGGIR GAMBAR)
    row_margin = round(row_crop * row)
    col_margin = round(col_crop * col)
    pic_crop = pic_ori[row_margin:row-row_margin, col_margin:col-col_margin, :]  #np.uint8

    #ALGORITMA UNTUK DOWNSIZE DENGAN AVERAGE POOLING
    #Ukuran Asal
    pic_asal = pic_crop.astype(np.float16)  #Karena kita akan melakukan operasi dg pecahan.
    row_asal, col_asal, ch_asal = pic_asal.shape
    #print(row_asal, col_asal, ch_asal)
    #Ukuran Hasil
    row_hasil, col_hasil, ch_hasil = m, n, ch
    #print(row_hasil, col_hasil, ch_hasil)
    #Hitung Faktor Resize
    faktor_row = row_hasil/row_asal
    faktor_col = col_hasil/col_asal

    #Siapkan Template untuk Hasil
    pic_res = np.zeros(shape=(row_hasil, col_hasil, ch_hasil), dtype=np.float16)
    #Ukuran Pooling Region
    delta_row = round(row_asal/m)
    delta_col = round(col_asal/n)

    #Downsizing Menggunakan Average Pooling
    i_res = -1
    for i_asal in range(0, row_asal-delta_row, delta_row):  #The step leaps every delta-row.
        i_res += 1
        j_res = -1
        for j_asal in range(0, col_asal-delta_col, delta_col): #The step leaps every delta-col.
            j_res += 1
            counter = 0
            sum_r, sum_g, sum_b = 0, 0, 0                     #Tampungan untuk sum r, g, dan b.
            for k in range (i_asal, i_asal+delta_row):
                for l in range (j_asal, j_asal+delta_col):
                    counter += 1
                    sum_r = sum_r + pic_asal[k, l, 0]
                    sum_g = sum_g + pic_asal[k, l, 1]
                    sum_b = sum_b + pic_asal[k, l, 2]
                    #print(i_res, j_res, counter, sum_r, sum_g, sum_b)
            pic_res[i_res, j_res, 0] = sum_r/counter
            pic_res[i_res, j_res, 1] = sum_g/counter
            pic_res[i_res, j_res, 2] = sum_b/counter

    #Buat gambar grayscale.
    pic_gs = 1 * pic_res                                                             #np.float16
    average = (pic_res[:, :, 0] + pic_res[:, :, 1] + pic_res[:, :, 2]) / 3
    pic_gs[:, :, 0] = average
    pic_gs[:, :, 1] = average
    pic_gs[:, :, 2] = average

    #Ubah latar menjadi hitam dan objek menjadi putih.
    pic_gs_hitam = 1 * pic_gs
    pic_gs_hitam[:, :, :] =  255 - pic_gs[:, :, :]

    #Maksimalkan Kontras untuk Gambar Grayscale
    pic_gs_hitam_kontras = 1 * pic_gs_hitam
    row, col = np.shape(pic_gs_hitam_kontras)[0], np.shape(pic_gs_hitam_kontras)[1]
    for i in range (0, row):
        for j in range(0, col):
            if pic_gs_hitam_kontras[i, j, 0] < th:
                pic_gs_hitam_kontras[i, j, 0] = 0
                pic_gs_hitam_kontras[i, j, 1] = 0
                pic_gs_hitam_kontras[i, j, 2] = 0
            if pic_gs_hitam_kontras[i, j, 0] >= th:
                pic_gs_hitam_kontras[i, j, 0] = 255
                pic_gs_hitam_kontras[i, j, 1] = 255
                pic_gs_hitam_kontras[i, j, 2] = 255

    return pic_ori, pic_crop, pic_res, pic_gs, pic_gs_hitam, pic_gs_hitam_kontras

#MAIN PROGRAM
i = 0
while file_input[i] != "nul":                                             #Jika bertemu nilai "nul" iterasi berhenti.
    for k in range(0, j):
        file_gambar = file_input[i] + " (" + str(k) + ")"
        print(file_gambar)
        pic_ori, pic_crop, pic_res, pic_gs, pic_gs_hitam, pic_gs_hitam_kontras = preprocessing(dir, file_gambar, ext, th)
        pic_gs_hitam_kontras = pic_gs_hitam_kontras.astype(np.uint8)
        plt.imsave(os.path.join(dir, file_gambar + "_ready" + ext), pic_gs_hitam_kontras)
    i += 1

plt.figure("pic_ori")
plt.imshow(pic_ori)
plt.figure("pic_crop")
plt.imshow(pic_crop)
plt.figure("pic_res")
plt.imshow(pic_res.astype(np.uint8))
plt.figure("pic_gs")
plt.imshow(pic_gs.astype(np.uint8))
plt.figure("pic_gs_hitam")
plt.imshow(pic_gs_hitam.astype(np.uint8))
plt.figure("pic_gs_hitam_kontras")
plt.imshow(pic_gs_hitam_kontras)
plt.show()
