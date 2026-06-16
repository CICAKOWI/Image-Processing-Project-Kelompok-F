import numpy as np
import matplotlib.pyplot as plt
import os

#USER ENTRY
dir_in  = r"F:\tugas proyek img process\00_Dataset_Mentah"
dir_out = r"F:\tugas proyek img process\01_ImageProcessing"

file_input = []
file_input.append("aksun_ca")
file_input.append("aksun_ga")
file_input.append("aksun_ja")
file_input.append("aksun_ka")
file_input.append("aksun_nga")
file_input.append("nul")

j = 20
ext = ".jpg"

m, n = 28, 28
mode_pooling = "max"

th_manual = 90
padding_ratio = 0.25
min_neighbor = 2
save_debug = True


#FUNCTION / LIBRARY
def baca_gambar(dir, file_input, ext):
    pic = plt.imread(os.path.join(dir, file_input + ext))

    if pic.dtype != np.uint8:
        pic = (pic * 255).astype(np.uint8)

    if len(pic.shape) == 2:
        temp = np.zeros(shape=(pic.shape[0], pic.shape[1], 3), dtype=np.uint8)
        temp[:, :, 0] = pic
        temp[:, :, 1] = pic
        temp[:, :, 2] = pic
        pic = temp

    if pic.shape[2] > 3:
        pic = pic[:, :, 0:3]

    return pic


def grayscale(pic):
    row, col = pic.shape[0], pic.shape[1]
    pic_gs = np.zeros(shape=(row, col), dtype=np.uint8)

    r = pic[:, :, 0].astype(np.float32)
    g = pic[:, :, 1].astype(np.float32)
    b = pic[:, :, 2].astype(np.float32)

    average = (r + g + b) / 3
    pic_gs[:, :] = average[:, :]

    return pic_gs


def otsu_threshold(pic_gs):
    hist = np.zeros(shape=(256), dtype=np.float64)

    row, col = pic_gs.shape
    for i in range(0, row):
        for j in range(0, col):
            nilai = pic_gs[i, j]
            hist[nilai] += 1

    total = row * col
    sum_total = 0

    for i in range(0, 256):
        sum_total += i * hist[i]

    sum_background = 0
    weight_background = 0
    var_max = -1
    th = 127

    for i in range(0, 256):
        weight_background += hist[i]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += i * hist[i]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        var_between = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)

        if var_between > var_max:
            var_max = var_between
            th = i

    return th


def binarisasi(pic_gs):
    th_otsu = otsu_threshold(pic_gs)

    if th_otsu > th_manual:
        th = th_manual
    else:
        th = th_otsu

    row, col = pic_gs.shape
    pic_bin = np.zeros(shape=(row, col), dtype=np.uint8)

    for i in range(0, row):
        for j in range(0, col):
            if pic_gs[i, j] < th:
                pic_bin[i, j] = 255
            else:
                pic_bin[i, j] = 0

    return pic_bin, th


def hapus_noise_titik(pic_bin):
    row, col = pic_bin.shape
    pic_clean = np.zeros(shape=(row, col), dtype=np.uint8)

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if pic_bin[i, j] == 255:
                counter = 0

                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        if pic_bin[k, l] == 255:
                            counter += 1

                if counter >= min_neighbor:
                    pic_clean[i, j] = 255
                else:
                    pic_clean[i, j] = 0

    return pic_clean


def crop_objek(pic_bin):
    row, col = pic_bin.shape

    row_sum = np.zeros(shape=(row), dtype=np.uint16)
    col_sum = np.zeros(shape=(col), dtype=np.uint16)

    for i in range(0, row):
        counter = 0
        for j in range(0, col):
            if pic_bin[i, j] == 255:
                counter += 1
        row_sum[i] = counter

    for j in range(0, col):
        counter = 0
        for i in range(0, row):
            if pic_bin[i, j] == 255:
                counter += 1
        col_sum[j] = counter

    row_isi = []
    col_isi = []

    batas_row = 2
    batas_col = 2

    for i in range(0, row):
        if row_sum[i] >= batas_row:
            row_isi.append(i)

    for j in range(0, col):
        if col_sum[j] >= batas_col:
            col_isi.append(j)

    if len(row_isi) == 0 or len(col_isi) == 0:
        return pic_bin

    r_min = min(row_isi)
    r_max = max(row_isi)
    c_min = min(col_isi)
    c_max = max(col_isi)

    tinggi = r_max - r_min + 1
    lebar = c_max - c_min + 1

    pad_r = int(padding_ratio * tinggi)
    pad_c = int(padding_ratio * lebar)

    r_min = max(0, r_min - pad_r)
    r_max = min(row - 1, r_max + pad_r)
    c_min = max(0, c_min - pad_c)
    c_max = min(col - 1, c_max + pad_c)

    pic_crop = pic_bin[r_min:r_max + 1, c_min:c_max + 1]

    return pic_crop


def buat_kanvas_persegi(pic_crop):
    row, col = pic_crop.shape
    ukuran = max(row, col)

    canvas = np.zeros(shape=(ukuran, ukuran), dtype=np.uint8)

    mulai_row = int((ukuran - row) / 2)
    mulai_col = int((ukuran - col) / 2)

    canvas[mulai_row:mulai_row + row, mulai_col:mulai_col + col] = pic_crop

    return canvas


def down_pooling(pic, m, n, mode_pooling):
    row_asal, col_asal = pic.shape
    row_hasil, col_hasil = m, n

    pic_res = np.zeros(shape=(row_hasil, col_hasil), dtype=np.float32)

    for i_res in range(0, row_hasil):
        for j_res in range(0, col_hasil):
            r_start = int(i_res * row_asal / row_hasil)
            r_end = int((i_res + 1) * row_asal / row_hasil)

            c_start = int(j_res * col_asal / col_hasil)
            c_end = int((j_res + 1) * col_asal / col_hasil)

            if r_end <= r_start:
                r_end = r_start + 1

            if c_end <= c_start:
                c_end = c_start + 1

            if mode_pooling == "average":
                counter = 0
                jumlah = 0

                for k in range(r_start, r_end):
                    for l in range(c_start, c_end):
                        jumlah += int(pic[k, l])
                        counter += 1

                pic_res[i_res, j_res] = jumlah / counter

            if mode_pooling == "max":
                nilai_max = 0

                for k in range(r_start, r_end):
                    for l in range(c_start, c_end):
                        if pic[k, l] > nilai_max:
                            nilai_max = pic[k, l]

                pic_res[i_res, j_res] = nilai_max

    pic_final = np.zeros(shape=(row_hasil, col_hasil), dtype=np.uint8)

    for i in range(0, row_hasil):
        for j in range(0, col_hasil):
            if pic_res[i, j] > 0:
                pic_final[i, j] = 255
            else:
                pic_final[i, j] = 0

    return pic_final


def ubah_ke_rgb(pic_gs):
    row, col = pic_gs.shape
    pic_rgb = np.zeros(shape=(row, col, 3), dtype=np.uint8)

    pic_rgb[:, :, 0] = pic_gs
    pic_rgb[:, :, 1] = pic_gs
    pic_rgb[:, :, 2] = pic_gs

    return pic_rgb


def simpan_debug(dir_out, file_input, pic_ori, pic_gs, pic_bin, pic_clean, pic_crop, pic_square, pic_ready):
    dir_debug = os.path.join(dir_out, "_debug")

    if not os.path.exists(dir_debug):
        os.makedirs(dir_debug)

    plt.figure(figsize=(14, 4))

    plt.subplot(1, 7, 1)
    plt.title("Original")
    plt.imshow(pic_ori)
    plt.axis("off")

    plt.subplot(1, 7, 2)
    plt.title("Grayscale")
    plt.imshow(pic_gs, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 7, 3)
    plt.title("Binary")
    plt.imshow(pic_bin, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 7, 4)
    plt.title("Clean")
    plt.imshow(pic_clean, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 7, 5)
    plt.title("Crop")
    plt.imshow(pic_crop, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 7, 6)
    plt.title("Square")
    plt.imshow(pic_square, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 7, 7)
    plt.title("Ready")
    plt.imshow(pic_ready, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(dir_debug, file_input + "_debug.jpg"), dpi=150)
    plt.close()


def preprocessing(dir_in, dir_out, file_input, ext):
    pic_ori = baca_gambar(dir_in, file_input, ext)

    pic_gs = grayscale(pic_ori)

    pic_bin, th = binarisasi(pic_gs)

    pic_clean = hapus_noise_titik(pic_bin)

    pic_crop = crop_objek(pic_clean)

    pic_square = buat_kanvas_persegi(pic_crop)

    pic_ready = down_pooling(pic_square, m, n, mode_pooling)

    pic_ready_rgb = ubah_ke_rgb(pic_ready)

    plt.imsave(os.path.join(dir_out, file_input + "_ready.jpg"), pic_ready_rgb)

    if save_debug == True:
        simpan_debug(dir_out, file_input, pic_ori, pic_gs, pic_bin, pic_clean, pic_crop, pic_square, pic_ready)

    print(file_input, "selesai diproses. Threshold =", th)

    return pic_ori, pic_gs, pic_bin, pic_clean, pic_crop, pic_square, pic_ready


#MAIN PROGRAM
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

i = 0
while file_input[i] != "nul":
    for k in range(0, j):
        nama_file = file_input[i] + " (" + str(k) + ")"
        preprocessing(dir_in, dir_out, nama_file, ext)
    i += 1

print("")
print("Semua gambar selesai diproses.")
print("Hasil ready tersimpan di:", dir_out)
print("Hasil debug tersimpan di:", os.path.join(dir_out, "_debug"))