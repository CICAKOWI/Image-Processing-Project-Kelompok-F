# 04_ANN_Training_Testing_Insample — Pelatihan & Pengujian JST

## Tujuan
Melatih Artificial Neural Network (ANN) dari nol (tanpa framework) untuk mengenali Aksara Sunda.

## Arsitektur
- Input: 400 neuron (20×20 pixel)
- Hidden: 20 neuron (sigmoid/ReLU)
- Output: 10 neuron (sigmoid)
- Backpropagation manual, MSE loss, SGD, learning rate 0.001

## Alur
1. Memuat `random_inputs_*.npy` dan `random_labels_*.npy` dari `03_Randomize/`
2. Training 100 epoch
3. Simpan bobot ke file `.npy`
4. Mode interaktif: user masukkan index sampel → ANN tampilkan prediksi

## Script
| File | Fungsi |
|------|--------|
| `01_ann_huruf_training_testing_insample.py` | Training + testing interaktif |

## Output
- `a3_w_i_h.npy`, `a3_b_i_h.npy` — bobot & bias input-hidden
- `a3_w_h_o.npy`, `a3_b_h_o.npy` — bobot & bias hidden-output
