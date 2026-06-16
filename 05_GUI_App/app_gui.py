import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_INPUT = os.path.join(CURRENT_DIR, "input")
PATH_OUTPUT = os.path.join(CURRENT_DIR, "output")
PATH_OUT_GS = os.path.join(PATH_OUTPUT, "grayscale")
PATH_OUT_BIN = os.path.join(PATH_OUTPUT, "binarization")
PATH_OUT_RESIZE_AVG = os.path.join(PATH_OUTPUT, "resizing", "average")
PATH_OUT_RESIZE_MAX = os.path.join(PATH_OUTPUT, "resizing", "max")
PATH_OUT_READY_AVG = os.path.join(PATH_OUTPUT, "ready", "average")
PATH_OUT_READY_MAX = os.path.join(PATH_OUTPUT, "ready", "max")
PATH_DATASET = os.path.join(CURRENT_DIR, "dataset")
PATH_ANN = os.path.join(CURRENT_DIR, "ann")
PATH_PREVIEW = os.path.join(CURRENT_DIR, "preview")
PATH_LOG = os.path.join(CURRENT_DIR, "log")
PATH_SAME_FOLDER = os.path.join(PATH_OUTPUT, "same_folder")


KELAS = ["Aquarius", "Aries", "Cancer", "Capricorn", "Gemini", "Leo", "Libra", "Pisces", "Sagitarius", "Scorpio", "Taurus", "Virgo"]
JUMLAH_SAMPEL_PER_KELAS = 20

# Colors for Space/Constellation Theme
COLOR_BG_SPACE = "#080C14"       # Deep Space Black
COLOR_CARD = "#121A2B"           # Nebula Blue Card
COLOR_BORDER = "#1E2A45"         # Constellation Line Gray
COLOR_TEXT_PRIMARY = "#E2E8F0"   # Star Light Gray
COLOR_TEXT_MUTED = "#94A3B8"     # Space Dust Gray
COLOR_GOLD = "#FBBF24"           # Star Gold Accent
COLOR_CYAN = "#38BDF8"           # Icy Cyan Accent
COLOR_GREEN = "#10B981"          # Nebula Green Accent
COLOR_RED = "#F87171"            # Red Dwarf Accent
COLOR_BUTTON_HOVER = "#1E293B"   # Dark Gray Hover

def create_required_folders():
    paths = [
        PATH_INPUT, PATH_OUT_GS, PATH_OUT_BIN, 
        PATH_OUT_RESIZE_AVG, PATH_OUT_RESIZE_MAX, 
        PATH_OUT_READY_AVG, PATH_OUT_READY_MAX,
        PATH_DATASET, PATH_ANN, PATH_PREVIEW, PATH_LOG,
        PATH_SAME_FOLDER
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ====================================================================
# IMAGE PROCESSING ALGORITHMS
# ====================================================================
def baca_gambar(dir_in, file_input):
    path = os.path.join(dir_in, file_input)
    if not os.path.exists(path):
        return None
    pic = plt.imread(path)
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
    hist, _ = np.histogram(pic_gs, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = pic_gs.size
    sum_total = np.dot(np.arange(256), hist)
    sum_background = 0.0
    weight_background = 0.0
    var_max = -1.0
    th = 127
    for i in range(0, 256):
        weight_background += hist[i]
        if weight_background == 0: continue
        weight_foreground = total - weight_background
        if weight_foreground == 0: break
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        var_between = weight_background * weight_foreground * ((mean_background - mean_foreground) ** 2)
        if var_between > var_max:
            var_max = var_between
            th = i
    return th

def binarisasi(pic_gs, th_manual=90):
    th_otsu = otsu_threshold(pic_gs)
    th = th_manual if th_otsu > th_manual else th_otsu
    pic_bin = np.where(pic_gs < th, 255, 0).astype(np.uint8)
    return pic_bin, th

def hapus_noise_titik(pic_bin, min_neighbor=2):
    row, col = pic_bin.shape
    # Pad to handle borders correctly matching the loop limits [1, row-2], [1, col-2]
    padded = np.pad(pic_bin == 255, 1, mode='constant')
    # Sum up 3x3 neighborhoods
    neighbor_count = np.zeros_like(pic_bin, dtype=np.int32)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            neighbor_count += padded[1+di : 1+di+row, 1+dj : 1+dj+col]
    # Keep only pixels that are foreground (255) and have at least min_neighbor neighbors
    pic_clean = np.where((pic_bin == 255) & (neighbor_count >= min_neighbor), 255, 0).astype(np.uint8)
    # Clear borders since the original loop only ran from 1 to row-2 and 1 to col-2
    pic_clean[0, :] = 0
    pic_clean[-1, :] = 0
    pic_clean[:, 0] = 0
    pic_clean[:, -1] = 0
    return pic_clean

def crop_objek(pic_bin, padding_ratio=0.25):
    row, col = pic_bin.shape
    row_sum = np.sum(pic_bin == 255, axis=1)
    col_sum = np.sum(pic_bin == 255, axis=0)
    batas_row = 2
    batas_col = 2
    row_isi = np.where(row_sum >= batas_row)[0]
    col_isi = np.where(col_sum >= batas_col)[0]
    if len(row_isi) == 0 or len(col_isi) == 0:
        return pic_bin
    r_min = row_isi[0]
    r_max = row_isi[-1]
    c_min = col_isi[0]
    c_max = col_isi[-1]
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
            if r_end <= r_start: r_end = r_start + 1
            if c_end <= c_start: c_end = c_start + 1
            block = pic[r_start:r_end, c_start:c_end]
            if mode_pooling == "average":
                pic_res[i_res, j_res] = np.mean(block)
            elif mode_pooling == "max":
                pic_res[i_res, j_res] = np.max(block)
    pic_final = np.where(pic_res > 90, 255, 0).astype(np.uint8)
    return pic_final, pic_res

def ubah_ke_rgb(pic_gs):
    row, col = pic_gs.shape
    pic_rgb = np.zeros(shape=(row, col, 3), dtype=np.uint8)
    pic_rgb[:, :, 0] = pic_gs
    pic_rgb[:, :, 1] = pic_gs
    pic_rgb[:, :, 2] = pic_gs
    return pic_rgb

# ====================================================================
# ANN MATHEMATICS
# ====================================================================
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# ====================================================================
# GUI INTERFACE
# ====================================================================
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("COSMOS: Zodiac Preprocessing & ANN Constellation Classifier")
        self.root.geometry("1200x820")
        self.root.configure(bg=COLOR_BG_SPACE)
        
        # State Variables
        self.selected_step = tk.IntVar(value=1)
        self.pooling_mode = tk.StringVar(value="average")
        self.row_size = tk.IntVar(value=28)
        self.col_size = tk.IntVar(value=28)
        
        self.preview_class = tk.StringVar(value=KELAS[0])
        self.preview_index = tk.IntVar(value=0)
        self.preview_stage = tk.StringVar(value="Original")
        
        # ANN test properties
        self.test_index = tk.IntVar(value=0)
        self.last_test_img = None
        self.last_test_true = ""
        self.last_test_pred = ""
        self.last_test_probs = None
        self.custom_original_img = None
        
        # ANN Training Curve History
        self.loss_history = []
        self.accuracy_history = []
        
        create_required_folders()
        self.setup_styles()
        self.build_layout()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Custom styles for dark space mode
        style.configure(".", bg=COLOR_BG_SPACE, fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 10))
        style.configure("TFrame", background=COLOR_BG_SPACE)
        style.configure("Card.TFrame", background=COLOR_CARD, borderwidth=1, relief="solid", bordercolor=COLOR_BORDER)
        
        style.configure("TLabel", background=COLOR_BG_SPACE, foreground=COLOR_TEXT_PRIMARY)
        style.configure("CardTitle.TLabel", background=COLOR_CARD, foreground=COLOR_GOLD, font=("Segoe UI", 11, "bold"))
        style.configure("Sub.TLabel", background=COLOR_BG_SPACE, foreground=COLOR_TEXT_MUTED, font=("Segoe UI", 9))
        
        style.configure("TCombobox", fieldbackground=COLOR_CARD, background=COLOR_BORDER, foreground=COLOR_TEXT_PRIMARY)
        style.map("TCombobox", fieldbackground=[("readonly", COLOR_CARD)], foreground=[("readonly", COLOR_TEXT_PRIMARY)])
        
        style.configure("TRadiobutton", background=COLOR_CARD, foreground=COLOR_TEXT_PRIMARY)
        style.map("TRadiobutton", foreground=[("selected", COLOR_CYAN)])

    def make_flat_button(self, parent, text, command, bg="#1E293B", fg=COLOR_TEXT_PRIMARY, hover_bg="#334155", font=("Segoe UI", 10, "bold")):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            font=font,
            cursor="hand2",
            padx=15,
            pady=8
        )
        def on_enter(e):
            btn['bg'] = hover_bg
        def on_leave(e):
            btn['bg'] = bg
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def build_layout(self):
        # TOP HEADER
        header_frame = tk.Frame(self.root, bg=COLOR_CARD, height=80, relief="solid", bd=1)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🌌 COSMOS PIPELINE TOOL", bg=COLOR_CARD, fg=COLOR_GOLD, font=("Segoe UI", 18, "bold"))
        title_label.pack(anchor=tk.W, padx=20, pady=(10, 0))
        sub_label = tk.Label(header_frame, text="Zodiac Star Constellation Image Preprocessing & Artificial Neural Network Classifier (INF322)", bg=COLOR_CARD, fg=COLOR_TEXT_MUTED, font=("Segoe UI", 9, "italic"))
        sub_label.pack(anchor=tk.W, padx=20)
        
        # MAIN CONTAINER
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # LEFT CONTROLLER
        left_panel = ttk.Frame(main_frame, width=440)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Step Selector Frame
        step_card = ttk.Frame(left_panel, style="Card.TFrame", padding=12)
        step_card.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(step_card, text="STEPS PIPELINE SELECTION", style="CardTitle.TLabel").pack(anchor=tk.W, pady=(0, 8))
        
        steps = [
            (1, "1. Color-to-Grayscale Conversion"),
            (2, "2. Binarization (Otsu Threshold)"),
            (3, "3. Resizing (Down-pooling Configuration)"),
            (4, "4. Create Dataset & Labels (.npy output)"),
            (5, "5. Randomize Dataset & Labels"),
            (6, "6. Train Artificial Neural Network"),
            (7, "7. Interactive Test ANN by Image Index"),
            (8, "8. Upload & Predict Custom Image")
        ]
        
        self.step_radio_btns = []
        for val, txt in steps:
            rb = tk.Radiobutton(
                step_card,
                text=txt,
                variable=self.selected_step,
                value=val,
                bg=COLOR_CARD,
                fg=COLOR_TEXT_PRIMARY,
                selectcolor=COLOR_CARD,
                activebackground=COLOR_CARD,
                activeforeground=COLOR_GOLD,
                font=("Segoe UI", 10),
                anchor=tk.W,
                command=self.on_step_selected
            )
            rb.pack(fill=tk.X, pady=3)
            self.step_radio_btns.append(rb)
            
        # Configurations Frame
        self.config_card = ttk.Frame(left_panel, style="Card.TFrame", padding=12)
        self.config_card.pack(fill=tk.X, pady=(0, 10))
        
        self.config_title = ttk.Label(self.config_card, text="STEP CONFIGURATION", style="CardTitle.TLabel")
        self.config_title.pack(anchor=tk.W, pady=(0, 8))
        
        # Dynamic configuration container
        self.config_inner_frame = ttk.Frame(self.config_card)
        self.config_inner_frame.pack(fill=tk.BOTH, expand=True)
        
        # Execution Card
        exec_card = ttk.Frame(left_panel, style="Card.TFrame", padding=12)
        exec_card.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_start = self.make_flat_button(exec_card, "START OPERATION ▷", self.execute_current_step, bg=COLOR_GOLD, fg=COLOR_BG_SPACE, hover_bg="#E0A82E")
        self.btn_start.pack(fill=tk.X, pady=5)
        
        # Status & Log Frame
        log_card = ttk.Frame(left_panel, style="Card.TFrame", padding=12)
        log_card.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(log_card, text="SYSTEM ACTIVITY LOG", style="CardTitle.TLabel").pack(anchor=tk.W, pady=(0, 5))
        
        self.log_txt = tk.Text(log_card, bg=COLOR_BG_SPACE, fg=COLOR_TEXT_PRIMARY, bd=0, font=("Consolas", 9), height=10)
        self.log_txt.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT VIEW PANEL (Observatory)
        right_panel = ttk.Frame(main_frame, style="Card.TFrame", padding=12)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Preview Selection Controls
        preview_ctrls = ttk.Frame(right_panel)
        preview_ctrls.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(preview_ctrls, text="Zodiac Class:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.cb_class = ttk.Combobox(preview_ctrls, textvariable=self.preview_class, values=KELAS, state="readonly", width=12)
        self.cb_class.pack(side=tk.LEFT, padx=(0, 15))
        self.cb_class.bind("<<ComboboxSelected>>", lambda e: self.refresh_observatory())
        
        ttk.Label(preview_ctrls, text="Sample Index:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.cb_index = ttk.Combobox(preview_ctrls, textvariable=self.preview_index, values=list(range(JUMLAH_SAMPEL_PER_KELAS)), state="readonly", width=4)
        self.cb_index.pack(side=tk.LEFT, padx=(0, 15))
        self.cb_index.bind("<<ComboboxSelected>>", lambda e: self.refresh_observatory())
        
        self.btn_prev = self.make_flat_button(preview_ctrls, "◀ Prev", self.prev_preview, font=("Segoe UI", 9, "bold"))
        self.btn_prev.pack(side=tk.LEFT, padx=3)
        self.btn_next = self.make_flat_button(preview_ctrls, "Next ▶", self.next_preview, font=("Segoe UI", 9, "bold"))
        self.btn_next.pack(side=tk.LEFT, padx=3)
        
        # Matplotlib visualization canvas
        self.fig, self.axes = plt.subplots(1, 2, figsize=(8, 5))
        self.fig.patch.set_facecolor(COLOR_CARD)
        for ax in self.axes:
            ax.set_facecolor(COLOR_BG_SPACE)
            ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=8)
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Initialize Config Panel & Observatory View
        self.on_step_selected()
        self.refresh_observatory()
        self.log_message("System initialized. Ready for operations.")

    # ====================================================================
    # EVENT HANDLERS & CALLBACKS
    # ====================================================================
    def log_message(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {text}\n"
        self.log_txt.insert(tk.END, log_entry)
        self.log_txt.see(tk.END)
        
        # Write to disk log file
        log_file = os.path.join(PATH_LOG, "app_log.txt")
        with open(log_file, "a") as f:
            f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")

    def on_step_selected(self):
        # Clear config inner layout
        for child in self.config_inner_frame.winfo_children():
            child.destroy()
            
        step = self.selected_step.get()
        self.btn_start.configure(text=f"START PIPELINE STEP {step} ▷")
        
        if step == 1:
            self.config_title.configure(text="STEP 1: COLOR-TO-GRAYSCALE")
            lbl = ttk.Label(self.config_inner_frame, text="Converts RGB images to single-channel luminosity.\nGenerates: 05_GUI_App/output/grayscale/\nNo configuration needed.", wraplength=400, justify=tk.LEFT)
            lbl.pack(pady=10, fill=tk.X)
            
        elif step == 2:
            self.config_title.configure(text="STEP 2: BINARIZATION (OTSU)")
            lbl = ttk.Label(self.config_inner_frame, text="Applies Otsu Threshold to separate stars from background.\nGenerates: 05_GUI_App/output/binarization/\nNo configuration needed.", wraplength=400, justify=tk.LEFT)
            lbl.pack(pady=10, fill=tk.X)
            
        elif step == 3:
            self.config_title.configure(text="STEP 3: RESIZING & POOLING")
            
            # Row Col Size inputs
            size_f = ttk.Frame(self.config_inner_frame)
            size_f.pack(fill=tk.X, pady=5)
            
            ttk.Label(size_f, text="Row:").grid(row=0, column=0, sticky=tk.W, pady=2)
            ent_row = ttk.Entry(size_f, textvariable=self.row_size, width=6)
            ent_row.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
            
            ttk.Label(size_f, text="Col:").grid(row=1, column=0, sticky=tk.W, pady=2)
            ent_col = ttk.Entry(size_f, textvariable=self.col_size, width=6)
            ent_col.grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
            
            # Pooling selector
            pool_f = ttk.Frame(self.config_inner_frame)
            pool_f.pack(fill=tk.X, pady=8)
            ttk.Label(pool_f, text="Pooling Method:").pack(anchor=tk.W, pady=2)
            
            rb_avg = ttk.Radiobutton(pool_f, text="Resizing with Average Pooling", variable=self.pooling_mode, value="average")
            rb_avg.pack(anchor=tk.W, pady=2)
            
            rb_max = ttk.Radiobutton(pool_f, text="Resizing with Max Pooling", variable=self.pooling_mode, value="max")
            rb_max.pack(anchor=tk.W, pady=2)
            
        elif step == 4:
            self.config_title.configure(text="STEP 4: CREATE DATASET")
            lbl = ttk.Label(self.config_inner_frame, text="Flattens 2D processed binary images into input vectors.\nSaves inputs_*.npy and labels_*.npy to dataset/.\nReads current Row/Col & Pooling config.", wraplength=400, justify=tk.LEFT)
            lbl.pack(pady=10, fill=tk.X)
            
        elif step == 5:
            self.config_title.configure(text="STEP 5: RANDOMIZE DATASET")
            lbl = ttk.Label(self.config_inner_frame, text="Applies synchronized shuffle to inputs and labels.\nSaves random_inputs_*.npy and random_labels_*.npy.", wraplength=400, justify=tk.LEFT)
            lbl.pack(pady=10, fill=tk.X)
            
        elif step == 6:
            self.config_title.configure(text="STEP 6: TRAIN ANN MODEL")
            lbl = ttk.Label(self.config_inner_frame, text="Trains a NumPy Backpropagation Neural Network (MLP).\nArchitecture: Inputs -> 20 Hidden -> 12 Outputs.\nEpochs: 100 | Learning Rate: 0.01\nUpdates weights under ann/.", wraplength=400, justify=tk.LEFT)
            lbl.pack(pady=10, fill=tk.X)
            
        elif step == 7:
            self.config_title.configure(text="STEP 7: INTERACTIVE TEST INDEX")
            test_f = ttk.Frame(self.config_inner_frame)
            test_f.pack(fill=tk.X, pady=5)
            
            max_idx = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS - 1
            ttk.Label(test_f, text=f"Data Index to Test (0 - {max_idx}):").pack(anchor=tk.W, pady=2)
            
            ent_idx = ttk.Entry(test_f, textvariable=self.test_index, width=8)
            ent_idx.pack(anchor=tk.W, pady=2)
            
            lbl_desc = ttk.Label(self.config_inner_frame, text="Retrieves the randomized sample from dataset, feeds it forward through weights, and matches it with output labels.", wraplength=400, justify=tk.LEFT)
            lbl_desc.pack(pady=10, fill=tk.X)
            
        elif step == 8:
            self.config_title.configure(text="STEP 8: UPLOAD & PREDICT CUSTOM IMAGE")
            lbl_desc = ttk.Label(self.config_inner_frame, text="Upload a raw constellation image from your computer. The app will automatically convert it, threshold it, pool it to 28x28, and run model prediction.", wraplength=400, justify=tk.LEFT)
            lbl_desc.pack(pady=10, fill=tk.X)
            btn_upload = self.make_flat_button(self.config_inner_frame, "Upload & Predict Image 📁", self.run_custom_predict, bg=COLOR_CYAN, fg=COLOR_BG_SPACE, hover_bg="#19B5FE")
            btn_upload.pack(fill=tk.X, pady=5)
            
        self.refresh_observatory()

    def prev_preview(self):
        cur = self.preview_index.get()
        if cur > 0:
            self.preview_index.set(cur - 1)
            self.refresh_observatory()

    def next_preview(self):
        cur = self.preview_index.get()
        if cur < JUMLAH_SAMPEL_PER_KELAS - 1:
            self.preview_index.set(cur + 1)
            self.refresh_observatory()

    # ====================================================================
    # OBSERVATORY DRAW LOGIC (Side by Side Visualizer)
    # ====================================================================
    def refresh_observatory(self):
        for ax in self.axes:
            ax.clear()
            ax.axis("on")
            ax.grid(False)
            ax.set_facecolor(COLOR_BG_SPACE)
            
        step = self.selected_step.get()
        kelas = self.preview_class.get()
        idx = self.preview_index.get()
        fname = f"{kelas} ({idx}).jpg"
        
        # Configure fonts and titles in Matplotlib dark style
        self.fig.suptitle(f"STELLAR PREVIEW: {kelas} Constellation Sample ({idx})", color=COLOR_TEXT_PRIMARY, fontname="Segoe UI", fontsize=12, weight="bold")
        
        try:
            if step == 1: # Color to Grayscale
                # Input
                img_in = baca_gambar(PATH_INPUT, fname)
                if img_in is not None:
                    self.axes[0].imshow(img_in)
                    self.axes[0].set_title("Input: Color Image", color=COLOR_CYAN, fontsize=10, weight="bold")
                
                # Output
                path_out = os.path.join(PATH_OUT_GS, fname)
                if os.path.exists(path_out):
                    img_out = plt.imread(path_out)
                    self.axes[1].imshow(img_out, cmap="gray")
                    self.axes[1].set_title("Output: Grayscale", color=COLOR_CYAN, fontsize=10, weight="bold")
                else:
                    self.axes[1].text(0.5, 0.5, "Click Start to process", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
            elif step == 2: # Grayscale to Binarization
                # Input
                path_in = os.path.join(PATH_OUT_GS, fname)
                if os.path.exists(path_in):
                    img_in = plt.imread(path_in)
                    self.axes[0].imshow(img_in, cmap="gray")
                    self.axes[0].set_title("Input: Grayscale", color=COLOR_CYAN, fontsize=10, weight="bold")
                else:
                    self.axes[0].text(0.5, 0.5, "Requires Grayscale (Step 1)", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[0].axis("off")
                
                # Output
                path_out = os.path.join(PATH_OUT_BIN, fname)
                if os.path.exists(path_out):
                    img_out = plt.imread(path_out)
                    self.axes[1].imshow(img_out, cmap="gray")
                    self.axes[1].set_title("Output: Binarized (Otsu)", color=COLOR_CYAN, fontsize=10, weight="bold")
                else:
                    self.axes[1].text(0.5, 0.5, "Click Start to process", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
            elif step == 3: # Binarization to Resizing (Ready)
                # Input
                path_in = os.path.join(PATH_OUT_BIN, fname)
                if os.path.exists(path_in):
                    img_in = plt.imread(path_in)
                    self.axes[0].imshow(img_in, cmap="gray")
                    self.axes[0].set_title("Input: Binary Image", color=COLOR_CYAN, fontsize=10, weight="bold")
                else:
                    self.axes[0].text(0.5, 0.5, "Requires Binarization (Step 2)", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[0].axis("off")
                
                # Output
                mode = self.pooling_mode.get()
                out_ready_dir = PATH_OUT_READY_AVG if mode == "average" else PATH_OUT_READY_MAX
                path_out = os.path.join(out_ready_dir, f"{kelas} ({idx})_ready.jpg")
                if os.path.exists(path_out):
                    img_out = plt.imread(path_out)
                    self.axes[1].imshow(img_out, cmap="gray")
                    self.axes[1].set_title(f"Output: Resized ({self.row_size.get()}x{self.col_size.get()} {mode})", color=COLOR_CYAN, fontsize=10, weight="bold")
                else:
                    self.axes[1].text(0.5, 0.5, "Click Start to process", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
            elif step == 4: # Dataset Creation
                mode = self.pooling_mode.get()
                out_ready_dir = PATH_OUT_READY_AVG if mode == "average" else PATH_OUT_READY_MAX
                path_in = os.path.join(out_ready_dir, f"{kelas} ({idx})_ready.jpg")
                
                if os.path.exists(path_in):
                    img_in = plt.imread(path_in)
                    self.axes[0].imshow(img_in, cmap="gray")
                    self.axes[0].set_title("Input: Ready Image", color=COLOR_CYAN, fontsize=10, weight="bold")
                else:
                    self.axes[0].text(0.5, 0.5, "Requires Ready (Step 3)", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[0].axis("off")
                    
                # Output: Plot a 1D visualization of the vector in dataset if files exist
                r = self.row_size.get()
                c = self.col_size.get()
                col = r * c + 1
                dataset_file = os.path.join(PATH_DATASET, f"inputs_{len(KELAS)*JUMLAH_SAMPEL_PER_KELAS}_{col}.npy")
                if os.path.exists(dataset_file):
                    data = np.load(dataset_file)
                    # Find corresponding row in dataset
                    row_idx = KELAS.index(kelas) * JUMLAH_SAMPEL_PER_KELAS + idx
                    pixel_vector = data[row_idx, 1:]
                    self.axes[1].plot(pixel_vector, color=COLOR_GOLD)
                    self.axes[1].set_title("Output: 1D Feature Vector", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[1].set_xlabel("Pixel Index", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[1].set_ylabel("Luminosity", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[1].set_facecolor(COLOR_BG_SPACE)
                else:
                    self.axes[1].text(0.5, 0.5, "Dataset not created yet", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
            elif step == 5: # Randomization
                r = self.row_size.get()
                c = self.col_size.get()
                col = r * c + 1
                dataset_file = os.path.join(PATH_DATASET, f"inputs_{len(KELAS)*JUMLAH_SAMPEL_PER_KELAS}_{col}.npy")
                rand_dataset_file = os.path.join(PATH_DATASET, f"random_inputs_{len(KELAS)*JUMLAH_SAMPEL_PER_KELAS}_{col}.npy")
                
                # Plot sequential indices vs randomized indices
                if os.path.exists(dataset_file):
                    data = np.load(dataset_file)
                    self.axes[0].plot(data[:, 0], 'o', color=COLOR_CYAN, markersize=2)
                    self.axes[0].set_title("Input: Sequential Index Order", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[0].set_xlabel("Sample Row ID", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[0].set_ylabel("Original File ID", color=COLOR_TEXT_MUTED, fontsize=8)
                else:
                    self.axes[0].text(0.5, 0.5, "Base dataset not found", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[0].axis("off")
                    
                if os.path.exists(rand_dataset_file):
                    rand_data = np.load(rand_dataset_file)
                    self.axes[1].plot(rand_data[:, 0], 'o', color=COLOR_GOLD, markersize=2)
                    self.axes[1].set_title("Output: Shuffled Index Order", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[1].set_xlabel("Sample Row ID", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[1].set_ylabel("Original File ID", color=COLOR_TEXT_MUTED, fontsize=8)
                else:
                    self.axes[1].text(0.5, 0.5, "Click Start to Shuffle", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
            elif step == 6: # ANN Training loss accuracy curves
                if len(self.loss_history) > 0:
                    self.axes[0].plot(self.loss_history, color=COLOR_RED, label="Loss")
                    self.axes[0].set_title("ANN Training: Loss Profile", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[0].set_xlabel("Epoch", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[0].set_ylabel("Squared Error", color=COLOR_TEXT_MUTED, fontsize=8)
                    
                    self.axes[1].plot(self.accuracy_history, color=COLOR_GREEN, label="Accuracy")
                    self.axes[1].set_title("ANN Training: Accuracy Profile", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[1].set_xlabel("Epoch", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[1].set_ylabel("Accuracy %", color=COLOR_TEXT_MUTED, fontsize=8)
                else:
                    # If loaded from HDD, display weight parameters
                    path_weights = os.path.join(PATH_ANN, "a3_w_i_h.npy")
                    if os.path.exists(path_weights):
                        w = np.load(path_weights)
                        self.axes[0].imshow(w, aspect="auto", cmap="viridis")
                        self.axes[0].set_title("Hidden Weights Matrix", color=COLOR_CYAN, fontsize=10, weight="bold")
                        self.axes[0].set_xlabel("Features", color=COLOR_TEXT_MUTED, fontsize=8)
                        self.axes[0].set_ylabel("Neurons", color=COLOR_TEXT_MUTED, fontsize=8)
                        
                        self.axes[1].text(0.5, 0.5, "Ready for evaluation.\nModel trained and saved.\nClick Start to re-train.", ha="center", va="center", color=COLOR_GOLD)
                        self.axes[1].axis("off")
                    else:
                        self.axes[0].text(0.5, 0.5, "No weights found.\nClick Start to train.", ha="center", va="center", color=COLOR_TEXT_MUTED)
                        self.axes[0].axis("off")
                        self.axes[1].text(0.5, 0.5, "Accuracy Curve will appear here.", ha="center", va="center", color=COLOR_TEXT_MUTED)
                        self.axes[1].axis("off")
                        
            elif step == 7: # ANN Testing predictions
                if self.last_test_img is not None:
                    # Draw testing input image
                    self.axes[0].imshow(self.last_test_img, cmap="Greys")
                    self.axes[0].set_title(f"Input Sample (Index: {self.test_index.get()})", color=COLOR_CYAN, fontsize=10, weight="bold")
                    
                    # Draw probability distributions
                    y_pos = np.arange(len(KELAS))
                    self.axes[1].barh(y_pos, self.last_test_probs * 100, color=COLOR_GOLD)
                    self.axes[1].set_yticks(y_pos)
                    self.axes[1].set_yticklabels(KELAS, color=COLOR_TEXT_PRIMARY, fontsize=7)
                    self.axes[1].set_title("Prediction Probabilities (%)", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[1].set_xlabel("Probability (%)", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[1].set_xlim(0, 100)
                    self.axes[1].set_facecolor(COLOR_BG_SPACE)
                    
                    # Add highlight on prediction vs true labels
                    fig_title = f"True: {self.last_test_true} | Pred: {self.last_test_pred}"
                    color_accent = COLOR_GREEN if self.last_test_true == self.last_test_pred else COLOR_RED
                    self.fig.suptitle(fig_title, color=color_accent, fontname="Segoe UI", fontsize=12, weight="bold")
                else:
                    self.axes[0].text(0.5, 0.5, "Enter index in config panel\nand click Start.", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[0].axis("off")
                    self.axes[1].text(0.5, 0.5, "Bar chart will display prediction scores.", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
            elif step == 8: # Custom Image Prediction
                if hasattr(self, 'custom_original_img') and self.custom_original_img is not None:
                    # Left: Show original custom image
                    self.axes[0].imshow(self.custom_original_img)
                    self.axes[0].set_title("Original Custom Image", color=COLOR_CYAN, fontsize=10, weight="bold")
                    
                    # Right: Draw probability distributions
                    y_pos = np.arange(len(KELAS))
                    self.axes[1].barh(y_pos, self.last_test_probs * 100, color=COLOR_GOLD)
                    self.axes[1].set_yticks(y_pos)
                    self.axes[1].set_yticklabels(KELAS, color=COLOR_TEXT_PRIMARY, fontsize=7)
                    self.axes[1].set_title("Prediction Probabilities (%)", color=COLOR_CYAN, fontsize=10, weight="bold")
                    self.axes[1].set_xlabel("Probability (%)", color=COLOR_TEXT_MUTED, fontsize=8)
                    self.axes[1].set_xlim(0, 100)
                    self.axes[1].set_facecolor(COLOR_BG_SPACE)
                    
                    fig_title = f"Predicted Class: {self.last_test_pred}"
                    self.fig.suptitle(fig_title, color=COLOR_GOLD, fontname="Segoe UI", fontsize=12, weight="bold")
                else:
                    self.axes[0].text(0.5, 0.5, "Click Upload & Predict\nin config panel.", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[0].axis("off")
                    self.axes[1].text(0.5, 0.5, "Bar chart will display prediction scores.", ha="center", va="center", color=COLOR_TEXT_MUTED)
                    self.axes[1].axis("off")
                    
        except Exception as e:
            self.axes[0].text(0.5, 0.5, f"Preview Error:\n{str(e)}", ha="center", va="center", color=COLOR_RED)
            self.axes[0].axis("off")
            self.axes[1].text(0.5, 0.5, "Failed to load visualization data", ha="center", va="center", color=COLOR_TEXT_MUTED)
            self.axes[1].axis("off")
            
        self.canvas.draw()

    # ====================================================================
    # PIPELINE EXECUTION WRAPPER
    # ====================================================================
    def execute_current_step(self):
        step = self.selected_step.get()
        
        if step == 1:
            self.run_grayscale()
        elif step == 2:
            self.run_binarization()
        elif step == 3:
            self.run_resizing()
        elif step == 4:
            self.run_create_dataset()
        elif step == 5:
            self.run_randomize()
        elif step == 6:
            self.run_ann_training()
        elif step == 7:
            self.run_ann_testing()
        elif step == 8:
            self.run_custom_predict()

    def run_grayscale(self):
        self.log_message("Starting batch color-to-grayscale conversion...")
        self.btn_start.configure(state=tk.DISABLED)
        self.root.config(cursor="watch")
        self.root.update()
        
        success = 0
        total = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS
        
        for idx_k, kelas in enumerate(KELAS):
            for idx_s in range(JUMLAH_SAMPEL_PER_KELAS):
                fname = f"{kelas} ({idx_s}).jpg"
                pic = baca_gambar(PATH_INPUT, fname)
                if pic is not None:
                    pic_gs = grayscale(pic)
                    plt.imsave(os.path.join(PATH_OUT_GS, fname), pic_gs, cmap="gray")
                    # Also save to same_folder for proof
                    plt.imsave(os.path.join(PATH_SAME_FOLDER, f"{kelas} ({idx_s})_0_original.jpg"), pic)
                    plt.imsave(os.path.join(PATH_SAME_FOLDER, f"{kelas} ({idx_s})_1_grayscale.jpg"), pic_gs, cmap="gray")
                    success += 1
                
                # Keep GUI refreshed and update class combos
                if success % 20 == 0:
                    self.preview_class.set(kelas)
                    self.preview_index.set(idx_s)
                    self.refresh_observatory()
                    self.root.update()
                    
        self.btn_start.configure(state=tk.NORMAL)
        self.root.config(cursor="")
        self.log_message(f"Grayscale conversion completed: {success}/{total} processed successfully.")
        self.refresh_observatory()
        messagebox.showinfo("Success", "Color-to-Grayscale conversion completed!")

    def run_binarization(self):
        self.log_message("Starting batch image binarization...")
        
        # Check dependency
        if not os.path.exists(PATH_OUT_GS) or len(os.listdir(PATH_OUT_GS)) == 0:
            messagebox.showerror("Dependency Error", "Grayscale outputs not found! Please run Step 1 first.")
            return
            
        self.btn_start.configure(state=tk.DISABLED)
        self.root.config(cursor="watch")
        self.root.update()
        
        success = 0
        total = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS
        
        for idx_k, kelas in enumerate(KELAS):
            for idx_s in range(JUMLAH_SAMPEL_PER_KELAS):
                fname = f"{kelas} ({idx_s}).jpg"
                pic_gs = baca_gambar(PATH_OUT_GS, fname)
                if pic_gs is not None:
                    # Convert to grayscale 2D array if read as rgb
                    pic_gs_2d = pic_gs[:, :, 0]
                    pic_bin, th = binarisasi(pic_gs_2d)
                    plt.imsave(os.path.join(PATH_OUT_BIN, fname), pic_bin, cmap="gray")
                    # Also save to same_folder for proof
                    plt.imsave(os.path.join(PATH_SAME_FOLDER, f"{kelas} ({idx_s})_2_binarization.jpg"), pic_bin, cmap="gray")
                    success += 1
                    
                if success % 20 == 0:
                    self.preview_class.set(kelas)
                    self.preview_index.set(idx_s)
                    self.refresh_observatory()
                    self.root.update()
                    
        self.btn_start.configure(state=tk.NORMAL)
        self.root.config(cursor="")
        self.log_message(f"Binarization completed: {success}/{total} processed.")
        self.refresh_observatory()
        messagebox.showinfo("Success", "Binarization completed!")

    def run_resizing(self):
        r = self.row_size.get()
        c = self.col_size.get()
        mode = self.pooling_mode.get()
        
        self.log_message(f"Starting batch resizing using {mode} pooling ({r}x{c})...")
        
        # Check dependencies
        if not os.path.exists(PATH_OUT_BIN) or len(os.listdir(PATH_OUT_BIN)) == 0:
            messagebox.showerror("Dependency Error", "Binarized outputs not found! Please run Step 2 first.")
            return
            
        self.btn_start.configure(state=tk.DISABLED)
        self.root.config(cursor="watch")
        self.root.update()
        
        out_res_dir = PATH_OUT_RESIZE_AVG if mode == "average" else PATH_OUT_RESIZE_MAX
        out_ready_dir = PATH_OUT_READY_AVG if mode == "average" else PATH_OUT_READY_MAX
        os.makedirs(out_res_dir, exist_ok=True)
        os.makedirs(out_ready_dir, exist_ok=True)
        
        success = 0
        total = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS
        
        for idx_k, kelas in enumerate(KELAS):
            for idx_s in range(JUMLAH_SAMPEL_PER_KELAS):
                fname = f"{kelas} ({idx_s}).jpg"
                pic_bin = baca_gambar(PATH_OUT_BIN, fname)
                if pic_bin is not None:
                    pic_bin_2d = pic_bin[:, :, 0]
                    pic_clean = hapus_noise_titik(pic_bin_2d)
                    pic_crop = crop_objek(pic_clean)
                    pic_square = buat_kanvas_persegi(pic_crop)
                    pic_ready, pic_res = down_pooling(pic_square, r, c, mode)
                    
                    pic_ready_rgb = ubah_ke_rgb(pic_ready)
                    
                    out_name = f"{kelas} ({idx_s})_ready.jpg"
                    plt.imsave(os.path.join(out_res_dir, fname), pic_res, cmap="gray")
                    plt.imsave(os.path.join(out_ready_dir, out_name), pic_ready_rgb)
                    # Also save to same_folder for proof
                    plt.imsave(os.path.join(PATH_SAME_FOLDER, f"{kelas} ({idx_s})_3_resized.jpg"), pic_res, cmap="gray")
                    plt.imsave(os.path.join(PATH_SAME_FOLDER, f"{kelas} ({idx_s})_4_ready.jpg"), pic_ready_rgb)
                    success += 1
                    
                if success % 20 == 0:
                    self.preview_class.set(kelas)
                    self.preview_index.set(idx_s)
                    self.refresh_observatory()
                    self.root.update()
                    
        self.btn_start.configure(state=tk.NORMAL)
        self.root.config(cursor="")
        self.log_message(f"Resizing and down-pooling completed: {success}/{total} processed.")
        self.refresh_observatory()
        messagebox.showinfo("Success", f"Resizing and Ready images generated successfully under ready/{mode}/.")

    def run_create_dataset(self):
        r = self.row_size.get()
        c = self.col_size.get()
        mode = self.pooling_mode.get()
        
        self.log_message("Compiling inputs and labels datasets...")
        
        # Check dependencies
        out_ready_dir = PATH_OUT_READY_AVG if mode == "average" else PATH_OUT_READY_MAX
        if not os.path.exists(out_ready_dir) or len(os.listdir(out_ready_dir)) == 0:
            messagebox.showerror("Dependency Error", f"Ready images for {mode} pooling not found. Please run Step 3 Resizing first.")
            return
            
        juml_kelas = len(KELAS)
        juml_sampel = juml_kelas * JUMLAH_SAMPEL_PER_KELAS
        juml_kolom = r * c + 1
        
        # 1. Inputs
        dataset_inp = np.zeros(shape=(juml_sampel, juml_kolom), dtype=np.uint16)
        for k in range(0, juml_sampel):
            dataset_inp[k, 0] = k
            
        k = 0
        for kelas in KELAS:
            for idx_s in range(JUMLAH_SAMPEL_PER_KELAS):
                nama_file = f"{kelas} ({idx_s})_ready.jpg"
                path_img = os.path.join(out_ready_dir, nama_file)
                if os.path.exists(path_img):
                    pic = plt.imread(path_img)
                    if pic.dtype != np.uint8:
                        pic = (pic * 255).astype(np.uint8)
                    temp = pic[:, :, 0] if len(pic.shape) == 3 else pic[:, :]
                    
                    data = temp.reshape(r * c)
                    dataset_inp[k, 1:juml_kolom] = data
                k += 1
                
        nama_file_dataset = f"inputs_{juml_sampel}_{juml_kolom}.npy"
        np.save(os.path.join(PATH_DATASET, nama_file_dataset), dataset_inp)
        np.save(os.path.join(PATH_SAME_FOLDER, nama_file_dataset), dataset_inp)
        self.log_message(f"Input dataset saved: {nama_file_dataset}")
        
        # 2. Labels
        labels = np.zeros(shape=(juml_sampel, juml_kelas + 1), dtype=float)
        for k in range(0, juml_sampel):
            labels[k, 0] = k
            posisi = int(k / JUMLAH_SAMPEL_PER_KELAS) + 1
            labels[k, posisi] = 1.0
            
        nama_file_labels = f"labels_{juml_sampel}_{juml_kelas + 1}.npy"
        np.save(os.path.join(PATH_DATASET, nama_file_labels), labels)
        np.save(os.path.join(PATH_SAME_FOLDER, nama_file_labels), labels)
        self.log_message(f"Label dataset saved: {nama_file_labels}")
        
        self.refresh_observatory()
        messagebox.showinfo("Success", f"Dataset Compiled!\nInputs: {dataset_inp.shape}\nLabels: {labels.shape}")

    def run_randomize(self):
        r = self.row_size.get()
        c = self.col_size.get()
        mode = self.pooling_mode.get()
        
        juml_sampel = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS
        col = r * c + 1
        juml_label_kolom = len(KELAS) + 1
        
        file_input = f"inputs_{juml_sampel}_{col}.npy"
        file_label = f"labels_{juml_sampel}_{juml_label_kolom}.npy"
        
        path_in = os.path.join(PATH_DATASET, file_input)
        path_lb = os.path.join(PATH_DATASET, file_label)
        
        if not os.path.exists(path_in) or not os.path.exists(path_lb):
            messagebox.showerror("Dependency Error", "Base inputs/labels dataset files not found. Run Step 4 first.")
            return
            
        self.log_message("Applying randomized index mapping...")
        inputs = np.load(path_in)
        labels = np.load(path_lb)
        
        m, n = np.shape(inputs)
        a = list(range(0, m))
        np.random.shuffle(a)
        
        inputs_random = np.zeros(shape=(m, n), dtype=float)
        labels_random = np.zeros(shape=(m, labels.shape[1]), dtype=float)
        
        for i in range(0, m):
            inputs_random[i, :] = inputs[a[i], :]
            labels_random[i, :] = labels[a[i], :]
            
        np.save(os.path.join(PATH_DATASET, "random_" + file_input), inputs_random)
        np.save(os.path.join(PATH_DATASET, "random_" + file_label), labels_random)
        np.save(os.path.join(PATH_SAME_FOLDER, "random_" + file_input), inputs_random)
        np.save(os.path.join(PATH_SAME_FOLDER, "random_" + file_label), labels_random)
        
        self.log_message("Randomized dataset files created successfully.")
        self.refresh_observatory()
        messagebox.showinfo("Success", "Synchronized randomization completed!")

    def run_ann_training(self):
        r = self.row_size.get()
        c = self.col_size.get()
        
        juml_sampel = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS
        col = r * c + 1
        juml_label_kolom = len(KELAS) + 1
        
        file_input_rand = f"random_inputs_{juml_sampel}_{col}.npy"
        file_label_rand = f"random_labels_{juml_sampel}_{juml_label_kolom}.npy"
        
        path_in = os.path.join(PATH_DATASET, file_input_rand)
        path_lb = os.path.join(PATH_DATASET, file_label_rand)
        
        if not os.path.exists(path_in) or not os.path.exists(path_lb):
            messagebox.showerror("Dependency Error", "Randomized inputs/labels dataset not found. Run Step 5 first.")
            return
            
        self.log_message("Starting NumPy Artificial Neural Network training...")
        self.btn_start.configure(state=tk.DISABLED)
        self.root.config(cursor="watch")
        self.root.update()
        
        inputs_full = np.load(path_in)
        labels_full = np.load(path_lb)
        
        inputs = inputs_full[:, 1:] / 255.0  # Normalize
        labels = labels_full[:, 1:]
        
        m, n = inputs.shape
        o_n = labels.shape[1]
        h_n = 20
        epochs = 100
        learn_rate = 0.01
        
        # Initialize weights
        w_i_h = np.random.uniform(-0.5, 0.5, (h_n, n))
        w_h_o = np.random.uniform(-0.5, 0.5, (o_n, h_n))
        b_i_h = np.zeros((h_n, 1))
        b_h_o = np.zeros((o_n, 1))
        
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(1, epochs + 1):
            nr_correct = 0
            epoch_loss = 0.0
            
            for inp, label in zip(inputs, labels):
                inp = inp.reshape(r * c, 1)
                label = label.reshape(o_n, 1)
                
                # Forward Pass
                h_pre = b_i_h + w_i_h @ inp
                h = sigmoid(h_pre)
                
                o_pre = b_h_o + w_h_o @ h
                output = sigmoid(o_pre)
                
                nr_correct += int(np.argmax(output) == np.argmax(label))
                epoch_loss += np.sum((output - label) ** 2)
                
                # Backprop
                delta_o = output - label
                w_h_o += -learn_rate * delta_o @ np.transpose(h)
                b_h_o += -learn_rate * delta_o
                
                delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
                w_i_h += -learn_rate * delta_h @ np.transpose(inp)
                b_i_h += -learn_rate * delta_h
                
            acc = (nr_correct / m) * 100
            self.loss_history.append(epoch_loss / m)
            self.accuracy_history.append(acc)
            
            if epoch % 10 == 0 or epoch == 1:
                self.log_message(f"Epoch {epoch:03d}/100: Loss = {self.loss_history[-1]:.4f} | Accuracy = {acc:.2f}%")
                self.refresh_observatory()
                self.root.update()
                
        # Save trained weights
        np.save(os.path.join(PATH_ANN, "a3_b_i_h.npy"), b_i_h)
        np.save(os.path.join(PATH_ANN, "a3_w_i_h.npy"), w_i_h)
        np.save(os.path.join(PATH_ANN, "a3_b_h_o.npy"), b_h_o)
        np.save(os.path.join(PATH_ANN, "a3_w_h_o.npy"), w_h_o)
        
        self.btn_start.configure(state=tk.NORMAL)
        self.root.config(cursor="")
        self.log_message(f"ANN training completed successfully. Final Insample Accuracy: {self.accuracy_history[-1]:.2f}%")
        self.refresh_observatory()
        messagebox.showinfo("Success", f"ANN Training completed!\nFinal Accuracy: {self.accuracy_history[-1]:.2f}%")

    def run_ann_testing(self):
        idx = self.test_index.get()
        juml_sampel = len(KELAS) * JUMLAH_SAMPEL_PER_KELAS
        
        if idx < 0 or idx >= juml_sampel:
            messagebox.showerror("Error", f"Test index out of bounds. Must be 0 - {juml_sampel - 1}")
            return
            
        r = self.row_size.get()
        c = self.col_size.get()
        col = r * c + 1
        juml_label_kolom = len(KELAS) + 1
        
        file_input_rand = f"random_inputs_{juml_sampel}_{col}.npy"
        file_label_rand = f"random_labels_{juml_sampel}_{juml_label_kolom}.npy"
        
        try:
            inputs_full = np.load(os.path.join(PATH_DATASET, file_input_rand))
            labels_full = np.load(os.path.join(PATH_DATASET, file_label_rand))
            b_i_h = np.load(os.path.join(PATH_ANN, "a3_b_i_h.npy"))
            w_i_h = np.load(os.path.join(PATH_ANN, "a3_w_i_h.npy"))
            b_h_o = np.load(os.path.join(PATH_ANN, "a3_b_h_o.npy"))
            w_h_o = np.load(os.path.join(PATH_ANN, "a3_w_h_o.npy"))
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load weight model / test dataset files:\n{str(e)}")
            return
            
        inputs = inputs_full[:, 1:] / 255.0
        labels = labels_full[:, 1:]
        
        inp = inputs[idx].reshape(r * c, 1)
        label_asli_arr = labels[idx]
        
        # Feedforward
        h = sigmoid(b_i_h + w_i_h @ inp)
        output = sigmoid(b_h_o + w_h_o @ h)
        
        true_class = KELAS[label_asli_arr.argmax()]
        pred_class = KELAS[output.argmax()]
        
        self.last_test_img = inputs[idx].reshape(r, c)
        self.last_test_true = true_class
        self.last_test_pred = pred_class
        self.last_test_probs = output.flatten()
        
        self.log_message(f"ANN Test Index {idx} -> True Class: {true_class} | Predicted Class: {pred_class}")
        self.refresh_observatory()

    def run_custom_predict(self):
        filepath = filedialog.askopenfilename(
            title="Pilih Citra Rasi Bintang Mentah",
            filetypes=[("Gambar Citra", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return
            
        try:
            b_i_h = np.load(os.path.join(PATH_ANN, "a3_b_i_h.npy"))
            w_i_h = np.load(os.path.join(PATH_ANN, "a3_w_i_h.npy"))
            b_h_o = np.load(os.path.join(PATH_ANN, "a3_b_h_o.npy"))
            w_h_o = np.load(os.path.join(PATH_ANN, "a3_w_h_o.npy"))
        except Exception as e:
            messagebox.showerror("Model Error", f"Model weights not found. Please train the ANN model (Step 6) first!\nError: {str(e)}")
            return
            
        self.log_message(f"Selected custom image: {os.path.basename(filepath)}")
        
        try:
            pic = plt.imread(filepath)
            if pic.dtype != np.uint8:
                pic = (pic * 255).astype(np.uint8)
            
            # Keep copy of original for preview
            self.custom_original_img = pic.copy()
            
            # If color, convert to grayscale
            if len(pic.shape) == 3:
                pic_gs = grayscale(pic)
            else:
                pic_gs = pic.copy()
                
            # Make sure it's 2D grayscale
            if len(pic_gs.shape) == 3:
                pic_gs_2d = pic_gs[:, :, 0]
            else:
                pic_gs_2d = pic_gs
                
            pic_bin, th = binarisasi(pic_gs_2d)
            pic_clean = hapus_noise_titik(pic_bin)
            pic_crop = crop_objek(pic_clean)
            pic_sq = buat_kanvas_persegi(pic_crop)
            
            r = self.row_size.get()
            c = self.col_size.get()
            mode = self.pooling_mode.get()
            
            pic_final, pic_res = down_pooling(pic_sq, r, c, mode)
            
            # Feedforward Prediction
            inp = pic_res.reshape(r * c, 1) / 255.0  # Normalize vector
            
            h = sigmoid(b_i_h + w_i_h @ inp)
            output = sigmoid(b_h_o + w_h_o @ h)
            
            pred_class = KELAS[output.argmax()]
            probs_sum = np.sum(output)
            confidence = (output.max() / probs_sum * 100) if probs_sum > 0 else 0.0
            
            self.last_test_img = pic_res
            self.last_test_true = "Custom Upload"
            self.last_test_pred = pred_class
            self.last_test_probs = output.flatten()
            
            self.log_message(f"Custom Image Prediction: {pred_class} (Confidence: {confidence:.2f}%)")
            self.refresh_observatory()
            messagebox.showinfo("Prediction Result", f"Predicted Constellation: {pred_class}\nConfidence: {confidence:.2f}%")
            
        except Exception as e:
            self.log_message(f"Error processing custom image: {str(e)}")
            messagebox.showerror("Error", f"Failed to process and predict custom image:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()
