import os
import sys
import unittest
import numpy as np
import tkinter as tk
from unittest.mock import MagicMock, patch

# Ensure we can import app_gui
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import app_gui

class TestCOSMOSPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create Tkinter root and hide it
        cls.root = tk.Tk()
        cls.root.withdraw()
        
        # Patch tkinter messagebox methods so they do not block execution
        cls.mock_showinfo = patch('tkinter.messagebox.showinfo').start()
        cls.mock_showerror = patch('tkinter.messagebox.showerror').start()
        cls.mock_showwarning = patch('tkinter.messagebox.showwarning').start()
        
        # Instantiate AppGUI
        cls.app = app_gui.AppGUI(cls.root)
        
    @classmethod
    def tearDownClass(cls):
        # Stop all patches
        patch.stopall()
        # Destroy Tkinter root
        cls.root.destroy()

    def test_01_folders_exist(self):
        """Test if all required folders are created on initialization"""
        folders = [
            app_gui.PATH_INPUT, app_gui.PATH_OUT_GS, app_gui.PATH_OUT_BIN, 
            app_gui.PATH_OUT_RESIZE_AVG, app_gui.PATH_OUT_RESIZE_MAX, 
            app_gui.PATH_OUT_READY_AVG, app_gui.PATH_OUT_READY_MAX,
            app_gui.PATH_DATASET, app_gui.PATH_ANN, app_gui.PATH_PREVIEW, app_gui.PATH_LOG
        ]
        for path in folders:
            self.assertTrue(os.path.exists(path), f"Folder should exist: {path}")
            
    def test_02_step1_grayscale(self):
        """Test Step 1: Color-to-Grayscale Conversion"""
        print("\n--- Running Blackbox Test: Step 1 (Grayscale) ---")
        self.app.selected_step.set(1)
        self.app.execute_current_step()
        
        # Check that files were created
        total_files = len(app_gui.KELAS) * app_gui.JUMLAH_SAMPEL_PER_KELAS
        created_files = os.listdir(app_gui.PATH_OUT_GS)
        self.assertEqual(len(created_files), total_files, "Grayscale files count mismatch")
        
        # Verify a specific file
        test_file = os.path.join(app_gui.PATH_OUT_GS, f"{app_gui.KELAS[0]} (0).jpg")
        self.assertTrue(os.path.exists(test_file), f"Sample grayscale file missing: {test_file}")

    def test_03_step2_binarization(self):
        """Test Step 2: Binarization (Otsu Threshold)"""
        print("\n--- Running Blackbox Test: Step 2 (Binarization) ---")
        self.app.selected_step.set(2)
        self.app.execute_current_step()
        
        # Check files count
        total_files = len(app_gui.KELAS) * app_gui.JUMLAH_SAMPEL_PER_KELAS
        created_files = os.listdir(app_gui.PATH_OUT_BIN)
        self.assertEqual(len(created_files), total_files, "Binarized files count mismatch")
        
        # Verify a specific file
        test_file = os.path.join(app_gui.PATH_OUT_BIN, f"{app_gui.KELAS[0]} (0).jpg")
        self.assertTrue(os.path.exists(test_file), f"Sample binarized file missing: {test_file}")

    def test_04_step3_resizing_average(self):
        """Test Step 3: Resizing with Average Pooling"""
        print("\n--- Running Blackbox Test: Step 3 (Average Pooling) ---")
        self.app.selected_step.set(3)
        self.app.pooling_mode.set("average")
        self.app.row_size.set(28)
        self.app.col_size.set(28)
        self.app.execute_current_step()
        
        # Check resized and ready files
        total_files = len(app_gui.KELAS) * app_gui.JUMLAH_SAMPEL_PER_KELAS
        
        res_files = os.listdir(app_gui.PATH_OUT_RESIZE_AVG)
        ready_files = os.listdir(app_gui.PATH_OUT_READY_AVG)
        
        self.assertEqual(len(res_files), total_files, "Resized average files count mismatch")
        self.assertEqual(len(ready_files), total_files, "Ready average files count mismatch")

    def test_05_step3_resizing_max(self):
        """Test Step 3: Resizing with Max Pooling"""
        print("\n--- Running Blackbox Test: Step 3 (Max Pooling) ---")
        self.app.selected_step.set(3)
        self.app.pooling_mode.set("max")
        self.app.row_size.set(28)
        self.app.col_size.set(28)
        self.app.execute_current_step()
        
        # Check resized and ready files
        total_files = len(app_gui.KELAS) * app_gui.JUMLAH_SAMPEL_PER_KELAS
        
        res_files = os.listdir(app_gui.PATH_OUT_RESIZE_MAX)
        ready_files = os.listdir(app_gui.PATH_OUT_READY_MAX)
        
        self.assertEqual(len(res_files), total_files, "Resized max files count mismatch")
        self.assertEqual(len(ready_files), total_files, "Ready max files count mismatch")

    def test_06_step4_create_dataset(self):
        """Test Step 4: Create Dataset & Labels"""
        print("\n--- Running Blackbox Test: Step 4 (Create Dataset) ---")
        self.app.selected_step.set(4)
        # We will use average pooling dataset for the rest of the testing
        self.app.pooling_mode.set("average")
        self.app.execute_current_step()
        
        # Check files
        r, c = self.app.row_size.get(), self.app.col_size.get()
        juml_sampel = len(app_gui.KELAS) * app_gui.JUMLAH_SAMPEL_PER_KELAS
        juml_kolom = r * c + 1
        juml_label_kolom = len(app_gui.KELAS) + 1
        
        file_input = f"inputs_{juml_sampel}_{juml_kolom}.npy"
        file_label = f"labels_{juml_sampel}_{juml_label_kolom}.npy"
        
        self.assertTrue(os.path.exists(os.path.join(app_gui.PATH_DATASET, file_input)), f"Dataset inputs file missing: {file_input}")
        self.assertTrue(os.path.exists(os.path.join(app_gui.PATH_DATASET, file_label)), f"Dataset labels file missing: {file_label}")
        
        # Load and verify shapes
        inputs = np.load(os.path.join(app_gui.PATH_DATASET, file_input))
        labels = np.load(os.path.join(app_gui.PATH_DATASET, file_label))
        
        self.assertEqual(inputs.shape, (juml_sampel, juml_kolom))
        self.assertEqual(labels.shape, (juml_sampel, juml_label_kolom))

    def test_07_step5_randomize_dataset(self):
        """Test Step 5: Randomize Dataset"""
        print("\n--- Running Blackbox Test: Step 5 (Randomize Dataset) ---")
        self.app.selected_step.set(5)
        self.app.execute_current_step()
        
        r, c = self.app.row_size.get(), self.app.col_size.get()
        juml_sampel = len(app_gui.KELAS) * app_gui.JUMLAH_SAMPEL_PER_KELAS
        juml_kolom = r * c + 1
        juml_label_kolom = len(app_gui.KELAS) + 1
        
        file_input_rand = f"random_inputs_{juml_sampel}_{juml_kolom}.npy"
        file_label_rand = f"random_labels_{juml_sampel}_{juml_label_kolom}.npy"
        
        self.assertTrue(os.path.exists(os.path.join(app_gui.PATH_DATASET, file_input_rand)), f"Randomized inputs file missing: {file_input_rand}")
        self.assertTrue(os.path.exists(os.path.join(app_gui.PATH_DATASET, file_label_rand)), f"Randomized labels file missing: {file_label_rand}")
        
        # Load and verify
        inputs_rand = np.load(os.path.join(app_gui.PATH_DATASET, file_input_rand))
        labels_rand = np.load(os.path.join(app_gui.PATH_DATASET, file_label_rand))
        
        self.assertEqual(inputs_rand.shape, (juml_sampel, juml_kolom))
        self.assertEqual(labels_rand.shape, (juml_sampel, juml_label_kolom))
        
        # Ensure indices match between randomized inputs and labels
        np.testing.assert_array_equal(inputs_rand[:, 0], labels_rand[:, 0])

    def test_08_step6_ann_training(self):
        """Test Step 6: Train ANN Model"""
        print("\n--- Running Blackbox Test: Step 6 (Train ANN Model) ---")
        self.app.selected_step.set(6)
        
        # Set random seed in test so weight updates are deterministic if needed
        np.random.seed(42)
        
        self.app.execute_current_step()
        
        # Check weights and biases are saved
        weight_files = ["a3_b_i_h.npy", "a3_w_i_h.npy", "a3_b_h_o.npy", "a3_w_h_o.npy"]
        for wf in weight_files:
            path = os.path.join(app_gui.PATH_ANN, wf)
            self.assertTrue(os.path.exists(path), f"ANN parameter file missing: {wf}")
            
        # Verify training accuracy list
        self.assertEqual(len(self.app.accuracy_history), 100)
        self.assertEqual(len(self.app.loss_history), 100)
        
        # Insample accuracy should generally increase or be high
        print(f"Initial training accuracy: {self.app.accuracy_history[0]:.2f}%")
        print(f"Final training accuracy: {self.app.accuracy_history[-1]:.2f}%")
        self.assertGreater(self.app.accuracy_history[-1], self.app.accuracy_history[0])

    def test_09_step7_ann_testing(self):
        """Test Step 7: Interactive Test Index"""
        print("\n--- Running Blackbox Test: Step 7 (Interactive Test Index) ---")
        self.app.selected_step.set(7)
        
        # Test a valid index
        self.app.test_index.set(15)
        self.app.execute_current_step()
        
        # Check prediction results
        self.assertIsNotNone(self.app.last_test_img)
        self.assertIn(self.app.last_test_true, app_gui.KELAS)
        self.assertIn(self.app.last_test_pred, app_gui.KELAS)
        self.assertEqual(len(self.app.last_test_probs), len(app_gui.KELAS))
        print(f"Tested Index 15 -> True Class: {self.app.last_test_true} | Predicted: {self.app.last_test_pred}")

        # Test index out of bounds
        self.app.test_index.set(999)
        self.app.execute_current_step()
        self.mock_showerror.assert_called_with("Error", "Test index out of bounds. Must be 0 - 239")

if __name__ == "__main__":
    unittest.main()
