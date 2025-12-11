import sys
import os
import time
import numpy as np
import pandas as pd
from PIL import Image

# === PyQt5 dependence ===
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QTextEdit, QProgressBar, 
                             QTabWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap, QImage, QDropEvent, QDragEnterEvent

# === Image processing and deep learning dependence ===
from skimage.filters import gaussian, meijering, threshold_otsu
from skimage import exposure, morphology
from skimage.morphology import binary_closing, disk
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

# =========================================================
# =========================================================
MODEL_DIR = './architectureandweights'  # Model folder root
SEX_JSON = os.path.join(MODEL_DIR, 'sex_architecture.json')
SEX_WEIGHTS = os.path.join(MODEL_DIR, 'sex_cnn_weights121.h5')
GENO_JSON = os.path.join(MODEL_DIR, 'geno_architecture.json')
GENO_WEIGHTS = os.path.join(MODEL_DIR, 'geno_weights.h5')

# =========================================================
#  back-end logic classes 
# =========================================================

class ImageProcessor:
    def __init__(self):
        self.target_canvas = (1360, 1024)
        self.bg_color = (255, 255, 255)
        self.crop_w = 1280
        self.crop_h = 1024
        self.downsample_step = 2
        self.gaussian_sigma = 1.0
        self.meijering_range = range(2, 8)
        self.min_obj_size = 200

    def process_to_binary(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            target_w, target_h = self.target_canvas
            w, h = img.size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = img.resize((new_w, new_h), Image.LANCZOS)
            img_padded = Image.new("RGB", (target_w, target_h), self.bg_color)
            img_padded.paste(img_resized, ((target_w - new_w)//2, (target_h - new_h)//2))

            img_gray = img_padded.convert("L")
            left = (img_gray.width - self.crop_w) // 2
            img_cropped = img_gray.crop((left, 0, left + self.crop_w, self.crop_h))

            img_arr = np.array(img_cropped)
            img_down = img_arr[::self.downsample_step, ::self.downsample_step]
            
            img_float = img_down.astype("float32")
            img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)
            smooth = gaussian(img_float, sigma=self.gaussian_sigma)
            
            vein_resp = meijering(smooth, sigmas=self.meijering_range, black_ridges=True)
            vein_resp = exposure.rescale_intensity(vein_resp, out_range=(0, 1))
            
            try:
                th_val = threshold_otsu(vein_resp)
            except:
                th_val = 0.5
            
            binary = vein_resp > (th_val * 0.1)
            binary = binary_closing(binary, disk(2))
            clean = morphology.remove_small_objects(binary, min_size=self.min_obj_size)

            return (clean.astype(np.uint8) * 255)
        except Exception as e:
            print(f"Image processing fails: {e}")
            return None

class SexPredictor:
    def __init__(self, json_path, weights_path):
        self.model = None
        self.class_names = ["Female", "Male"]
        self.target_size = (320, 256)
        try:
            with open(json_path, 'r') as f:
                model_json = f.read()
            self.model = model_from_json(model_json)
            self.model.load_weights(weights_path)
        except Exception as e:
            raise Exception(f"Gender model loading failed: {e}")

    def predict(self, binary_img_arr):
        if self.model is None: return "Error", 0.0, [0,0]
        pil_img = Image.fromarray(binary_img_arr)
        pil_resized = pil_img.resize(self.target_size, Image.BILINEAR)
        img_arr = np.array(pil_resized)
        img_norm = img_arr.astype("float32") / 255.0
        img_input = np.expand_dims(img_norm, axis=-1)
        img_input = np.expand_dims(img_input, axis=0)
        probs = self.model.predict(img_input, verbose=0)[0]
        label = self.class_names[np.argmax(probs)]
        conf = np.max(probs)
        return label, conf, probs

class GenotypePredictor:
    def __init__(self, json_path, weights_path):
        self.model = None
        self.class_names = ["egfr", "mam", "samw", "star", "tkv"]
        self.target_size = None
        try:
            with open(json_path, 'r') as f:
                model_json = f.read()
            self.model = model_from_json(model_json)
            self.model.load_weights(weights_path)
            input_shape = self.model.input_shape
            if input_shape is not None:
                h, w = input_shape[1], input_shape[2]
                self.target_size = (w, h)
            else:
                self.target_size = (320, 256)
        except Exception as e:
            raise Exception(f"Genetic model loading failed: {e}")

    def predict(self, binary_img_arr):
        if self.model is None: return "Error", 0.0, []
        pil_img = Image.fromarray(binary_img_arr)
        if self.target_size:
            pil_resized = pil_img.resize(self.target_size, Image.BILINEAR)
            img_resized_arr = np.array(pil_resized)
        else:
            img_resized_arr = binary_img_arr
        
        img_float = img_resized_arr.astype("float32")
        img_rgb = np.repeat(img_float[..., np.newaxis], 3, axis=-1)
        img_input = np.expand_dims(img_rgb, axis=0)
        img_preprocessed = resnet_preprocess(img_input)
        
        probs = self.model.predict(img_preprocessed, verbose=0)[0]
        label = self.class_names[np.argmax(probs)]
        conf = np.max(probs)
        return label, conf, probs

# =========================================================
#  multithreading Worker
# =========================================================
class BatchWorker(QThread):
    progress_signal = pyqtSignal(int)     
    log_signal = pyqtSignal(str)          
    result_signal = pyqtSignal(dict)       
    finished_signal = pyqtSignal()       

    def __init__(self, folder_path, processor, sex_predictor, geno_predictor):
        super().__init__()
        self.folder_path = folder_path
        self.processor = processor
        self.sex_predictor = sex_predictor
        self.geno_predictor = geno_predictor
        self.is_running = True

    def run(self):
        valid_exts = {'.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp'}
        files = [f for f in os.listdir(self.folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
        total = len(files)
        
        if total == 0:
            self.log_signal.emit("Error: The picture file is not found in the folder。")
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"Started to deal with it, and found a total of {total} images...")

        for i, filename in enumerate(files):
            if not self.is_running: break
            
            full_path = os.path.join(self.folder_path, filename)
            try:
                # 1. dispose
                binary_data = self.processor.process_to_binary(full_path)
                if binary_data is not None:
                    # 2. predict
                    s_label, s_conf, _ = self.sex_predictor.predict(binary_data)
                    g_label, g_conf, _ = self.geno_predictor.predict(binary_data)
                    
                    # 3. send outcome 
                    res = {
                        "filename": filename,
                        "Gender": s_label,
                        "Gender confidence": f"{s_conf:.2%}",
                        "genotype": g_label,
                        "genotype confidence": f"{g_conf:.2%}",
                        "path": full_path
                    }
                    self.result_signal.emit(res)
                else:
                    self.log_signal.emit(f"fail: {filename} (Preprocessing failed)")
            except Exception as e:
                self.log_signal.emit(f"error: {filename} ({str(e)})")

            #progress updates
            progress = int((i + 1) / total * 100)
            self.progress_signal.emit(progress)

        self.finished_signal.emit()

    def stop(self):
        self.is_running = False

# =========================================================
#  GUI Main window
# =========================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("An Intelligent Identification System of Drosophila Sex and Genotype ranrukaifa")
        self.resize(1000, 700)
        self.setAcceptDrops(True) 

        # Initialize the backend objects.
        self.processor = ImageProcessor()
        self.sex_model = None
        self.geno_model = None
        
        # UI layout
        self.init_ui()
        
        # Delayed loading model (avoid slow startup)
        QThread.msleep(100)
        self.load_models()

    def init_ui(self):
        # Main pendant
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Top: Status information
        self.status_label = QLabel("The system is being initialized...")
        self.status_label.setStyleSheet("color: blue; font-weight: bold;")
        layout.addWidget(self.status_label)

        # Middle: Tab
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Tab 1: Single-image prediction ---
        self.tab_single = QWidget()
        self.init_tab_single()
        self.tabs.addTab(self.tab_single, "Single-image prediction & reporting")

        # --- Tab 2: Batch processing ---
        self.tab_batch = QWidget()
        self.init_tab_batch()
        self.tabs.addTab(self.tab_batch, "Batch processing")

    def init_tab_single(self):
        layout = QHBoxLayout(self.tab_single)

        # Left side: Image display area
        left_panel = QVBoxLayout()
        self.img_label = QLabel("Please drag in the image or click the button below to import")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 2px dashed gray; background-color: #f0f0f0;")
        self.img_label.setFixedSize(500, 400)
        left_panel.addWidget(self.img_label)

        btn_load = QPushButton("choose images")
        btn_load.clicked.connect(self.load_single_image_dialog)
        left_panel.addWidget(btn_load)
        
        layout.addLayout(left_panel)

        # Right side: Result display area
        right_panel = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("The forecast results will be displayed here....")
        self.result_text.setStyleSheet("font-size: 14px;")
        right_panel.addWidget(self.result_text)

        self.btn_save_single = QPushButton("Save the current results to Excel")
        self.btn_save_single.setEnabled(False)
        self.btn_save_single.clicked.connect(self.save_single_result)
        right_panel.addWidget(self.btn_save_single)

        layout.addLayout(right_panel)
        
        # Temporarily store the results of a single image processing task
        self.current_single_result = None

    def init_tab_batch(self):
        layout = QVBoxLayout(self.tab_batch)

        # Top Operation Bar
        h_layout = QHBoxLayout()
        self.path_input = QTextEdit()
        self.path_input.setFixedHeight(30)
        self.path_input.setPlaceholderText("Please select the folder path...")
        h_layout.addWidget(self.path_input)

        btn_browse = QPushButton("Browse folders")
        btn_browse.clicked.connect(self.browse_folder)
        h_layout.addWidget(btn_browse)

        self.btn_start_batch = QPushButton("Starting batch processing")
        self.btn_start_batch.clicked.connect(self.start_batch_processing)
        self.btn_start_batch.setEnabled(False) # The model is only available after it has been loaded.
        h_layout.addWidget(self.btn_start_batch)
        
        layout.addLayout(h_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # The table displays the results
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Filename", "Gender", "Gender Confidence", "Genotype", "Genotype Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Bottom button
        self.btn_save_batch = QPushButton("Export all results to Excel")
        self.btn_save_batch.clicked.connect(self.save_batch_results)
        self.btn_save_batch.setEnabled(False)
        layout.addWidget(self.btn_save_batch)

        # Batch data storage
        self.batch_results_data = []

    def load_models(self):
        """Loading the model"""
        try:
            self.status_label.setText("Loading deep learning model, please wait...")
            QApplication.processEvents() 

            if not os.path.exists(SEX_WEIGHTS) or not os.path.exists(GENO_WEIGHTS):
                raise FileNotFoundError("Can't find the model file, check the path configuration！")

            self.sex_model = SexPredictor(SEX_JSON, SEX_WEIGHTS)
            self.geno_model = GenotypePredictor(GENO_JSON, GENO_WEIGHTS)

            self.status_label.setText("Model loading complete! system ready")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.btn_start_batch.setEnabled(True)
            
        except Exception as e:
            self.status_label.setText(f"error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Model loading failed", str(e))

    # ================= Single-image logic =================
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if self.tabs.currentIndex() != 0: return
        
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if self.is_image_file(file_path):
                self.process_single_image(file_path)
            else:
                QMessageBox.warning(self, "Format error", "Unsupported file formats! Please drag in the image")

    def is_image_file(self, path):
        return path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))

    def load_single_image_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'choose image', '.', "Image files (*.jpg *.png *.tif *.tiff *.bmp)")
        if fname:
            self.process_single_image(fname)

    def process_single_image(self, file_path):
        if not self.sex_model or not self.geno_model:
            QMessageBox.warning(self, "not ready", "The model has not yet been fully loaded！")
            return

        # Display the image
        pixmap = QPixmap(file_path)
        self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        self.result_text.setText("Analyzing...")
        QApplication.processEvents()

        try:
            # 1. Preprocessing
            binary_data = self.processor.process_to_binary(file_path)
            
            if binary_data is not None:
                # 2. Prediction
                s_label, s_conf, s_probs = self.sex_model.predict(binary_data)
                g_label, g_conf, g_probs = self.geno_model.predict(binary_data)

                # 3. Results display
                report = f"【file】: {os.path.basename(file_path)}\n"
                report += "-"*30 + "\n"
                report += f"【Gender prediction】: {s_label}\n"
                report += f"  - Confidence: {s_conf:.2%}\n"
                report += f"  - Probability: F={s_probs[0]:.4f}, M={s_probs[1]:.4f}\n\n"
                report += f"【Genotype prediction】: {g_label}\n"
                report += f"  - Confidence: {g_conf:.2%}\n"
                report += f"  - Probability: {np.round(g_probs, 4)}\n"
                
                self.result_text.setText(report)
                
                # The storage results are used for preservation
                self.current_single_result = {
                    "Filename": os.path.basename(file_path),
                    "Gender": s_label,
                    "Gender prediction": s_conf,
                    "Genotype": g_label,
                    "Genotype prediction": g_conf,
                    "Time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                self.btn_save_single.setEnabled(True)
            else:
                self.result_text.setText("The image preprocessing fails and the outline cannot be generated")
                self.btn_save_single.setEnabled(False)

        except Exception as e:
            self.result_text.setText(f"analysis error:\n{str(e)}")
            self.btn_save_single.setEnabled(False)

    def save_single_result(self):
        if not self.current_single_result: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save results", "result.xlsx", "Excel Files (*.xlsx)")
        if path:
            try:
                df = pd.DataFrame([self.current_single_result])
                df.to_excel(path, index=False)
                QMessageBox.information(self, "succeed", "The file has been saved！")
            except Exception as e:
                QMessageBox.critical(self, "failed", f"Save error: {str(e)}")

    # ================= Batch logic =================
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder")
        if folder:
            self.path_input.setText(folder)

    def start_batch_processing(self):
        folder = self.path_input.toPlainText().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "pathway error", "Please select a valid folder path！")
            return
        
        # Clear the table
        self.table.setRowCount(0)
        self.batch_results_data = []
        self.btn_save_batch.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # Disable buttons to prevent repeated clicks
        self.btn_start_batch.setEnabled(False)
        self.status_label.setText("Batch processing is in progress...")

        # 启动线程
        self.worker = BatchWorker(folder, self.processor, self.sex_model, self.geno_model)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.log_signal.connect(lambda msg: self.status_label.setText(msg))
        self.worker.result_signal.connect(self.add_batch_result)
        self.worker.finished_signal.connect(self.batch_finished)
        self.worker.start()

    def add_batch_result(self, res):
        self.batch_results_data.append(res)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(res['Filename']))
        self.table.setItem(row, 1, QTableWidgetItem(res['Gender']))
        self.table.setItem(row, 2, QTableWidgetItem(res['Gender prediction']))
        self.table.setItem(row, 3, QTableWidgetItem(res['Genotype']))
        self.table.setItem(row, 4, QTableWidgetItem(res['Genotype prediction']))
        
        # Automatically scroll to the bottom
        self.table.scrollToBottom()

    def batch_finished(self):
        self.btn_start_batch.setEnabled(True)
        self.btn_save_batch.setEnabled(True)
        self.status_label.setText("The batch processing is complete！")
        QMessageBox.information(self, "complete", f"deal with {len(self.batch_results_data)} images")

    def save_batch_results(self):
        if not self.batch_results_data: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save the batch results", "batch_results.xlsx", "Excel Files (*.xlsx)")
        if path:
            try:
                df = pd.DataFrame(self.batch_results_data)
                df.to_excel(path, index=False)
                QMessageBox.information(self, "success", "The batch results have been saved！")
            except Exception as e:
                QMessageBox.critical(self, "error", f"Saving failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
