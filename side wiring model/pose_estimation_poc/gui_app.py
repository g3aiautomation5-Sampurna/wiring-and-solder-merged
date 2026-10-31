"""
Wire Harness Pose Inspection GUI (PyQt6)
Allows uploading images, folder batch testing, drag-and-drop,
confidence adjustment, and real-time topological verification.
"""

import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QSlider, QSplitter, QFrame, QScrollArea, QProgressBar,
    QSizePolicy, QMessageBox, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QColor, QPainter, QPen

# Import the inspection engine
from ultralytics import YOLO
from inspect_pose import inspect_image

class InspectionWorker(QThread):
    finished_single = pyqtSignal(dict, str)
    finished_batch = pyqtSignal(list)
    progress_batch = pyqtSignal(int, int)

    def __init__(self, model, file_paths, conf=0.35):
        super().__init__()
        self.model = model
        self.file_paths = file_paths
        self.conf = conf
        self.is_single = len(file_paths) == 1

    def run(self):
        results = []
        total = len(self.file_paths)
        for idx, fpath in enumerate(self.file_paths):
            res = inspect_image(self.model, fpath, conf=self.conf, return_details=True)
            res["image_path"] = fpath
            results.append(res)
            self.progress_batch.emit(idx + 1, total)

        if self.is_single:
            self.finished_single.emit(results[0], self.file_paths[0])
        else:
            self.finished_batch.emit(results)


class DropImageLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_pixmap = None
        self.setStyleSheet("""
            QLabel {
                background-color: #121418;
                border: 2px dashed #2d3748;
                border-radius: 8px;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    background-color: #1a202c;
                    border: 2px dashed #4299e1;
                    border-radius: 8px;
                }
            """)

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                background-color: #121418;
                border: 2px dashed #2d3748;
                border-radius: 8px;
            }
        """)

    def dropEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                background-color: #121418;
                border: 2px dashed #2d3748;
                border-radius: 8px;
            }
        """)
        for url in event.mimeData().urls():
            fpath = url.toLocalFile()
            if fpath.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.file_dropped.emit(fpath)
                break

    def set_display_pixmap(self, pixmap):
        self.current_pixmap = pixmap
        self.update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap:
            self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            scaled = self.current_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)


class WireInspectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wire Harness Pose Estimation & Inspection POC")
        self.resize(1280, 840)
        self.setMinimumSize(1000, 680)

        self.model = None
        self.loaded_files = []
        self.current_file_idx = -1
        self.inspection_cache = {}  # fpath -> result dict
        self.view_mode = "annotated"  # "annotated" or "original"

        self.init_model()
        self.setup_ui()
        self.apply_dark_theme()

        # Load test samples automatically if available
        self.load_default_test_samples()

    def init_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "runs", "wire_pose_model", "weights", "best.pt")
        if not os.path.exists(model_path):
            model_path = "yolov8n-pose.pt"

        try:
            self.model = YOLO(model_path)
            self.model_status_text = f"Weights: {os.path.basename(model_path)}"
        except Exception as e:
            self.model = None
            self.model_status_text = f"Model load error: {e}"

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)

        # ---------------- HEADER BAR ----------------
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("background-color: #1a202c; border-radius: 6px; padding: 4px 12px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)

        title_lbl = QLabel("WIRE HARNESS POSE ESTIMATION & INSPECTION")
        title_lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title_lbl.setStyleSheet("color: #edf2f7;")

        self.model_badge = QLabel(self.model_status_text)
        self.model_badge.setStyleSheet("color: #68d391; font-weight: bold; background: #22543d; padding: 4px 10px; border-radius: 4px;")

        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.model_badge)
        main_layout.addWidget(header)

        # ---------------- MAIN SPLITTER ----------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # LEFT SIDEBAR: Controls & Results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(10)

        # Group 1: File Actions
        file_box = QFrame()
        file_box.setStyleSheet("background-color: #1e2430; border-radius: 6px; padding: 8px;")
        file_box_layout = QVBoxLayout(file_box)
        file_box_layout.setSpacing(6)

        btn_row = QHBoxLayout()
        self.btn_browse = QPushButton("📁 Upload Image(s)")
        self.btn_browse.clicked.connect(self.on_browse_images)
        self.btn_browse.setStyleSheet(self.btn_style_primary())

        self.btn_folder = QPushButton("📂 Open Folder")
        self.btn_folder.clicked.connect(self.on_browse_folder)
        self.btn_folder.setStyleSheet(self.btn_style_secondary())

        btn_row.addWidget(self.btn_browse)
        btn_row.addWidget(self.btn_folder)
        file_box_layout.addLayout(btn_row)

        self.btn_samples = QPushButton("⚡ Quick Load Test Samples")
        self.btn_samples.clicked.connect(self.load_default_test_samples)
        self.btn_samples.setStyleSheet(self.btn_style_accent())
        file_box_layout.addWidget(self.btn_samples)

        left_layout.addWidget(file_box)

        # Group 2: File List & Navigation
        list_box = QFrame()
        list_box.setStyleSheet("background-color: #1e2430; border-radius: 6px; padding: 8px;")
        list_box_layout = QVBoxLayout(list_box)
        list_box_layout.setSpacing(6)

        list_header = QHBoxLayout()
        list_lbl = QLabel("Loaded Files")
        list_lbl.setStyleSheet("font-weight: bold; color: #a0aec0;")
        self.file_count_lbl = QLabel("(0)")
        self.file_count_lbl.setStyleSheet("color: #718096;")
        list_header.addWidget(list_lbl)
        list_header.addWidget(self.file_count_lbl)
        list_header.addStretch()
        list_box_layout.addLayout(list_header)

        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #121418;
                border: 1px solid #2d3748;
                border-radius: 4px;
                color: #e2e8f0;
            }
            QListWidget::item:selected {
                background-color: #2b6cb0;
                color: white;
            }
        """)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        list_box_layout.addWidget(self.file_list)

        # Nav Buttons
        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.on_prev_file)
        self.btn_prev.setStyleSheet(self.btn_style_secondary())

        self.nav_status = QLabel("0 / 0")
        self.nav_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.nav_status.setStyleSheet("color: #cbd5e0; font-weight: bold;")

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.on_next_file)
        self.btn_next.setStyleSheet(self.btn_style_secondary())

        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.nav_status)
        nav_row.addWidget(self.btn_next)
        list_box_layout.addLayout(nav_row)

        left_layout.addWidget(list_box)

        # Group 3: Confidence Slider & Action
        conf_box = QFrame()
        conf_box.setStyleSheet("background-color: #1e2430; border-radius: 6px; padding: 8px;")
        conf_box_layout = QVBoxLayout(conf_box)
        conf_box_layout.setSpacing(4)

        conf_header = QHBoxLayout()
        conf_lbl = QLabel("Confidence Threshold:")
        conf_lbl.setStyleSheet("color: #a0aec0;")
        self.conf_val_lbl = QLabel("0.35")
        self.conf_val_lbl.setStyleSheet("color: #63b3ed; font-weight: bold;")
        conf_header.addWidget(conf_lbl)
        conf_header.addWidget(self.conf_val_lbl)
        conf_header.addStretch()
        conf_box_layout.addLayout(conf_header)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(35)
        self.conf_slider.valueChanged.connect(self.on_conf_changed)
        conf_box_layout.addWidget(self.conf_slider)

        action_row = QHBoxLayout()
        self.btn_test_current = QPushButton("🚀 Test Image")
        self.btn_test_current.clicked.connect(self.on_test_current)
        self.btn_test_current.setStyleSheet(self.btn_style_primary())

        self.btn_test_all = QPushButton("▶▶ Test All")
        self.btn_test_all.clicked.connect(self.on_test_all)
        self.btn_test_all.setStyleSheet(self.btn_style_secondary())

        action_row.addWidget(self.btn_test_current)
        action_row.addWidget(self.btn_test_all)
        conf_box_layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d3748;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #48bb78;
                border-radius: 4px;
            }
        """)
        conf_box_layout.addWidget(self.progress_bar)

        left_layout.addWidget(conf_box)

        # Group 4: Verdict & Rules Breakdown
        verdict_box = QFrame()
        verdict_box.setStyleSheet("background-color: #1e2430; border-radius: 6px; padding: 10px;")
        self.verdict_box_layout = QVBoxLayout(verdict_box)
        self.verdict_box_layout.setSpacing(8)

        # Large Banner
        self.verdict_badge = QLabel("READY TO TEST")
        self.verdict_badge.setFixedHeight(48)
        self.verdict_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_badge.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.verdict_badge.setStyleSheet("""
            background-color: #2d3748;
            color: #e2e8f0;
            border-radius: 6px;
        """)
        self.verdict_box_layout.addWidget(self.verdict_badge)

        self.latency_lbl = QLabel("Latency: -- ms")
        self.latency_lbl.setStyleSheet("color: #718096; font-size: 11px;")
        self.verdict_box_layout.addWidget(self.latency_lbl)

        # Regional cards
        self.rule_um_lbl = QLabel("• Upper Middle: Waiting...")
        self.rule_um_lbl.setStyleSheet("color: #a0aec0; font-size: 11px;")
        self.rule_um_lbl.setWordWrap(True)
        self.verdict_box_layout.addWidget(self.rule_um_lbl)

        self.rule_bl_lbl = QLabel("• Bottom Left: Waiting...")
        self.rule_bl_lbl.setStyleSheet("color: #a0aec0; font-size: 11px;")
        self.rule_bl_lbl.setWordWrap(True)
        self.verdict_box_layout.addWidget(self.rule_bl_lbl)

        self.rule_br_lbl = QLabel("• Bottom Right: Waiting...")
        self.rule_br_lbl.setStyleSheet("color: #a0aec0; font-size: 11px;")
        self.rule_br_lbl.setWordWrap(True)
        self.verdict_box_layout.addWidget(self.rule_br_lbl)

        left_layout.addWidget(verdict_box)
        left_layout.addStretch()

        left_panel.setFixedWidth(340)
        splitter.addWidget(left_panel)

        # RIGHT VIEW CANVAS
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        # Canvas Top Control Bar
        canvas_bar = QFrame()
        canvas_bar.setFixedHeight(40)
        canvas_bar.setStyleSheet("background-color: #1e2430; border-radius: 6px; padding: 4px 8px;")
        canvas_bar_layout = QHBoxLayout(canvas_bar)
        canvas_bar_layout.setContentsMargins(8, 0, 8, 0)

        self.current_filename_lbl = QLabel("No image loaded")
        self.current_filename_lbl.setStyleSheet("color: #e2e8f0; font-weight: bold;")
        canvas_bar_layout.addWidget(self.current_filename_lbl)
        canvas_bar_layout.addStretch()

        # View Mode Switch
        view_lbl = QLabel("View:")
        view_lbl.setStyleSheet("color: #a0aec0;")
        canvas_bar_layout.addWidget(view_lbl)

        self.radio_annotated = QRadioButton("Annotated Pose")
        self.radio_annotated.setChecked(True)
        self.radio_annotated.toggled.connect(self.on_view_mode_changed)
        self.radio_annotated.setStyleSheet("color: #edf2f7;")

        self.radio_orig = QRadioButton("Original Image")
        self.radio_orig.setStyleSheet("color: #edf2f7;")

        canvas_bar_layout.addWidget(self.radio_annotated)
        canvas_bar_layout.addWidget(self.radio_orig)

        self.btn_save_res = QPushButton("💾 Save Image")
        self.btn_save_res.clicked.connect(self.on_save_inspected_image)
        self.btn_save_res.setStyleSheet(self.btn_style_secondary())
        canvas_bar_layout.addWidget(self.btn_save_res)

        right_layout.addWidget(canvas_bar)

        # Main Drag-and-Drop Image Canvas
        self.image_canvas = DropImageLabel()
        self.image_canvas.file_dropped.connect(self.on_image_dropped)
        self.image_canvas.setText("Drag and drop images here\nor click 'Upload Image(s)'")
        self.image_canvas.setFont(QFont("Segoe UI", 12))
        right_layout.addWidget(self.image_canvas, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

    # ---------------- STYLES ----------------
    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121418;
            }
            QLabel {
                color: #e2e8f0;
            }
            QScrollBar:vertical {
                background: #1a202c;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #4a5568;
                border-radius: 4px;
            }
        """)

    def btn_style_primary(self):
        return """
            QPushButton {
                background-color: #3182ce;
                color: white;
                font-weight: bold;
                border: none;
                padding: 7px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2b6cb0;
            }
            QPushButton:pressed {
                background-color: #2c5282;
            }
        """

    def btn_style_secondary(self):
        return """
            QPushButton {
                background-color: #2d3748;
                color: #edf2f7;
                font-weight: 500;
                border: 1px solid #4a5568;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a5568;
            }
            QPushButton:pressed {
                background-color: #1a202c;
            }
        """

    def btn_style_accent(self):
        return """
            QPushButton {
                background-color: #2b6cb0;
                color: white;
                font-weight: 600;
                border: none;
                padding: 7px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #2c5282;
            }
        """

    # ---------------- HANDLERS ----------------
    def load_default_test_samples(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(script_dir, "test_samples")
        if os.path.exists(test_dir):
            files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png'))]
            if files:
                self.set_loaded_files(files)

    def on_browse_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images to Inspect", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if files:
            self.set_loaded_files(files)

    def on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images")
        if folder:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if files:
                self.set_loaded_files(files)
            else:
                QMessageBox.information(self, "No Images", "No supported image files found in the selected folder.")

    def on_image_dropped(self, fpath):
        if fpath not in self.loaded_files:
            self.loaded_files.append(fpath)
            self.update_file_list()
        self.current_file_idx = self.loaded_files.index(fpath)
        self.file_list.setCurrentRow(self.current_file_idx)
        self.run_single_inspection(fpath)

    def set_loaded_files(self, files):
        self.loaded_files = sorted(files)
        self.inspection_cache.clear()
        self.update_file_list()
        if self.loaded_files:
            self.current_file_idx = 0
            self.file_list.setCurrentRow(0)
            self.on_test_current()

    def update_file_list(self):
        self.file_list.clear()
        for idx, f in enumerate(self.loaded_files):
            bname = os.path.basename(f)
            status_prefix = "⚪ "
            if f in self.inspection_cache:
                status_prefix = "🟢 " if self.inspection_cache[f]["is_pass"] else "🔴 "
            item = QListWidgetItem(f"{status_prefix}{bname}")
            self.file_list.addItem(item)

        self.file_count_lbl.setText(f"({len(self.loaded_files)})")
        self.update_nav_status()

    def update_nav_status(self):
        total = len(self.loaded_files)
        curr = (self.current_file_idx + 1) if total > 0 else 0
        self.nav_status.setText(f"{curr} / {total}")
        self.btn_prev.setEnabled(self.current_file_idx > 0)
        self.btn_next.setEnabled(self.current_file_idx < total - 1)

    def on_file_selected(self, row):
        if 0 <= row < len(self.loaded_files):
            self.current_file_idx = row
            fpath = self.loaded_files[row]
            self.current_filename_lbl.setText(os.path.basename(fpath))
            self.update_nav_status()

            if fpath in self.inspection_cache:
                self.display_inspection_result(self.inspection_cache[fpath])
            else:
                # Display original image until test button is pressed
                img = cv2.imread(fpath)
                if img is not None:
                    self.show_image_on_canvas(img)
                self.reset_verdict_display()

    def on_prev_file(self):
        if self.current_file_idx > 0:
            self.file_list.setCurrentRow(self.current_file_idx - 1)

    def on_next_file(self):
        if self.current_file_idx < len(self.loaded_files) - 1:
            self.file_list.setCurrentRow(self.current_file_idx + 1)

    def on_conf_changed(self, val):
        conf_float = val / 100.0
        self.conf_val_lbl.setText(f"{conf_float:.2f}")

    def on_view_mode_changed(self):
        self.view_mode = "annotated" if self.radio_annotated.isChecked() else "original"
        if 0 <= self.current_file_idx < len(self.loaded_files):
            fpath = self.loaded_files[self.current_file_idx]
            if fpath in self.inspection_cache:
                res = self.inspection_cache[fpath]
                target_img = res["annotated_img"] if self.view_mode == "annotated" else res["orig_img"]
                self.show_image_on_canvas(target_img)

    def on_test_current(self):
        if 0 <= self.current_file_idx < len(self.loaded_files):
            fpath = self.loaded_files[self.current_file_idx]
            self.run_single_inspection(fpath)

    def run_single_inspection(self, fpath):
        if not self.model:
            QMessageBox.warning(self, "No Model", "YOLO model could not be loaded.")
            return

        conf = self.conf_slider.value() / 100.0
        self.verdict_badge.setText("INSPECTING...")
        self.verdict_badge.setStyleSheet("background-color: #2b6cb0; color: white; border-radius: 6px;")

        self.worker = InspectionWorker(self.model, [fpath], conf=conf)
        self.worker.finished_single.connect(self.on_single_done)
        self.worker.start()

    def on_single_done(self, result, fpath):
        self.inspection_cache[fpath] = result
        self.update_file_list_item(fpath, result["is_pass"])
        if 0 <= self.current_file_idx < len(self.loaded_files):
            if self.loaded_files[self.current_file_idx] == fpath:
                self.display_inspection_result(result)

    def on_test_all(self):
        if not self.loaded_files:
            return
        conf = self.conf_slider.value() / 100.0
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_test_all.setEnabled(False)

        self.batch_worker = InspectionWorker(self.model, self.loaded_files, conf=conf)
        self.batch_worker.progress_batch.connect(self.on_batch_progress)
        self.batch_worker.finished_batch.connect(self.on_batch_done)
        self.batch_worker.start()

    def on_batch_progress(self, curr, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)

    def on_batch_done(self, results):
        self.progress_bar.setVisible(False)
        self.btn_test_all.setEnabled(True)

        for res in results:
            fpath = res["image_path"]
            self.inspection_cache[fpath] = res
            self.update_file_list_item(fpath, res["is_pass"])

        # Display current
        if 0 <= self.current_file_idx < len(self.loaded_files):
            fpath = self.loaded_files[self.current_file_idx]
            if fpath in self.inspection_cache:
                self.display_inspection_result(self.inspection_cache[fpath])

        # Summary box
        passes = sum(1 for r in results if r["is_pass"])
        fails = len(results) - 1 if False else (len(results) - passes)
        rate = (passes / len(results)) * 100 if len(results) else 0
        QMessageBox.information(
            self, "Batch Inspection Complete",
            f"Processed {len(results)} images.\n\n"
            f"✔ PASS: {passes}\n"
            f"✘ FAIL: {fails}\n"
            f"Pass Rate: {rate:.1f}%"
        )

    def update_file_list_item(self, fpath, is_pass):
        if fpath in self.loaded_files:
            idx = self.loaded_files.index(fpath)
            prefix = "🟢 " if is_pass else "🔴 "
            bname = os.path.basename(fpath)
            item = self.file_list.item(idx)
            if item:
                item.setText(f"{prefix}{bname}")

    def reset_verdict_display(self):
        self.verdict_badge.setText("READY TO TEST")
        self.verdict_badge.setStyleSheet("background-color: #2d3748; color: #e2e8f0; border-radius: 6px;")
        self.latency_lbl.setText("Latency: -- ms")
        self.rule_um_lbl.setText("• Upper Middle: Waiting...")
        self.rule_bl_lbl.setText("• Bottom Left: Waiting...")
        self.rule_br_lbl.setText("• Bottom Right: Waiting...")

    def display_inspection_result(self, result):
        is_pass = result["is_pass"]
        if is_pass:
            self.verdict_badge.setText("OVERALL: PASS")
            self.verdict_badge.setStyleSheet("""
                background-color: #22543d;
                color: #68d391;
                font-weight: bold;
                border-radius: 6px;
                border: 2px solid #38a169;
            """)
        else:
            self.verdict_badge.setText("OVERALL: FAIL")
            self.verdict_badge.setStyleSheet("""
                background-color: #742a2a;
                color: #feb2b2;
                font-weight: bold;
                border-radius: 6px;
                border: 2px solid #e53e3e;
            """)

        self.latency_lbl.setText(f"Latency: {result['latency_ms']:.1f} ms | Wires Detected: {result.get('total_wires', 0)}")

        # Format regions
        regions = result.get("regions", {})
        um = regions.get("UM", {})
        bl = regions.get("BL", {})
        br = regions.get("BR", {})

        def fmt_rule(reg, name):
            if not reg:
                return f"• {name}: Not evaluated"
            p = reg.get("pass", False)
            sym = "✔ PASS: " if p else "✘ FAIL: "
            color = "#68d391" if p else "#fc8181"
            msg = reg.get("msg", "")
            return f"<span style='color:{color}; font-weight:bold;'>{sym}</span><span style='color:#e2e8f0;'>{msg}</span>"

        self.rule_um_lbl.setText(fmt_rule(um, "Upper Middle"))
        self.rule_bl_lbl.setText(fmt_rule(bl, "Bottom Left"))
        self.rule_br_lbl.setText(fmt_rule(br, "Bottom Right"))

        # Render on Canvas
        target_img = result["annotated_img"] if self.view_mode == "annotated" else result["orig_img"]
        self.show_image_on_canvas(target_img)

    def show_image_on_canvas(self, bgr_img):
        if bgr_img is None:
            return
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_canvas.set_display_pixmap(pix)

    def on_save_inspected_image(self):
        if 0 <= self.current_file_idx < len(self.loaded_files):
            fpath = self.loaded_files[self.current_file_idx]
            if fpath in self.inspection_cache:
                res = self.inspection_cache[fpath]
                img = res["annotated_img"]
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Inspected Image", f"inspected_{os.path.basename(fpath)}", "JPEG (*.jpg);;PNG (*.png)"
                )
                if save_path:
                    cv2.imwrite(save_path, img)
                    QMessageBox.information(self, "Saved", f"Image saved successfully to:\n{save_path}")
            else:
                QMessageBox.information(self, "Notice", "Please test the image first before saving.")


def main():
    app = QApplication(sys.argv)
    window = WireInspectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
