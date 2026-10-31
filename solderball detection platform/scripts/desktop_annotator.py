"""
Native PyQt5 Offline Solder Ball Annotator
Desktop GUI application for annotating solder balls directly on Windows.

Usage:
    python scripts/desktop_annotator.py
"""

import os
import glob
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QPushButton, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsItem, QSlider, QStatusBar, QShortcut,
    QMessageBox
)
from PyQt5.QtGui import QPixmap, QPen, QBrush, QColor, QFont, QKeySequence, QPainter
from PyQt5.QtCore import Qt, QRectF, QPointF

class AnnotatorGraphicsView(QGraphicsView):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.drawing = False
        self.start_point = None
        self.temp_rect_item = None

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton or event.modifiers() == Qt.ShiftModifier:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if self.parent_window.mode == 'click':
                # Quick click: create box of fixed size
                r = self.parent_window.click_radius
                rect = QRectF(scene_pos.x() - r, scene_pos.y() - r, r * 2, r * 2)
                self.parent_window.add_box(rect)
            else:
                # Drag rectangle
                self.drawing = True
                self.start_point = scene_pos
                pen = QPen(QColor(0, 200, 255), 2, Qt.DashLine)
                self.temp_rect_item = self.scene().addRect(QRectF(self.start_point, self.start_point), pen)

    def mouseMoveEvent(self, event):
        if self.drawing and self.temp_rect_item:
            current_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_pos).normalized()
            self.temp_rect_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton or self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.temp_rect_item:
                rect = self.temp_rect_item.rect().normalized()
                self.scene().removeItem(self.temp_rect_item)
                self.temp_rect_item = None
                if rect.width() >= 4 and rect.height() >= 4:
                    self.parent_window.add_box(rect)

class DesktopAnnotatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Solder Ball Offline Annotator (Desktop GUI)")
        self.resize(1300, 850)

        self.dataset_dir = os.path.abspath('dataset')
        self.ann_dir = os.path.abspath('annotations')
        os.makedirs(self.ann_dir, exist_ok=True)

        self.image_files = sorted(glob.glob(os.path.join(self.dataset_dir, '*.jpg')) +
                                  glob.glob(os.path.join(self.dataset_dir, '*.png')))

        self.current_idx = 0
        self.current_img_pixmap = None
        self.boxes = [] # List of QRectF
        self.box_items = []
        self.mode = 'drag' # 'drag' or 'click'
        self.click_radius = 20

        self.init_ui()
        if self.image_files:
            self.load_image(0)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel: Image list & stats
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(260)

        left_layout.addWidget(QLabel("<b>PCB Images (Offline)</b>"))
        self.progress_lbl = QLabel(f"0 / {len(self.image_files)} Annotated")
        left_layout.addWidget(self.progress_lbl)

        self.list_widget = QListWidget()
        for f in self.image_files:
            self.list_widget.addItem(os.path.basename(f))
        self.list_widget.currentRowChanged.connect(self.on_image_selected)
        left_layout.addWidget(self.list_widget)

        main_layout.addWidget(left_panel)

        # Right panel: Toolbar, Canvas, Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Toolbar
        toolbar = QHBoxLayout()

        self.btn_prev = QPushButton("◀ Prev (A)")
        self.btn_prev.clicked.connect(self.prev_image)
        toolbar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next (D) ▶")
        self.btn_next.clicked.connect(self.next_image)
        toolbar.addWidget(self.btn_next)

        self.btn_mode_drag = QPushButton("🔲 Drag Mode (1)")
        self.btn_mode_drag.setCheckable(True)
        self.btn_mode_drag.setChecked(True)
        self.btn_mode_drag.clicked.connect(lambda: self.set_mode('drag'))
        toolbar.addWidget(self.btn_mode_drag)

        self.btn_mode_click = QPushButton("🎯 Click Mode (2)")
        self.btn_mode_click.setCheckable(True)
        self.btn_mode_click.clicked.connect(lambda: self.set_mode('click'))
        toolbar.addWidget(self.btn_mode_click)

        self.btn_undo = QPushButton("↩ Undo (Z)")
        self.btn_undo.clicked.connect(self.undo_box)
        toolbar.addWidget(self.btn_undo)

        self.btn_clear = QPushButton("Clear Boxes")
        self.btn_clear.clicked.connect(self.clear_boxes)
        toolbar.addWidget(self.btn_clear)

        self.btn_delete = QPushButton("🗑 Delete Image")
        self.btn_delete.setStyleSheet("background-color: #8b1d1d; color: #ffcccc; font-weight: bold;")
        self.btn_delete.clicked.connect(self.delete_current_image)
        toolbar.addWidget(self.btn_delete)

        self.btn_save = QPushButton("💾 Save (S)")
        self.btn_save.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_annotations)
        toolbar.addWidget(self.btn_save)

        right_layout.addLayout(toolbar)

        # Graphics Scene & View
        self.scene = QGraphicsScene()
        self.view = AnnotatorGraphicsView(self)
        self.view.setScene(self.scene)
        right_layout.addWidget(self.view)

        # Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.ball_count_label = QLabel("Balls: 0")
        self.ball_count_label.setStyleSheet("color: #00e676; font-weight: bold; font-size: 13px; padding-right: 20px;")
        self.status.addPermanentWidget(self.ball_count_label)
        self.status.showMessage("Wheel: Zoom | Right-Click Drag: Pan | Left-Click: Box Solder Ball")

        main_layout.addWidget(right_panel)

        # Keyboard shortcuts
        QShortcut(QKeySequence("A"), self, self.prev_image)
        QShortcut(QKeySequence("D"), self, self.next_image)
        QShortcut(QKeySequence("Left"), self, self.prev_image)
        QShortcut(QKeySequence("Right"), self, self.next_image)
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_box)
        QShortcut(QKeySequence("Z"), self, self.undo_box)
        QShortcut(QKeySequence("1"), self, lambda: self.set_mode('drag'))
        QShortcut(QKeySequence("2"), self, lambda: self.set_mode('click'))
        QShortcut(QKeySequence("S"), self, self.save_annotations)

    def set_mode(self, mode):
        self.mode = mode
        self.btn_mode_drag.setChecked(mode == 'drag')
        self.btn_mode_click.setChecked(mode == 'click')

    def on_image_selected(self, row):
        if row >= 0 and row != self.current_idx:
            self.save_annotations()
            self.load_image(row)

    def load_image(self, idx):
        if idx < 0 or idx >= len(self.image_files):
            return
        self.current_idx = idx
        self.list_widget.setCurrentRow(idx)

        img_path = self.image_files[idx]
        self.current_img_pixmap = QPixmap(img_path)

        self.scene.clear()
        self.box_items.clear()
        self.boxes.clear()

        # Add image to scene
        self.scene.addPixmap(self.current_img_pixmap)
        self.scene.setSceneRect(QRectF(self.current_img_pixmap.rect()))

        # Load existing labels
        base = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(self.ann_dir, f"{base}.txt")
        iw, ih = self.current_img_pixmap.width(), self.current_img_pixmap.height()

        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        x = (xc - w / 2) * iw
                        y = (yc - h / 2) * ih
                        self.add_box(QRectF(x, y, w * iw, h * ih), update_save=False)

        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.update_count_ui()

    def add_box(self, rect, update_save=True):
        self.boxes.append(rect)
        pen = QPen(QColor(0, 255, 100), 2)
        item = self.scene.addRect(rect, pen)
        self.box_items.append(item)
        self.update_count_ui()
        if update_save:
            self.save_annotations()

    def undo_box(self):
        if self.boxes:
            self.boxes.pop()
            item = self.box_items.pop()
            self.scene.removeItem(item)
            self.update_count_ui()
            self.save_annotations()

    def clear_boxes(self):
        if self.boxes:
            for item in self.box_items:
                self.scene.removeItem(item)
            self.boxes.clear()
            self.box_items.clear()
            self.update_count_ui()
            self.save_annotations()

    def update_count_ui(self):
        count = len(self.boxes)
        self.ball_count_label.setText(f"Solder Balls: {count}")

    def save_annotations(self):
        if not self.image_files or self.current_idx >= len(self.image_files):
            return
        img_path = self.image_files[self.current_idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(self.ann_dir, f"{base}.txt")

        iw, ih = self.current_img_pixmap.width(), self.current_img_pixmap.height()
        lines = []
        for rect in self.boxes:
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            xc = (x + w / 2) / iw
            yc = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        with open(ann_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        self.status.showMessage(f"Saved {len(self.boxes)} boxes to annotations/{base}.txt", 2000)

    def delete_current_image(self):
        if not self.image_files or self.current_idx >= len(self.image_files):
            return
        img_path = self.image_files[self.current_idx]
        bname = os.path.basename(img_path)
        base = os.path.splitext(bname)[0]
        ann_path = os.path.join(self.ann_dir, f"{base}.txt")

        reply = QMessageBox.question(
            self,
            "Delete Image",
            f"Are you sure you want to permanently delete:\n\n{bname}\n\nand its annotations?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                if os.path.exists(ann_path):
                    os.remove(ann_path)
                
                self.image_files.pop(self.current_idx)
                self.list_widget.takeItem(self.current_idx)

                if not self.image_files:
                    self.scene.clear()
                    self.boxes.clear()
                    self.update_count_ui()
                    self.progress_lbl.setText("0 / 0 Images")
                    self.status.showMessage("All images have been deleted.", 3000)
                    return

                if self.current_idx >= len(self.image_files):
                    self.current_idx = len(self.image_files) - 1

                self.progress_lbl.setText(f"{self.current_idx + 1} / {len(self.image_files)} Images")
                self.load_image(self.current_idx)
                self.status.showMessage(f"Deleted {bname}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete image: {e}")

    def prev_image(self):
        if self.current_idx > 0:
            self.save_annotations()
            self.load_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.save_annotations()
            self.load_image(self.current_idx + 1)

def main():
    app = QApplication(sys.argv)
    window = DesktopAnnotatorWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
