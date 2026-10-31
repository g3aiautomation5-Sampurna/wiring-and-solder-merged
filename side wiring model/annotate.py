"""
Origin 3 Custom Object Detection Annotation Tool
Built with Tkinter and Pillow.
Tailored for single-object annotation (origin_3) with YOLO format export.
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Configuration
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "dataset")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
CLASS_NAME = "origin_3"
CLASS_ID = 0

os.makedirs(LABELS_DIR, exist_ok=True)


class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Origin 3 - YOLO Object Detection Annotator")
        self.root.geometry("1200x820")
        self.root.minsize(900, 650)

        # Dataset list
        self.image_paths = sorted(
            glob.glob(os.path.join(DATASET_DIR, "*.jpg")) +
            glob.glob(os.path.join(DATASET_DIR, "*.png")) +
            glob.glob(os.path.join(DATASET_DIR, "*.jpeg"))
        )
        self.current_idx = 0

        # State for current image
        self.current_img_raw = None
        self.current_photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.disp_w = 0
        self.disp_h = 0

        # Bounding box in raw image pixels: [x1, y1, x2, y2]
        self.bbox_raw = None

        # Dragging state
        self.is_drawing = False
        self.drag_start = None

        self._build_ui()
        self._bind_keys()

        if self.image_paths:
            self.load_image(0)
        else:
            messagebox.showwarning("No Images", f"No images found in {DATASET_DIR}!")

    def _build_ui(self):
        # Top toolbar
        top_frame = ttk.Frame(self.root, padding=6)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_progress = ttk.Label(
            top_frame, text="Loading...", font=("Segoe UI", 11, "bold")
        )
        self.lbl_progress.pack(side=tk.LEFT, padx=10)

        self.lbl_status = ttk.Label(
            top_frame, text="", foreground="#007acc", font=("Segoe UI", 10)
        )
        self.lbl_status.pack(side=tk.LEFT, padx=15)

        # Quick Jump Combobox
        jump_frame = ttk.Frame(top_frame)
        jump_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(jump_frame, text="Jump to:").pack(side=tk.LEFT, padx=3)
        self.combo_images = ttk.Combobox(
            jump_frame,
            values=[os.path.basename(p) for p in self.image_paths],
            state="readonly",
            width=35,
        )
        self.combo_images.pack(side=tk.LEFT, padx=3)
        self.combo_images.bind("<<ComboboxSelected>>", self._on_combo_select)

        # Main canvas area
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#1e1e1e", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", lambda e: self.clear_box())  # Right click to clear

        # Bottom control bar
        bottom_frame = ttk.Frame(self.root, padding=8)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        btn_prev = ttk.Button(bottom_frame, text="◀ Previous (A)", command=self.prev_image)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = ttk.Button(bottom_frame, text="Next ▶ (D)", command=self.next_image)
        btn_next.pack(side=tk.LEFT, padx=5)

        btn_next_unlabeled = ttk.Button(
            bottom_frame, text="⏩ Next Unlabeled (W)", command=self.next_unlabeled_image
        )
        btn_next_unlabeled.pack(side=tk.LEFT, padx=10)

        btn_clear = ttk.Button(
            bottom_frame, text="🗑 Clear Box (Delete)", command=self.clear_box
        )
        btn_clear.pack(side=tk.LEFT, padx=10)

        btn_save = ttk.Button(
            bottom_frame, text="💾 Save Label (S)", command=self.save_current_label
        )
        btn_save.pack(side=tk.LEFT, padx=5)

        # Instructions label
        lbl_help = ttk.Label(
            bottom_frame,
            text="Controls: Drag to draw box around Origin 3 | A: Prev | D: Next | Del/Right-Click: Clear | S: Save",
            font=("Segoe UI", 9, "italic"),
            foreground="#666666",
        )
        lbl_help.pack(side=tk.RIGHT, padx=10)

    def _bind_keys(self):
        self.root.bind("<a>", lambda e: self.prev_image())
        self.root.bind("<A>", lambda e: self.prev_image())
        self.root.bind("<Left>", lambda e: self.prev_image())

        self.root.bind("<d>", lambda e: self.next_image())
        self.root.bind("<D>", lambda e: self.next_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        self.root.bind("<w>", lambda e: self.next_unlabeled_image())
        self.root.bind("<W>", lambda e: self.next_unlabeled_image())

        self.root.bind("<s>", lambda e: self.save_current_label())
        self.root.bind("<S>", lambda e: self.save_current_label())

        self.root.bind("<Delete>", lambda e: self.clear_box())
        self.root.bind("<BackSpace>", lambda e: self.clear_box())
        self.root.bind("<c>", lambda e: self.clear_box())
        self.root.bind("<C>", lambda e: self.clear_box())

    def _get_label_path(self, img_path):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(LABELS_DIR, f"{base_name}.txt")

    def count_labeled_images(self):
        count = 0
        for p in self.image_paths:
            lbl_p = self._get_label_path(p)
            if os.path.exists(lbl_p) and os.path.getsize(lbl_p) > 0:
                count += 1
        return count

    def load_image(self, idx):
        if not (0 <= idx < len(self.image_paths)):
            return

        # Save previous before switching
        if self.current_img_raw is not None:
            self.save_current_label(silent=True)

        self.current_idx = idx
        img_path = self.image_paths[idx]
        self.current_img_raw = Image.open(img_path)

        # Load label if exists
        lbl_path = self._get_label_path(img_path)
        self.bbox_raw = None
        if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
            with open(lbl_path, "r") as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id, xc, yc, w, h = map(float, parts[:5])
                        img_w, img_h = self.current_img_raw.size
                        # Convert YOLO normalized coords to pixel [x1, y1, x2, y2]
                        box_w = w * img_w
                        box_h = h * img_h
                        x1 = (xc * img_w) - (box_w / 2.0)
                        y1 = (yc * img_h) - (box_h / 2.0)
                        x2 = x1 + box_w
                        y2 = y1 + box_h
                        self.bbox_raw = [x1, y1, x2, y2]

        # Update combo and labels
        self.combo_images.current(idx)
        labeled_count = self.count_labeled_images()
        total_count = len(self.image_paths)
        status_text = "✅ Labeled" if self.bbox_raw is not None else "⏳ Not labeled yet"
        status_color = "#28a745" if self.bbox_raw is not None else "#d9534f"

        self.lbl_progress.config(
            text=f"Image {idx + 1} of {total_count} ({labeled_count}/{total_count} labeled - {int(labeled_count/total_count*100)}%)"
        )
        self.lbl_status.config(text=f"Status: {status_text}", foreground=status_color)

        self.render_canvas()

    def _on_canvas_resize(self, event):
        self.render_canvas()

    def render_canvas(self):
        if self.current_img_raw is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            return

        raw_w, raw_h = self.current_img_raw.size

        # Fit image to canvas maintaining aspect ratio
        self.scale = min(cw / raw_w, ch / raw_h)
        self.disp_w = int(raw_w * self.scale)
        self.disp_h = int(raw_h * self.scale)

        self.offset_x = (cw - self.disp_w) // 2
        self.offset_y = (ch - self.disp_h) // 2

        # Resize image for display
        resized = self.current_img_raw.resize(
            (self.disp_w, self.disp_h), Image.Resampling.BILINEAR
        )
        self.current_photo = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=self.current_photo
        )

        # Draw bbox if present
        if self.bbox_raw:
            self._draw_bbox_on_canvas(self.bbox_raw)

    def _raw_to_canvas(self, raw_x, raw_y):
        cx = self.offset_x + (raw_x * self.scale)
        cy = self.offset_y + (raw_y * self.scale)
        return cx, cy

    def _canvas_to_raw(self, cx, cy):
        raw_x = (cx - self.offset_x) / self.scale
        raw_y = (cy - self.offset_y) / self.scale
        # Clamp to image bounds
        raw_w, raw_h = self.current_img_raw.size
        raw_x = max(0, min(raw_x, raw_w))
        raw_y = max(0, min(raw_y, raw_h))
        return raw_x, raw_y

    def _draw_bbox_on_canvas(self, bbox_raw, is_active=False):
        x1, y1, x2, y2 = bbox_raw
        cx1, cy1 = self._raw_to_canvas(x1, y1)
        cx2, cy2 = self._raw_to_canvas(x2, y2)

        color = "#00ff66" if not is_active else "#ffff00"
        fill_color = ""

        # Outer box
        self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2, outline=color, width=3, tags="bbox"
        )
        # Label banner
        banner_y1 = max(cy1 - 22, self.offset_y)
        banner_y2 = banner_y1 + 20
        banner_x2 = min(cx1 + 80, cx2)
        self.canvas.create_rectangle(
            cx1, banner_y1, banner_x2, banner_y2, fill=color, outline=color, tags="bbox"
        )
        self.canvas.create_text(
            cx1 + 4,
            banner_y1 + 10,
            anchor=tk.W,
            text=CLASS_NAME,
            fill="#000000",
            font=("Segoe UI", 9, "bold"),
            tags="bbox",
        )

    def _on_mouse_down(self, event):
        if self.current_img_raw is None:
            return
        rx, ry = self._canvas_to_raw(event.x, event.y)
        self.is_drawing = True
        self.drag_start = (rx, ry)
        self.bbox_raw = None
        self.canvas.delete("bbox")

    def _on_mouse_drag(self, event):
        if not self.is_drawing or self.current_img_raw is None:
            return
        rx, ry = self._canvas_to_raw(event.x, event.y)
        x1 = min(self.drag_start[0], rx)
        y1 = min(self.drag_start[1], ry)
        x2 = max(self.drag_start[0], rx)
        y2 = max(self.drag_start[1], ry)

        self.canvas.delete("bbox")
        self._draw_bbox_on_canvas([x1, y1, x2, y2], is_active=True)

    def _on_mouse_up(self, event):
        if not self.is_drawing or self.current_img_raw is None:
            return
        self.is_drawing = False
        rx, ry = self._canvas_to_raw(event.x, event.y)

        x1 = min(self.drag_start[0], rx)
        y1 = min(self.drag_start[1], ry)
        x2 = max(self.drag_start[0], rx)
        y2 = max(self.drag_start[1], ry)

        # Minimum size check (at least 6x6 pixels)
        if (x2 - x1) >= 6 and (y2 - y1) >= 6:
            self.bbox_raw = [x1, y1, x2, y2]
            self.lbl_status.config(text="✅ Box drawn (unsaved)", foreground="#ff9900")
            self.save_current_label(silent=True)
        else:
            self.bbox_raw = None
            self.canvas.delete("bbox")
        self.render_canvas()

    def clear_box(self):
        self.bbox_raw = None
        self.canvas.delete("bbox")
        img_path = self.image_paths[self.current_idx]
        lbl_path = self._get_label_path(img_path)
        if os.path.exists(lbl_path):
            os.remove(lbl_path)

        labeled_count = self.count_labeled_images()
        total_count = len(self.image_paths)
        self.lbl_progress.config(
            text=f"Image {self.current_idx + 1} of {total_count} ({labeled_count}/{total_count} labeled - {int(labeled_count/total_count*100)}%)"
        )
        self.lbl_status.config(text="🗑 Box cleared", foreground="#d9534f")

    def save_current_label(self, silent=False):
        if self.current_img_raw is None:
            return
        img_path = self.image_paths[self.current_idx]
        lbl_path = self._get_label_path(img_path)

        if self.bbox_raw is not None:
            x1, y1, x2, y2 = self.bbox_raw
            raw_w, raw_h = self.current_img_raw.size

            bw = (x2 - x1) / raw_w
            bh = (y2 - y1) / raw_h
            xc = ((x1 + x2) / 2.0) / raw_w
            yc = ((y1 + y2) / 2.0) / raw_h

            # Format: class xc yc w h
            line = f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
            with open(lbl_path, "w") as f:
                f.write(line)

            if not silent:
                self.lbl_status.config(text="💾 Label Saved!", foreground="#28a745")
        else:
            if os.path.exists(lbl_path):
                os.remove(lbl_path)

        labeled_count = self.count_labeled_images()
        total_count = len(self.image_paths)
        status_text = "✅ Labeled" if self.bbox_raw is not None else "⏳ Not labeled yet"
        status_color = "#28a745" if self.bbox_raw is not None else "#d9534f"
        self.lbl_progress.config(
            text=f"Image {self.current_idx + 1} of {total_count} ({labeled_count}/{total_count} labeled - {int(labeled_count/total_count*100)}%)"
        )
        if not silent:
            self.lbl_status.config(text=f"Status: {status_text} (Saved)", foreground=status_color)

    def next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self.load_image(self.current_idx + 1)

    def prev_image(self):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)

    def next_unlabeled_image(self):
        # Scan forward from current_idx + 1
        for i in range(self.current_idx + 1, len(self.image_paths)):
            lbl_p = self._get_label_path(self.image_paths[i])
            if not os.path.exists(lbl_p) or os.path.getsize(lbl_p) == 0:
                self.load_image(i)
                return
        # If not found forward, scan from start
        for i in range(0, self.current_idx):
            lbl_p = self._get_label_path(self.image_paths[i])
            if not os.path.exists(lbl_p) or os.path.getsize(lbl_p) == 0:
                self.load_image(i)
                return
        messagebox.showinfo("Completed!", "🎉 All images in the dataset have been annotated!")

    def _on_combo_select(self, event):
        idx = self.combo_images.current()
        if idx != -1:
            self.load_image(idx)


def main():
    root = tk.Tk()
    app = AnnotatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
