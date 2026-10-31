"""
Offline Solder Ball Annotation Tool
A lightweight, high-performance local web app tailored for annotating tiny solder balls on PCBs.

Features:
- 100% offline (no internet connection needed, zero external CDNs)
- Smooth zoom (up to 20x) and pan for inspecting tiny solder balls on 1920x1080 images
- Two labeling modes:
    1. Drag Rectangle Mode (drag a custom box)
    2. Quick-Click Marker Mode (single click to auto-box a ball with adjustable radius)
- Auto-saves directly to YOLO format (annotations/{image_name}.txt)
- AI / CV Assist: Suggests solder ball locations using circle detection / trained model
- Keyboard shortcuts for rapid annotation (A: Prev, D: Next, Z: Undo, Del: Delete, Space: Pan)
"""

import os
import glob
import json
import webbrowser
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template_string

app = Flask(__name__)

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset')
ANNOTATIONS_DIR = os.path.join(WORKSPACE_DIR, 'annotations')
MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Solder Ball Offline Annotator</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #1a1a1e;
            color: #e0e0e6;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* Sidebar */
        #sidebar {
            width: 330px;
            background: #222228;
            border-right: 1px solid #33333d;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            z-index: 10;
        }
        .header {
            padding: 16px 18px;
            background: #2a2a33;
            border-bottom: 1px solid #383844;
        }
        .header h1 {
            font-size: 16px;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .progress-box {
            margin-top: 10px;
            background: #1a1a1f;
            border-radius: 6px;
            padding: 8px 10px;
        }
        .progress-bar-bg {
            height: 6px;
            background: #33333e;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 6px;
        }
        .progress-bar-fill {
            height: 100%;
            background: #00d26a;
            width: 0%;
            transition: width 0.3s;
        }
        .progress-text {
            font-size: 12px;
            color: #999;
            display: flex;
            justify-content: space-between;
        }
        .filter-controls {
            padding: 10px 14px;
            background: #25252c;
            border-bottom: 1px solid #33333d;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .search-input {
            width: 100%;
            padding: 7px 10px;
            border-radius: 6px;
            background: #1a1a1e;
            border: 1px solid #3d3d4a;
            color: #fff;
            font-size: 13px;
            outline: none;
        }
        .search-input:focus { border-color: #0088ff; }
        .filter-pills {
            display: flex;
            gap: 6px;
        }
        .filter-pill {
            flex: 1;
            padding: 5px 0;
            font-size: 11px;
            text-align: center;
            background: #1c1c22;
            border: 1px solid #333340;
            border-radius: 4px;
            cursor: pointer;
            color: #aaa;
        }
        .filter-pill.active {
            background: #0066cc;
            color: #fff;
            border-color: #0088ff;
            font-weight: 600;
        }
        #image-list {
            flex: 1;
            overflow-y: auto;
            padding: 6px;
        }
        .img-item {
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            cursor: pointer;
            font-size: 13px;
            background: #25252e;
            border: 1px solid transparent;
            transition: all 0.15s;
        }
        .img-item:hover { background: #2c2c38; }
        .img-item.active {
            background: #1e3a5f;
            border-color: #0088ff;
            color: #ffffff;
            font-weight: 600;
        }
        .badge {
            font-size: 11px;
            padding: 2px 7px;
            border-radius: 12px;
            background: #3a3a46;
            color: #bbb;
        }
        .badge.annotated {
            background: #0d4a2b;
            color: #00ff88;
            font-weight: 600;
        }

        /* Main Workspace */
        #main-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        /* Top Toolbar */
        #toolbar {
            height: 52px;
            background: #24242c;
            border-bottom: 1px solid #33333d;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            gap: 12px;
            z-index: 5;
        }
        .tool-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .btn {
            background: #32323e;
            color: #e0e0e0;
            border: 1px solid #454554;
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s;
        }
        .btn:hover { background: #3d3d4d; color: #fff; }
        .btn.active {
            background: #0066cc;
            border-color: #0088ff;
            color: #fff;
        }
        .btn-green {
            background: #116839;
            border-color: #00a854;
            color: #fff;
        }
        .btn-green:hover { background: #158348; }
        .btn-assist {
            background: #4a2870;
            border-color: #8b4ed4;
            color: #fff;
        }
        .btn-assist:hover { background: #5c328c; }
        .btn-danger {
            background: #6e2020;
            border-color: #9c2828;
            color: #fff;
        }
        .btn-danger:hover { background: #872727; }
        .btn-delete {
            background: #4a1515;
            border-color: #822222;
            color: #ff9999;
        }
        .btn-delete:hover {
            background: #731e1e;
            color: #ffffff;
        }

        .slider-label {
            font-size: 11px;
            color: #aaa;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .slider-label input[type=range] {
            width: 70px;
            cursor: pointer;
        }

        /* Canvas viewport */
        #viewport {
            flex: 1;
            position: relative;
            background: #141418;
            overflow: hidden;
            cursor: crosshair;
        }
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
            transform-origin: 0 0;
        }

        /* Bottom Status bar */
        #statusbar {
            height: 32px;
            background: #1d1d23;
            border-top: 1px solid #2e2e38;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            font-size: 12px;
            color: #888;
        }
        .status-highlight {
            color: #00ff88;
            font-weight: 600;
        }

        /* Shortcut helper modal */
        .shortcut-pill {
            background: #2a2a35;
            padding: 2px 5px;
            border-radius: 3px;
            border: 1px solid #444;
            font-size: 11px;
            color: #ffcc00;
        }
    </style>
</head>
<body>

    <!-- Left Sidebar -->
    <div id="sidebar">
        <div class="header">
            <h1>🔍 Solder Ball Annotator</h1>
            <div class="progress-box">
                <div class="progress-text">
                    <span>Progress</span>
                    <span id="progress-stat">0 / 0</span>
                </div>
                <div class="progress-bar-bg">
                    <div id="progress-bar" class="progress-bar-fill"></div>
                </div>
            </div>
        </div>

        <div class="filter-controls">
            <input type="text" id="search-box" class="search-input" placeholder="Search image (e.g. 15)...">
            <div class="filter-pills">
                <div class="filter-pill active" data-filter="all" onclick="setFilter('all')">All</div>
                <div class="filter-pill" data-filter="unannotated" onclick="setFilter('unannotated')">Unannotated</div>
                <div class="filter-pill" data-filter="annotated" onclick="setFilter('annotated')">Annotated</div>
            </div>
        </div>

        <div id="image-list"></div>
    </div>

    <!-- Main Workspace -->
    <div id="main-area">
        <!-- Top Toolbar -->
        <div id="toolbar">
            <div class="tool-group">
                <button class="btn" onclick="prevImage()" title="Shortcut: A or Left Arrow">◀ Prev (A)</button>
                <button class="btn" onclick="nextImage()" title="Shortcut: D or Right Arrow">Next (D) ▶</button>
                <span id="current-filename" style="font-weight: 600; font-size: 13px; color: #fff; margin-left: 8px;"></span>
            </div>

            <div class="tool-group">
                <button id="btn-mode-drag" class="btn active" onclick="setMode('drag')" title="Click & drag rectangle">
                    🔲 Box Drag (1)
                </button>
                <button id="btn-mode-click" class="btn" onclick="setMode('click')" title="Single-click to place box">
                    🎯 Quick Click (2)
                </button>

                <div class="slider-label" id="box-size-controls" style="display: none;">
                    Size: <input type="range" id="quick-radius" min="10" max="60" value="24" oninput="updateRadiusLabel(this.value)">
                    <span id="radius-val">24px</span>
                </div>
            </div>

            <div class="tool-group">
                <button class="btn btn-assist" onclick="runAutoDetect()" title="Automatically detects solder balls using CV">
                    ✨ Auto-Detect
                </button>
                <button class="btn" onclick="undoLastBox()" title="Shortcut: Ctrl+Z">
                    ↩ Undo (Z)
                </button>
                <button class="btn btn-danger" onclick="clearCurrentBoxes()" title="Clear all bounding boxes on this image">
                    Clear Boxes
                </button>
                <button class="btn btn-delete" onclick="deleteCurrentImage()" title="Permanently delete this image and its annotations">
                    🗑️ Delete Image
                </button>
                <button class="btn btn-green" onclick="saveAnnotations(true)">
                    💾 Saved
                </button>
            </div>
        </div>

        <!-- Canvas Viewport -->
        <div id="viewport">
            <canvas id="canvas"></canvas>
        </div>

        <!-- Status Bar -->
        <div id="statusbar">
            <div>
                Zoom: <span id="zoom-text" class="status-highlight">100%</span> | 
                Solder Balls: <span id="ball-count-text" class="status-highlight">0</span> |
                Mode: <span id="mode-text" class="status-highlight">Box Drag</span>
            </div>
            <div>
                <span class="shortcut-pill">Wheel</span> Zoom &bull; 
                <span class="shortcut-pill">Right-Click Drag / Space</span> Pan &bull; 
                <span class="shortcut-pill">Left-Click</span> Draw &bull; 
                <span class="shortcut-pill">F</span> Fit Screen
            </div>
        </div>
    </div>

    <script>
        // State variables
        let images = [];
        let currentIdx = 0;
        let currentFilter = 'all';
        let currentMode = 'drag'; // 'drag' or 'click'
        let quickRadius = 24;

        let imgElement = new Image();
        let boxes = []; // Array of {x1, y1, x2, y2} in image pixel coords

        // Viewport transform
        let scale = 1.0;
        let panX = 0;
        let panY = 0;
        let isPanning = false;
        let startPanX = 0, startPanY = 0;

        // Dragging box state
        let isDrawing = false;
        let drawStartX = 0, drawStartY = 0;
        let drawCurrentX = 0, drawCurrentY = 0;

        const viewport = document.getElementById('viewport');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');

        // Fetch image list
        async function fetchImages() {
            const res = await fetch('/api/images');
            images = await res.json();
            renderImageList();
            updateProgress();
            if (images.length > 0) {
                loadImage(0);
            }
        }

        function updateProgress() {
            const annotated = images.filter(img => img.count > 0 || img.has_ann).length;
            const total = images.length;
            const pct = total > 0 ? (annotated / total) * 100 : 0;
            document.getElementById('progress-stat').innerText = `${annotated} / ${total} (${pct.toFixed(0)}%)`;
            document.getElementById('progress-bar').style.width = `${pct}%`;
        }

        function setFilter(filter) {
            currentFilter = filter;
            document.querySelectorAll('.filter-pill').forEach(p => {
                p.classList.toggle('active', p.getAttribute('data-filter') === filter);
            });
            renderImageList();
        }

        document.getElementById('search-box').addEventListener('input', () => {
            renderImageList();
        });

        function renderImageList() {
            const listDiv = document.getElementById('image-list');
            const search = document.getElementById('search-box').value.toLowerCase();
            listDiv.innerHTML = '';

            images.forEach((img, idx) => {
                const matchSearch = img.name.toLowerCase().includes(search);
                const isAnn = img.count > 0 || img.has_ann;
                const matchFilter = currentFilter === 'all' || 
                                    (currentFilter === 'annotated' && isAnn) ||
                                    (currentFilter === 'unannotated' && !isAnn);

                if (matchSearch && matchFilter) {
                    const item = document.createElement('div');
                    item.className = 'img-item' + (idx === currentIdx ? ' active' : '');
                    item.onclick = () => {
                        saveAnnotations(false);
                        loadImage(idx);
                    };

                    const labelSpan = document.createElement('span');
                    labelSpan.innerText = img.name;

                    const badge = document.createElement('span');
                    badge.className = 'badge' + (isAnn ? ' annotated' : '');
                    badge.innerText = isAnn ? `${img.count} balls` : '0';

                    item.appendChild(labelSpan);
                    item.appendChild(badge);
                    listDiv.appendChild(item);
                }
            });
        }

        async function loadImage(idx) {
            if (idx < 0 || idx >= images.length) return;
            currentIdx = idx;
            const imgData = images[currentIdx];
            document.getElementById('current-filename').innerText = imgData.name;

            // Highlight in list
            document.querySelectorAll('.img-item').forEach((el, i) => {
                el.classList.toggle('active', images[i] && images[i].name === imgData.name);
            });

            // Load annotations
            const annRes = await fetch(`/api/annotation/${encodeURIComponent(imgData.name)}`);
            boxes = await annRes.json();
            updateBallCount();

            // Load image
            imgElement.onload = () => {
                canvas.width = imgElement.width;
                canvas.height = imgElement.height;
                fitToScreen();
                draw();
            };
            imgElement.src = `/api/image/${encodeURIComponent(imgData.name)}`;
        }

        function fitToScreen() {
            const vw = viewport.clientWidth;
            const vh = viewport.clientHeight;
            const iw = imgElement.width;
            const ih = imgElement.height;
            if (iw === 0 || ih === 0) return;

            scale = Math.min((vw - 40) / iw, (vh - 40) / ih, 1.0);
            panX = (vw - iw * scale) / 2;
            panY = (vh - ih * scale) / 2;
            updateTransform();
        }

        function updateTransform() {
            canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
            document.getElementById('zoom-text').innerText = `${Math.round(scale * 100)}%`;
        }

        function updateBallCount() {
            document.getElementById('ball-count-text').innerText = boxes.length;
            if (images[currentIdx]) {
                images[currentIdx].count = boxes.length;
                images[currentIdx].has_ann = boxes.length > 0;
            }
            updateProgress();
        }

        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(imgElement, 0, 0);

            // Draw confirmed boxes
            boxes.forEach((b, i) => {
                const w = b.x2 - b.x1;
                const h = b.y2 - b.y1;

                // Box
                ctx.strokeStyle = '#00ff66';
                ctx.lineWidth = Math.max(1, 2 / scale);
                ctx.strokeRect(b.x1, b.y1, w, h);

                // Center cross
                const cx = (b.x1 + b.x2) / 2;
                const cy = (b.y1 + b.y2) / 2;
                ctx.fillStyle = '#ff2255';
                ctx.beginPath();
                ctx.arc(cx, cy, Math.max(2, 3 / scale), 0, Math.PI * 2);
                ctx.fill();

                // Number badge
                const fontSize = Math.max(11, 14 / scale);
                ctx.font = `bold ${fontSize}px sans-serif`;
                const text = `#${i + 1}`;
                const textWidth = ctx.measureText(text).width;

                ctx.fillStyle = '#00ff66';
                ctx.fillRect(b.x1, Math.max(0, b.y1 - fontSize - 4), textWidth + 6, fontSize + 4);

                ctx.fillStyle = '#000000';
                ctx.fillText(text, b.x1 + 3, Math.max(fontSize, b.y1 - 3));
            });

            // Draw current drawing box
            if (isDrawing && currentMode === 'drag') {
                const x1 = Math.min(drawStartX, drawCurrentX);
                const y1 = Math.min(drawStartY, drawCurrentY);
                const x2 = Math.max(drawStartX, drawCurrentX);
                const y2 = Math.max(drawStartY, drawCurrentY);

                ctx.strokeStyle = '#00ccff';
                ctx.lineWidth = Math.max(1.5, 2.5 / scale);
                ctx.setLineDash([4 / scale, 4 / scale]);
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.setLineDash([]);
            }
        }

        // Coordinate converter
        function getCanvasCoords(e) {
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / scale;
            const y = (e.clientY - rect.top) / scale;
            return {
                x: Math.max(0, Math.min(canvas.width, Math.round(x))),
                y: Math.max(0, Math.min(canvas.height, Math.round(y)))
            };
        }

        // Viewport Event Listeners
        viewport.addEventListener('wheel', (e) => {
            e.preventDefault();
            const zoomFactor = e.deltaY < 0 ? 1.25 : 0.8;
            const newScale = Math.min(Math.max(0.1, scale * zoomFactor), 25.0);

            // Zoom towards mouse pointer
            const mouseX = e.clientX;
            const mouseY = e.clientY;
            panX = mouseX - (mouseX - panX) * (newScale / scale);
            panY = mouseY - (mouseY - panY) * (newScale / scale);
            scale = newScale;

            updateTransform();
            draw();
        }, { passive: false });

        viewport.addEventListener('mousedown', (e) => {
            // Right-click or Middle-click or Space held -> Pan
            if (e.button === 2 || e.button === 1 || e.shiftKey || isSpacePressed) {
                isPanning = true;
                startPanX = e.clientX - panX;
                startPanY = e.clientY - panY;
                viewport.style.cursor = 'grab';
                return;
            }

            // Left-click -> Annotate
            if (e.button === 0) {
                const coords = getCanvasCoords(e);
                if (currentMode === 'click') {
                    // Quick-click mode: create square box centered at click
                    const r = quickRadius;
                    const x1 = Math.max(0, coords.x - r);
                    const y1 = Math.max(0, coords.y - r);
                    const x2 = Math.min(canvas.width, coords.x + r);
                    const y2 = Math.min(canvas.height, coords.y + r);

                    boxes.push({ x1, y1, x2, y2 });
                    updateBallCount();
                    draw();
                    saveAnnotations(false);
                } else {
                    // Drag mode
                    isDrawing = true;
                    drawStartX = coords.x;
                    drawStartY = coords.y;
                    drawCurrentX = coords.x;
                    drawCurrentY = coords.y;
                }
            }
        });

        window.addEventListener('mousemove', (e) => {
            if (isPanning) {
                panX = e.clientX - startPanX;
                panY = e.clientY - startPanY;
                updateTransform();
                return;
            }

            if (isDrawing) {
                const coords = getCanvasCoords(e);
                drawCurrentX = coords.x;
                drawCurrentY = coords.y;
                draw();
            }
        });

        window.addEventListener('mouseup', (e) => {
            if (isPanning) {
                isPanning = false;
                viewport.style.cursor = 'crosshair';
            }

            if (isDrawing) {
                isDrawing = false;
                const x1 = Math.min(drawStartX, drawCurrentX);
                const y1 = Math.min(drawStartY, drawCurrentY);
                const x2 = Math.max(drawStartX, drawCurrentX);
                const y2 = Math.max(drawStartY, drawCurrentY);

                // Only add box if size > 4 pixels
                if (x2 - x1 >= 4 && y2 - y1 >= 4) {
                    boxes.push({ x1, y1, x2, y2 });
                    updateBallCount();
                    saveAnnotations(false);
                }
                draw();
            }
        });

        // Context menu disable on canvas for right-click panning
        viewport.addEventListener('contextmenu', e => e.preventDefault());

        // Mode toggling
        function setMode(mode) {
            currentMode = mode;
            document.getElementById('btn-mode-drag').classList.toggle('active', mode === 'drag');
            document.getElementById('btn-mode-click').classList.toggle('active', mode === 'click');
            document.getElementById('box-size-controls').style.display = mode === 'click' ? 'flex' : 'none';
            document.getElementById('mode-text').innerText = mode === 'click' ? 'Quick Click' : 'Box Drag';
        }

        function updateRadiusLabel(val) {
            quickRadius = parseInt(val);
            document.getElementById('radius-val').innerText = `${quickRadius}px`;
        }

        // Actions
        function undoLastBox() {
            if (boxes.length > 0) {
                boxes.pop();
                updateBallCount();
                draw();
                saveAnnotations(false);
            }
        }

        function clearCurrentBoxes() {
            if (boxes.length > 0 && confirm('Clear all boxes for this image?')) {
                boxes = [];
                updateBallCount();
                draw();
                saveAnnotations(false);
            }
        }

        async function deleteCurrentImage() {
            if (!images[currentIdx]) return;
            const imgName = images[currentIdx].name;
            const confirmed = confirm(`Are you sure you want to permanently delete this image: \n\n"${imgName}"\n\nand its annotations? This cannot be undone.`);
            if (!confirmed) return;

            try {
                const res = await fetch(`/api/image/${encodeURIComponent(imgName)}`, {
                    method: 'DELETE'
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    // Remove from list
                    images.splice(currentIdx, 1);
                    if (images.length === 0) {
                        alert('All images have been deleted.');
                        boxes = [];
                        updateBallCount();
                        renderImageList();
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        document.getElementById('current-filename').innerText = '(No images left)';
                        return;
                    }
                    if (currentIdx >= images.length) {
                        currentIdx = images.length - 1;
                    }
                    renderImageList();
                    updateProgress();
                    loadImage(currentIdx);
                } else {
                    alert('Error: ' + (data.message || 'Could not delete image'));
                }
            } catch (err) {
                alert('Request failed: ' + err);
            }
        }

        async function saveAnnotations(showMessage = true) {
            if (!images[currentIdx]) return;
            const imgName = images[currentIdx].name;
            const res = await fetch(`/api/annotation/${encodeURIComponent(imgName)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ boxes: boxes })
            });
            if (showMessage) {
                const btn = document.querySelector('.btn-green');
                const orig = btn.innerText;
                btn.innerText = '✓ Saved!';
                setTimeout(() => btn.innerText = orig, 1200);
            }
            renderImageList();
        }

        async function runAutoDetect() {
            if (!images[currentIdx]) return;
            const imgName = images[currentIdx].name;
            const res = await fetch(`/api/auto_detect/${encodeURIComponent(imgName)}`);
            const suggested = await res.json();
            if (suggested && suggested.length > 0) {
                boxes = [...boxes, ...suggested];
                updateBallCount();
                draw();
                saveAnnotations(false);
            } else {
                alert('No additional solder balls automatically detected on this image.');
            }
        }

        function prevImage() {
            if (currentIdx > 0) {
                saveAnnotations(false);
                loadImage(currentIdx - 1);
            }
        }

        function nextImage() {
            if (currentIdx < images.length - 1) {
                saveAnnotations(false);
                loadImage(currentIdx + 1);
            }
        }

        // Keyboard Shortcuts
        let isSpacePressed = false;
        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            if (e.key === ' ' || e.code === 'Space') {
                isSpacePressed = true;
                viewport.style.cursor = 'grab';
            } else if (e.key === 'a' || e.key === 'A' || e.key === 'ArrowLeft') {
                prevImage();
            } else if (e.key === 'd' || e.key === 'D' || e.key === 'ArrowRight') {
                nextImage();
            } else if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
                undoLastBox();
            } else if (e.key === '1') {
                setMode('drag');
            } else if (e.key === '2') {
                setMode('click');
            } else if (e.key === 'f' || e.key === 'F') {
                fitToScreen();
            }
        });

        window.addEventListener('keyup', (e) => {
            if (e.key === ' ' || e.code === 'Space') {
                isSpacePressed = false;
                viewport.style.cursor = 'crosshair';
            }
        });

        // Initialize
        fetchImages();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/images')
def list_images():
    files = sorted(glob.glob(os.path.join(DATASET_DIR, '*.jpg')) +
                   glob.glob(os.path.join(DATASET_DIR, '*.png')))
    
    result = []
    for f in files:
        bname = os.path.basename(f)
        bnoext = os.path.splitext(bname)[0]
        ann_path = os.path.join(ANNOTATIONS_DIR, f"{bnoext}.txt")
        count = 0
        has_ann = False
        if os.path.exists(ann_path):
            has_ann = True
            with open(ann_path, 'r') as af:
                lines = [l.strip() for l in af if l.strip()]
                count = len(lines)
        result.append({
            'name': bname,
            'has_ann': has_ann,
            'count': count
        })
    return jsonify(result)

@app.route('/api/image/<path:filename>', methods=['GET'])
def get_image(filename):
    fpath = os.path.join(DATASET_DIR, filename)
    if os.path.exists(fpath):
        return send_file(fpath)
    return "Not found", 404

@app.route('/api/image/<path:filename>', methods=['DELETE'])
def delete_image(filename):
    img_path = os.path.join(DATASET_DIR, filename)
    bnoext = os.path.splitext(filename)[0]
    ann_path = os.path.join(ANNOTATIONS_DIR, f"{bnoext}.txt")
    dataset_ann_path = os.path.join(DATASET_DIR, f"{bnoext}.txt")

    deleted_files = []
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted_files.append(img_path)
        if os.path.exists(ann_path):
            os.remove(ann_path)
            deleted_files.append(ann_path)
        if os.path.exists(dataset_ann_path):
            os.remove(dataset_ann_path)
            deleted_files.append(dataset_ann_path)
        return jsonify({'status': 'ok', 'deleted': deleted_files})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/annotation/<path:filename>', methods=['GET'])
def get_annotation(filename):
    bnoext = os.path.splitext(filename)[0]
    ann_path = os.path.join(ANNOTATIONS_DIR, f"{bnoext}.txt")
    img_path = os.path.join(DATASET_DIR, filename)

    boxes = []
    if os.path.exists(ann_path) and os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            ih, iw = img.shape[:2]
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = int(round((xc - w / 2.0) * iw))
                        y1 = int(round((yc - h / 2.0) * ih))
                        x2 = int(round((xc + w / 2.0) * iw))
                        y2 = int(round((yc + h / 2.0) * ih))
                        boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
    return jsonify(boxes)

@app.route('/api/annotation/<path:filename>', methods=['POST'])
def save_annotation(filename):
    bnoext = os.path.splitext(filename)[0]
    ann_path = os.path.join(ANNOTATIONS_DIR, f"{bnoext}.txt")
    img_path = os.path.join(DATASET_DIR, filename)

    data = request.json or {}
    boxes = data.get('boxes', [])

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            ih, iw = img.shape[:2]
            lines = []
            for b in boxes:
                x1, y1 = b['x1'], b['y1']
                x2, y2 = b['x2'], b['y2']
                # Clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(ih, y2)
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    xc = (x1 + x2) / (2.0 * iw)
                    yc = (y1 + y2) / (2.0 * ih)
                    norm_w = w / float(iw)
                    norm_h = h / float(ih)
                    lines.append(f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

            with open(ann_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

    return jsonify({'status': 'ok', 'count': len(boxes)})

@app.route('/api/auto_detect/<path:filename>')
def auto_detect(filename):
    img_path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(img_path):
        return jsonify([])

    img = cv2.imread(img_path)
    if img is None:
        return jsonify([])

    ih, iw = img.shape[:2]
    model_weights = os.path.join(MODELS_DIR, 'best_solderball_detector.pt')

    boxes = []
    # If trained model exists, use it
    if os.path.exists(model_weights):
        try:
            from ultralytics import YOLO
            yolo = YOLO(model_weights)
            results = yolo.predict(img, conf=0.25, verbose=False)
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                    boxes.append({
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2)
                    })
            return jsonify(boxes)
        except Exception:
            pass

    # Fallback to OpenCV circular blob detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=15, 
        param1=50, 
        param2=28, 
        minRadius=3, 
        maxRadius=25
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            pad = int(r * 1.3)
            boxes.append({
                'x1': max(0, x - pad),
                'y1': max(0, y - pad),
                'x2': min(iw, x + pad),
                'y2': min(ih, y + pad)
            })

    return jsonify(boxes)

if __name__ == '__main__':
    print("=" * 65)
    print("  SOLDER BALL OFFLINE ANNOTATION TOOL")
    print("=" * 65)
    print("  Server starting locally at: http://127.0.0.1:5000")
    print("  100% offline - zero internet connection needed.")
    print("  Annotations automatically save to: annotations/")
    print("=" * 65)
    
    # Open default browser
    try:
        webbrowser.open('http://127.0.0.1:5000')
    except Exception:
        pass

    app.run(host='127.0.0.1', port=5000, debug=False)
