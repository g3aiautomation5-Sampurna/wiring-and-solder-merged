"""
Unified Vision Quality Inspection Platform
Merges Wire Harness Pose Inspection & Solder Ball Detection Platform into a single unified tabbed application.
"""

import os
import sys
import time
import argparse
import webbrowser
import threading
from werkzeug.wrappers import Response
from flask import Flask, render_template_string

# Base directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_DIR = os.path.join(ROOT_DIR, "pose_estimation_poc")
SOLDER_DIR = os.path.join(ROOT_DIR, "solderball detection platform")

# Ensure both sub-projects are on sys.path
if POSE_DIR not in sys.path:
    sys.path.insert(0, POSE_DIR)
if SOLDER_DIR not in sys.path:
    sys.path.insert(0, SOLDER_DIR)

# Auto-heal data.yaml paths in both projects if folder was moved
def auto_heal_yaml_paths():
    # 1. Pose data.yaml
    pose_yaml = os.path.join(POSE_DIR, "data.yaml")
    pose_dataset = os.path.join(POSE_DIR, "dataset").replace("\\", "/")
    if os.path.exists(pose_yaml):
        try:
            with open(pose_yaml, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            for l in lines:
                if l.strip().startswith("path:"):
                    new_lines.append(f"path: {pose_dataset}\n")
                else:
                    new_lines.append(l)
            with open(pose_yaml, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"[WARN] Could not update pose data.yaml path: {e}")

    # 2. Solder data.yaml
    solder_yaml = os.path.join(SOLDER_DIR, "data.yaml")
    solder_dataset = os.path.join(SOLDER_DIR, "data").replace("\\", "/")
    if os.path.exists(solder_yaml):
        try:
            with open(solder_yaml, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            for l in lines:
                if l.strip().startswith("path:"):
                    new_lines.append(f"path: {solder_dataset}\n")
                else:
                    new_lines.append(l)
            with open(solder_yaml, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"[WARN] Could not update solder data.yaml path: {e}")

    # 3. Origin 3 Side data.yaml
    orig3_yaml = os.path.join(POSE_DIR, "origin3_dataset", "yolo_dataset", "data.yaml")
    orig3_dataset = os.path.join(POSE_DIR, "origin3_dataset", "yolo_dataset").replace("\\", "/")
    if os.path.exists(orig3_yaml):
        try:
            with open(orig3_yaml, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            for l in lines:
                if l.strip().startswith("path:"):
                    new_lines.append(f"path: {orig3_dataset}\n")
                else:
                    new_lines.append(l)
            with open(orig3_yaml, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"[WARN] Could not update Origin 3 data.yaml path: {e}")

# Packaged mode uses the writable application data directories.
if not getattr(sys, "frozen", False):
    auto_heal_yaml_paths()

# Import the sub-applications
import web_app as wiring_app
import web_platform as solder_app

# Create master management app
master_app = Flask("master_suite")

MASTER_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vision Quality Inspection Suite &mdash; Wiring & Soldering Platform</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏭</text></svg>">
    <style>
        :root {
            --bg-master: #07090e;
            --bg-header: #0f141e;
            --bg-tab-bar: #161e2e;
            --bg-tab-hover: #1f293d;
            --border: #232f45;
            --border-light: #334460;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --wire-color: #3b82f6;
            --wire-glow: rgba(59, 130, 246, 0.35);
            --solder-color: #10b981;
            --solder-glow: rgba(16, 185, 129, 0.35);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        html, body {
            width: 100%;
            height: 100%;
            overflow: hidden;
            background-color: var(--bg-master);
            color: var(--text-main);
        }

        /* Top Master Header */
        .master-header {
            height: 54px;
            background: var(--bg-header);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            user-select: none;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            position: relative;
            z-index: 100;
        }

        /* Brand Left */
        .master-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 260px;
        }

        .master-brand .brand-icon {
            font-size: 1.55rem;
            filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.5));
            animation: pulse-icon 3s ease-in-out infinite;
        }

        @keyframes pulse-icon {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.08); }
        }

        .master-brand .brand-text h1 {
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: 0.8px;
            color: #ffffff;
            text-transform: uppercase;
            line-height: 1.15;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .master-brand .brand-text span {
            font-size: 0.68rem;
            color: var(--text-muted);
            letter-spacing: 0.4px;
            font-weight: 500;
            text-transform: uppercase;
        }

        /* Master Utility Tabs (Center) */
        .master-tabs-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 1;
        }

        .master-tabs-bar {
            display: flex;
            gap: 8px;
            background: var(--bg-tab-bar);
            padding: 4px 6px;
            border-radius: 10px;
            border: 1px solid var(--border);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        .master-tab-btn {
            background: transparent;
            color: var(--text-muted);
            border: 1px solid transparent;
            padding: 6px 20px;
            border-radius: 7px;
            cursor: pointer;
            font-size: 0.86rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 9px;
            transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1);
            outline: none;
            position: relative;
        }

        .master-tab-btn:hover {
            color: #ffffff;
            background: var(--bg-tab-hover);
        }

        /* Wiring Active */
        .master-tab-btn.active.tab-wiring {
            background: #172554;
            color: #93c5fd;
            border-color: #3b82f6;
            box-shadow: 0 0 16px var(--wire-glow);
        }

        /* Soldering Active */
        .master-tab-btn.active.tab-soldering {
            background: #064e3b;
            color: #6ee7b7;
            border-color: #10b981;
            box-shadow: 0 0 16px var(--solder-glow);
        }

        .master-tab-btn .tab-icon {
            font-size: 1.1rem;
        }

        .master-tab-btn .tab-badge {
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #cbd5e1;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        .master-tab-btn.active.tab-wiring .tab-badge {
            background: rgba(59, 130, 246, 0.22);
            border-color: #3b82f6;
            color: #bfdbfe;
        }

        .master-tab-btn.active.tab-soldering .tab-badge {
            background: rgba(16, 185, 129, 0.22);
            border-color: #10b981;
            color: #a7f3d0;
        }

        /* Header Right Controls */
        .master-controls {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 260px;
            justify-content: flex-end;
        }

        .system-pill {
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
        }

        .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10b981;
            box-shadow: 0 0 8px #10b981;
            animation: pulse-dot 1.8s infinite;
        }

        @keyframes pulse-dot {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        .btn-header-action {
            background: var(--bg-tab-bar);
            border: 1px solid var(--border);
            color: var(--text-muted);
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 0.76rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s ease;
            text-decoration: none;
        }

        .btn-header-action:hover {
            color: #ffffff;
            border-color: var(--border-light);
            background: var(--bg-tab-hover);
        }

        .shortcut-hint {
            font-size: 0.68rem;
            color: #64748b;
            font-family: monospace;
            background: rgba(0, 0, 0, 0.3);
            padding: 2px 5px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Viewport Container & Iframes */
        .workspace-viewport {
            width: 100%;
            height: calc(100vh - 54px);
            position: relative;
            background: var(--bg-master);
        }

        .utility-view {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            display: none;
            border: none;
        }

        .utility-view.active {
            display: block;
        }

        iframe.utility-frame {
            width: 100%;
            height: 100%;
            border: none;
            display: block;
            background-color: var(--bg-master);
        }
    </style>
</head>
<body>
    <!-- Top Master Header -->
    <header class="master-header">
        <!-- Brand Left -->
        <div class="master-brand">
            <div class="brand-icon">🏭</div>
            <div class="brand-text">
                <h1>AOI Vision Suite</h1>
                <span>Industrial Quality Inspection</span>
            </div>
        </div>

        <!-- Master Utility Tabs (Center) -->
        <div class="master-tabs-wrap">
            <div class="master-tabs-bar" role="tablist">
                <button class="master-tab-btn active tab-wiring" id="tabBtnWiring" onclick="switchUtility('wiring')" title="Wire Harness Dual Pose & Side Inspection (Alt+1)">
                    <span class="tab-icon">🔌</span>
                    <span>Wiring Inspection</span>
                    <span class="tab-badge">Dual Pose & Side</span>
                </button>
                <button class="master-tab-btn tab-soldering" id="tabBtnSoldering" onclick="switchUtility('soldering')" title="Solder Ball Detection, Counting & Retraining (Alt+2)">
                    <span class="tab-icon">🔬</span>
                    <span>Soldering Inspection</span>
                    <span class="tab-badge">Detector Model</span>
                </button>
            </div>
        </div>

        <!-- Header Right Controls -->
        <div class="master-controls">
            <span class="shortcut-hint" title="Keyboard Shortcuts: Alt+1 for Wiring, Alt+2 for Soldering">Alt + 1 / 2</span>
            <div class="system-pill" title="Both Inspection Utilities Active and Ready">
                <div class="pulse-dot"></div>
                <span>Platform Ready</span>
            </div>
            <button class="btn-header-action" onclick="toggleFullscreen()" title="Toggle Fullscreen">
                <span id="fsIcon">⛶</span> Fullscreen
            </button>
        </div>
    </header>

    <!-- Workspace Viewport with Isolated Iframes -->
    <main class="workspace-viewport">
        <!-- Utility 1: Wiring Platform -->
        <div id="viewWiring" class="utility-view active">
            <iframe id="frameWiring" class="utility-frame" src="/wiring" allow="camera; microphone; autoplay; fullscreen"></iframe>
        </div>

        <!-- Utility 2: Soldering Platform -->
        <div id="viewSoldering" class="utility-view">
            <iframe id="frameSoldering" class="utility-frame" src="/soldering" allow="camera; microphone; autoplay; fullscreen"></iframe>
        </div>
    </main>

    <script>
        let currentUtility = 'wiring';

        function switchUtility(util) {
            if (util !== 'wiring' && util !== 'soldering') return;
            currentUtility = util;

            // Update Tab Buttons
            const btnWiring = document.getElementById('tabBtnWiring');
            const btnSoldering = document.getElementById('tabBtnSoldering');
            btnWiring.classList.toggle('active', util === 'wiring');
            btnSoldering.classList.toggle('active', util === 'soldering');

            // Update View Containers
            const viewWiring = document.getElementById('viewWiring');
            const viewSoldering = document.getElementById('viewSoldering');
            viewWiring.classList.toggle('active', util === 'wiring');
            viewSoldering.classList.toggle('active', util === 'soldering');

            // Save preference to localStorage and URL hash
            try {
                localStorage.setItem('aoi_active_utility', util);
                window.location.hash = util;
            } catch (e) {}
        }

        // Keyboard navigation: Alt+1 (Wiring), Alt+2 (Soldering)
        window.addEventListener('keydown', (e) => {
            if (e.altKey && e.key === '1') {
                e.preventDefault();
                switchUtility('wiring');
            } else if (e.altKey && e.key === '2') {
                e.preventDefault();
                switchUtility('soldering');
            }
        });

        // Restore tab on load
        window.addEventListener('DOMContentLoaded', () => {
            let initial = 'wiring';
            if (window.location.hash === '#soldering') {
                initial = 'soldering';
            } else if (window.location.hash === '#wiring') {
                initial = 'wiring';
            } else {
                try {
                    const saved = localStorage.getItem('aoi_active_utility');
                    if (saved === 'soldering' || saved === 'wiring') initial = saved;
                } catch (e) {}
            }
            switchUtility(initial);
        });

        function toggleFullscreen() {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
                document.getElementById('fsIcon').innerText = '🗗';
            } else {
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                    document.getElementById('fsIcon').innerText = '⛶';
                }
            }
        }

        // ==========================================
        // STOP MERGED UTILITY WHEN LEAVING
        // ==========================================

        window.addEventListener('pagehide', function () {

            console.log("Leaving Vision Quality Inspection Suite.");

            navigator.sendBeacon(
                "http://127.0.0.1:8000/api/pcb/shutdown"
            );

        });
    </script>
</body>
</html>
"""

@master_app.route("/")
def index():
    return render_template_string(MASTER_HTML_TEMPLATE)

# Master WSGI Dispatcher
def application(environ, start_response):
    path = environ.get("PATH_INFO", "")

    # Direct page rendering for Wiring & Soldering
    if path == "/wiring":
        env = dict(environ)
        env["PATH_INFO"] = "/"
        return wiring_app.app(env, start_response)

    elif path == "/soldering":
        env = dict(environ)
        env["PATH_INFO"] = "/"
        return solder_app.app(env, start_response)

    # Route Solderball API requests to solder_app
    elif path.startswith((
        "/api/detect",
        "/api/sample_images",
        "/api/sample_image",
        "/api/retrain",
        "/api/correction"
    )):
        return solder_app.app(environ, start_response)

    # Route Wire Harness Pose & Origin 3 API requests to wiring_app
    elif path.startswith((
        "/api/inspect",
        "/api/classes",
        "/api/annotation",
        "/api/train",
        "/api/origin3"
    )):
        return wiring_app.app(environ, start_response)

    # Master UI and all other requests
    else:
        return master_app(environ, start_response)

def open_browser(port):
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{port}")

def main():
    parser = argparse.ArgumentParser(description="Unified Vision Quality Inspection Platform")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web server on (default: 5000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    # Preload models
    print("\n" + "=" * 70)
    print("  INITIALIZING UNIFIED VISION QUALITY INSPECTION PLATFORM")
    print("=" * 70)
    print("  [1/3] Loading Wire Harness Pose Estimation Model...")
    wiring_app.get_model()
    print("  [2/3] Loading Side View (Origin 3) Detection Model...")
    wiring_app.get_origin3_model()
    print("  [3/3] Loading Solder Ball Detection Model...")
    solder_app.get_or_load_model()
    print("  All neural models initialized successfully!")
    print("=" * 70)
    print(f"  Master Dashboard URL: http://localhost:{args.port}")
    print(f"  Direct Wiring URL:    http://localhost:{args.port}/wiring")
    print(f"  Direct Soldering URL: http://localhost:{args.port}/soldering")
    print("=" * 70 + "\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    from werkzeug.serving import run_simple
    run_simple(args.host, args.port, application, use_reloader=False, threaded=True)

if __name__ == "__main__":
    main()
