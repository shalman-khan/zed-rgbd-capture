#!/usr/bin/env python3
"""
ZED RGB-D Capture Interface — ROS2 Humble
==========================================
Topics subscribed:
  /zed/zed_node/rgb/color/rect/image        (sensor_msgs/Image)
  /zed/zed_node/depth/depth_registered      (sensor_msgs/Image, 32FC1)
  /zed/zed_node/rgb/color/rect/camera_info  (sensor_msgs/CameraInfo)
  /zed/zed_node/depth/camera_info           (sensor_msgs/CameraInfo)

Outputs on Save:
  rgbd_<timestamp>_<16|32>bit.tiff          4-channel TIFF  (R, G, B, D)
  rgbd_<timestamp>_<16|32>bit.npz           NumPy archive   (rgb + depth arrays)
  rgb_camera_info_<timestamp>.txt           RGB intrinsics
  depth_camera_info_<timestamp>.txt         Depth intrinsics

16-bit mode : uint16 array — RGB [0–255], depth in millimetres
32-bit mode : float32 array — RGB [0.0–255.0], depth in metres

Keyboard shortcuts:
  Space / C  →  Capture
  S / Enter  →  Save (when a frame is captured)
  L          →  Resume live preview

Requirements:
  pip install PyQt5 tifffile opencv-python-headless numpy
"""

import sys
import time
import threading
import numpy as np
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import tifffile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel,
    QFileDialog, QGroupBox,
    QRadioButton, QMessageBox, QSpinBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# ─── Constants ────────────────────────────────────────────────────────────────

TOPIC_RGB       = "/zed/zed_node/rgb/color/rect/image"
TOPIC_DEPTH     = "/zed/zed_node/depth/depth_registered"
TOPIC_RGB_INFO  = "/zed/zed_node/rgb/color/rect/camera_info"
TOPIC_DEPTH_INFO= "/zed/zed_node/depth/camera_info"

PREVIEW_W = 640
PREVIEW_H = 480
DISPLAY_FPS = 20          # live preview update rate
DISCONNECT_TIMEOUT = 2.0  # seconds before marking topic as disconnected


# ─── ROS signals (thread-safe Qt ↔ ROS bridge) ────────────────────────────────

class RosSignals(QObject):
    rgb_frame    = pyqtSignal(object)   # np.ndarray  H×W×3 uint8
    depth_frame  = pyqtSignal(object)   # np.ndarray  H×W   float32 (metres)
    rgb_info     = pyqtSignal(object)   # sensor_msgs/CameraInfo
    depth_info   = pyqtSignal(object)   # sensor_msgs/CameraInfo


# ─── ROS2 subscriber node ─────────────────────────────────────────────────────

class ZedNode(Node):
    def __init__(self, signals: RosSignals):
        super().__init__("zed_rgbd_capture")
        self._bridge   = CvBridge()
        self._signals  = signals

        self.create_subscription(Image,      TOPIC_RGB,        self._cb_rgb,        10)
        self.create_subscription(Image,      TOPIC_DEPTH,      self._cb_depth,      10)
        self.create_subscription(CameraInfo, TOPIC_RGB_INFO,   self._cb_rgb_info,   10)
        self.create_subscription(CameraInfo, TOPIC_DEPTH_INFO, self._cb_depth_info, 10)

    def _cb_rgb(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            self._signals.rgb_frame.emit(img.copy())
        except Exception as exc:
            self.get_logger().error(f"RGB callback: {exc}")

    def _cb_depth(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self._signals.depth_frame.emit(img.copy())
        except Exception as exc:
            self.get_logger().error(f"Depth callback: {exc}")

    def _cb_rgb_info(self, msg: CameraInfo):
        self._signals.rgb_info.emit(msg)

    def _cb_depth_info(self, msg: CameraInfo):
        self._signals.depth_info.emit(msg)


# ─── Utility functions ────────────────────────────────────────────────────────

def colorize_depth(depth: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """
    Convert float32 depth (metres) to a colourised uint8 RGB image for display.
    Pixels with NaN / Inf are rendered black.
    """
    finite_mask = np.isfinite(depth)
    depth_clipped = np.where(finite_mask, np.clip(depth, min_m, max_m), 0.0)
    if max_m > min_m:
        depth_norm = ((depth_clipped - min_m) / (max_m - min_m) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_clipped, dtype=np.uint8)
    colored_bgr = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb[~finite_mask] = 0
    return colored_rgb


def ndarray_to_pixmap(img_rgb: np.ndarray,
                      max_w: int = PREVIEW_W,
                      max_h: int = PREVIEW_H) -> QPixmap:
    """Scale an H×W×3 uint8 RGB array to fit max_w×max_h and return a QPixmap."""
    h, w = img_rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = nh, nw
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def camera_info_to_text(msg: CameraInfo, label: str) -> str:
    lines = [
        f"# {label} Camera Intrinsics",
        f"# Generated: {datetime.now().isoformat()}",
        f"",
        f"image_width:  {msg.width}",
        f"image_height: {msg.height}",
        f"",
        f"# Intrinsic matrix K (row-major, 3×3)",
        f"#  [ fx   0  cx ]",
        f"#  [  0  fy  cy ]",
        f"#  [  0   0   1 ]",
        f"K: {list(msg.k)}",
        f"",
        f"fx: {msg.k[0]}",
        f"fy: {msg.k[4]}",
        f"cx: {msg.k[2]}",
        f"cy: {msg.k[5]}",
        f"",
        f"# Distortion model: {msg.distortion_model}",
        f"D: {list(msg.d)}",
        f"",
        f"# Rectification matrix R (row-major, 3×3)",
        f"R: {list(msg.r)}",
        f"",
        f"# Projection matrix P (row-major, 3×4)",
        f"P: {list(msg.p)}",
    ]
    return "\n".join(lines)


# ─── Status indicator helper ──────────────────────────────────────────────────

_CONNECTED_STYLE    = "color: #2ecc71; font-size: 11px; font-weight: bold;"
_DISCONNECTED_STYLE = "color: #e74c3c; font-size: 11px; font-weight: bold;"
_WARN_STYLE         = "color: #f39c12; font-size: 11px; font-weight: bold;"


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED RGB-D Capture  |  ROS2 Humble")
        self.setMinimumSize(1440, 780)

        # ── shared state (lock-protected) ──
        self._lock         = threading.Lock()
        self._latest_rgb   : np.ndarray | None = None
        self._latest_depth : np.ndarray | None = None
        self._last_rgb_t   : float = 0.0
        self._last_depth_t : float = 0.0

        # ── captured (frozen) frame ──
        self._cap_rgb   : np.ndarray | None = None
        self._cap_depth : np.ndarray | None = None
        self._live_mode : bool = True

        # ── camera info messages ──
        self.rgb_info_msg   = None
        self.depth_info_msg = None

        # ── save location ──
        self._save_folder = str(Path.home())

        self._build_ui()
        self._apply_dark_theme()

        # ── timers ──
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._tick_preview)
        self._preview_timer.start(1000 // DISPLAY_FPS)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._tick_status)
        self._status_timer.start(500)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # ═══ Preview row ═════════════════════════════════════════════════════
        preview_row = QHBoxLayout()
        preview_row.setSpacing(8)

        # RGB panel
        rgb_group = QGroupBox(f"RGB  ·  {TOPIC_RGB}")
        rgb_group.setFont(QFont("Monospace", 8))
        rgb_vbox = QVBoxLayout(rgb_group)
        rgb_vbox.setSpacing(4)

        self._lbl_rgb = QLabel("Waiting for RGB frames…")
        self._lbl_rgb.setAlignment(Qt.AlignCenter)
        self._lbl_rgb.setMinimumSize(PREVIEW_W, PREVIEW_H)
        self._lbl_rgb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_rgb.setStyleSheet(
            "background:#111; color:#666; border:1px solid #333; border-radius:3px;"
        )
        rgb_vbox.addWidget(self._lbl_rgb)

        self._lbl_rgb_status = QLabel("● Disconnected")
        self._lbl_rgb_status.setStyleSheet(_DISCONNECTED_STYLE)
        self._lbl_rgb_info = QLabel("")
        self._lbl_rgb_info.setStyleSheet("color:#888; font-size:10px;")
        row = QHBoxLayout()
        row.addWidget(self._lbl_rgb_status)
        row.addStretch()
        row.addWidget(self._lbl_rgb_info)
        rgb_vbox.addLayout(row)
        preview_row.addWidget(rgb_group)

        # Depth panel
        depth_group = QGroupBox(f"Depth  ·  {TOPIC_DEPTH}")
        depth_group.setFont(QFont("Monospace", 8))
        depth_vbox = QVBoxLayout(depth_group)
        depth_vbox.setSpacing(4)

        self._lbl_depth = QLabel("Waiting for Depth frames…")
        self._lbl_depth.setAlignment(Qt.AlignCenter)
        self._lbl_depth.setMinimumSize(PREVIEW_W, PREVIEW_H)
        self._lbl_depth.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_depth.setStyleSheet(
            "background:#111; color:#666; border:1px solid #333; border-radius:3px;"
        )
        depth_vbox.addWidget(self._lbl_depth)

        self._lbl_depth_status = QLabel("● Disconnected")
        self._lbl_depth_status.setStyleSheet(_DISCONNECTED_STYLE)
        self._lbl_depth_info = QLabel("")
        self._lbl_depth_info.setStyleSheet("color:#888; font-size:10px;")
        row2 = QHBoxLayout()
        row2.addWidget(self._lbl_depth_status)
        row2.addStretch()
        row2.addWidget(self._lbl_depth_info)
        depth_vbox.addLayout(row2)
        preview_row.addWidget(depth_group)

        root_layout.addLayout(preview_row, stretch=1)

        # ═══ Controls row ════════════════════════════════════════════════════
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        # ── Save folder ──
        folder_group = QGroupBox("Save Location")
        folder_layout = QHBoxLayout(folder_group)
        self._lbl_folder = QLabel(self._save_folder)
        self._lbl_folder.setStyleSheet("font-family: monospace; font-size: 10px;")
        self._lbl_folder.setWordWrap(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        folder_layout.addWidget(self._lbl_folder, stretch=1)
        folder_layout.addWidget(btn_browse)
        ctrl_row.addWidget(folder_group, stretch=3)

        # ── Bit depth ──
        bit_group = QGroupBox("Depth Storage Format")
        bit_vbox = QVBoxLayout(bit_group)
        self._radio_16 = QRadioButton("16-bit  uint16  (millimetres)")
        self._radio_32 = QRadioButton("32-bit  float32  (metres)")
        self._radio_32.setChecked(True)
        bit_vbox.addWidget(self._radio_32)
        bit_vbox.addWidget(self._radio_16)
        ctrl_row.addWidget(bit_group, stretch=2)

        # ── Depth display range ──
        range_group = QGroupBox("Depth Display Range (metres)")
        range_layout = QHBoxLayout(range_group)
        range_layout.addWidget(QLabel("Min:"))
        self._spin_min = QSpinBox()
        self._spin_min.setRange(0, 50)
        self._spin_min.setValue(0)
        self._spin_min.setSuffix(" m")
        range_layout.addWidget(self._spin_min)
        range_layout.addSpacing(10)
        range_layout.addWidget(QLabel("Max:"))
        self._spin_max = QSpinBox()
        self._spin_max.setRange(1, 100)
        self._spin_max.setValue(10)
        self._spin_max.setSuffix(" m")
        range_layout.addWidget(self._spin_max)
        ctrl_row.addWidget(range_group, stretch=2)

        # ── Action buttons ──
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout(action_group)
        action_layout.setSpacing(8)

        self._btn_capture = QPushButton("Capture  [Space]")
        self._btn_capture.setFixedHeight(52)
        self._btn_capture.setStyleSheet(_btn_style("#2980b9", "#3498db"))
        self._btn_capture.clicked.connect(self._on_capture)

        self._btn_live = QPushButton("Resume Live  [L]")
        self._btn_live.setFixedHeight(52)
        self._btn_live.setEnabled(False)
        self._btn_live.setStyleSheet(_btn_style("#7f8c8d", "#95a5a6"))
        self._btn_live.clicked.connect(self._on_resume_live)

        self._btn_save = QPushButton("Save  [S]")
        self._btn_save.setFixedHeight(52)
        self._btn_save.setEnabled(False)
        self._btn_save.setStyleSheet(_btn_style("#27ae60", "#2ecc71"))
        self._btn_save.clicked.connect(self._on_save)

        action_layout.addWidget(self._btn_capture)
        action_layout.addWidget(self._btn_live)
        action_layout.addWidget(self._btn_save)
        ctrl_row.addWidget(action_group, stretch=3)

        root_layout.addLayout(ctrl_row)

        # ── Status bar ──
        self.statusBar().showMessage("Ready  ·  Waiting for ROS2 topics…")

    # ── ROS slot handlers (called from Qt main thread via signals) ────────────

    def on_rgb_frame(self, img: np.ndarray):
        with self._lock:
            self._latest_rgb  = img
            self._last_rgb_t  = time.time()

    def on_depth_frame(self, img: np.ndarray):
        with self._lock:
            self._latest_depth = img
            self._last_depth_t = time.time()

    def on_rgb_info(self, msg):
        self.rgb_info_msg = msg

    def on_depth_info(self, msg):
        self.depth_info_msg = msg

    # ── Timer callbacks ───────────────────────────────────────────────────────

    def _tick_preview(self):
        if not self._live_mode:
            return  # frozen on captured frame

        with self._lock:
            rgb   = self._latest_rgb
            depth = self._latest_depth

        if rgb is not None:
            pix = ndarray_to_pixmap(rgb,
                                    self._lbl_rgb.width(),
                                    self._lbl_rgb.height())
            self._lbl_rgb.setPixmap(pix)
            h, w = rgb.shape[:2]
            self._lbl_rgb_info.setText(f"{w}×{h}")

        if depth is not None:
            colored = colorize_depth(depth,
                                     self._spin_min.value(),
                                     self._spin_max.value())
            pix = ndarray_to_pixmap(colored,
                                    self._lbl_depth.width(),
                                    self._lbl_depth.height())
            self._lbl_depth.setPixmap(pix)
            finite = depth[np.isfinite(depth)]
            if finite.size:
                dmin, dmax = finite.min(), finite.max()
                self._lbl_depth_info.setText(
                    f"{depth.shape[1]}×{depth.shape[0]}  "
                    f"range {dmin:.2f}–{dmax:.2f} m"
                )

    def _tick_status(self):
        now = time.time()
        rgb_ok   = (now - self._last_rgb_t)   < DISCONNECT_TIMEOUT
        depth_ok = (now - self._last_depth_t) < DISCONNECT_TIMEOUT

        self._lbl_rgb_status.setText("● Connected" if rgb_ok else "● Disconnected")
        self._lbl_rgb_status.setStyleSheet(
            _CONNECTED_STYLE if rgb_ok else _DISCONNECTED_STYLE
        )
        self._lbl_depth_status.setText("● Connected" if depth_ok else "● Disconnected")
        self._lbl_depth_status.setStyleSheet(
            _CONNECTED_STYLE if depth_ok else _DISCONNECTED_STYLE
        )

    # ── Action handlers ───────────────────────────────────────────────────────

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Save Folder", self._save_folder
        )
        if folder:
            self._save_folder = folder
            self._lbl_folder.setText(folder)

    def _on_capture(self):
        with self._lock:
            rgb   = self._latest_rgb
            depth = self._latest_depth

        if rgb is None or depth is None:
            QMessageBox.warning(
                self, "Nothing to Capture",
                "No frames received yet.\n"
                "Make sure the ZED camera is running and topics are publishing."
            )
            return

        self._cap_rgb   = rgb.copy()
        self._cap_depth = depth.copy()
        self._live_mode = False

        # Display frozen frame
        self._show_captured_frame()

        self._btn_save.setEnabled(True)
        self._btn_live.setEnabled(True)
        self._btn_capture.setText("Re-capture  [Space]")
        self.statusBar().showMessage(
            "Frame captured — press S / Save to write to disk, or L / Resume Live to go back."
        )

    def _show_captured_frame(self):
        if self._cap_rgb is not None:
            pix = ndarray_to_pixmap(self._cap_rgb,
                                    self._lbl_rgb.width(),
                                    self._lbl_rgb.height())
            self._lbl_rgb.setPixmap(pix)

        if self._cap_depth is not None:
            colored = colorize_depth(self._cap_depth,
                                     self._spin_min.value(),
                                     self._spin_max.value())
            pix = ndarray_to_pixmap(colored,
                                    self._lbl_depth.width(),
                                    self._lbl_depth.height())
            self._lbl_depth.setPixmap(pix)

    def _on_resume_live(self):
        self._live_mode = True
        self._cap_rgb   = None
        self._cap_depth = None
        self._btn_save.setEnabled(False)
        self._btn_live.setEnabled(False)
        self._btn_capture.setText("Capture  [Space]")
        self.statusBar().showMessage("Live preview resumed.")

    def _on_save(self):
        if self._cap_rgb is None or self._cap_depth is None:
            return

        use_16 = self._radio_16.isChecked()
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag    = "16bit" if use_16 else "32bit"
        outdir = Path(self._save_folder)

        try:
            rgb   = self._cap_rgb                          # H×W×3  uint8
            depth = self._cap_depth.astype(np.float32)    # H×W    float32 metres
            finite_mask = np.isfinite(depth)

            # ── build 4-channel array ──────────────────────────────────────
            if use_16:
                depth_mm  = np.where(finite_mask, depth * 1000.0, 0.0)
                depth_mm  = np.clip(depth_mm, 0, 65535).astype(np.uint16)
                rgb_u16   = rgb.astype(np.uint16)
                combined  = np.dstack([rgb_u16, depth_mm])    # H×W×4  uint16
                depth_npz = depth_mm.copy()
                depth_unit = "mm (uint16)"
            else:
                depth_f   = np.where(finite_mask, depth, 0.0).astype(np.float32)
                rgb_f     = rgb.astype(np.float32)
                combined  = np.dstack([rgb_f, depth_f])        # H×W×4  float32
                depth_npz = depth_f.copy()
                depth_unit = "m (float32)"

            # ── TIFF ──────────────────────────────────────────────────────
            tiff_path = outdir / f"rgbd_{ts}_{tag}.tiff"
            tifffile.imwrite(
                str(tiff_path),
                combined,
                metadata={"axes": "YXS",
                          "channels": ["R", "G", "B", "D"],
                          "depth_unit": depth_unit},
            )

            # ── NPZ ───────────────────────────────────────────────────────
            npz_path = outdir / f"rgbd_{ts}_{tag}.npz"
            save_dict: dict = {
                "rgb":        self._cap_rgb,     # always uint8
                "depth":      depth_npz,
                "depth_unit": np.bytes_(depth_unit),
                "timestamp":  np.bytes_(ts),
            }
            if self.rgb_info_msg is not None:
                save_dict["rgb_K"] = np.array(self.rgb_info_msg.k).reshape(3, 3)
                save_dict["rgb_D"] = np.array(self.rgb_info_msg.d)
            if self.depth_info_msg is not None:
                save_dict["depth_K"] = np.array(self.depth_info_msg.k).reshape(3, 3)
                save_dict["depth_D"] = np.array(self.depth_info_msg.d)
            np.savez_compressed(str(npz_path), **save_dict)

            # ── Intrinsics text ───────────────────────────────────────────
            saved_txt: list[str] = []

            if self.rgb_info_msg is not None:
                p = outdir / f"rgb_camera_info_{ts}.txt"
                p.write_text(camera_info_to_text(self.rgb_info_msg, "RGB"))
                saved_txt.append(p.name)

            if self.depth_info_msg is not None:
                p = outdir / f"depth_camera_info_{ts}.txt"
                p.write_text(camera_info_to_text(self.depth_info_msg, "Depth"))
                saved_txt.append(p.name)

            # ── success dialog ────────────────────────────────────────────
            file_list = (
                f"  • {tiff_path.name}\n"
                f"  • {npz_path.name}"
            )
            for t in saved_txt:
                file_list += f"\n  • {t}"
            if not saved_txt:
                file_list += "\n  (no camera_info received — intrinsics not saved)"

            QMessageBox.information(
                self, "Saved Successfully",
                f"Files written to:\n{outdir}\n\n{file_list}"
            )
            self.statusBar().showMessage(f"Saved  →  {outdir}")

            # Resume live after successful save
            self._on_resume_live()

        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save:\n\n{exc}")
            self.statusBar().showMessage(f"Save error: {exc}")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Space, Qt.Key_C):
            self._on_capture()
        elif key in (Qt.Key_S, Qt.Key_Return, Qt.Key_Enter):
            if self._btn_save.isEnabled():
                self._on_save()
        elif key == Qt.Key_L:
            if self._btn_live.isEnabled():
                self._on_resume_live()
        else:
            super().keyPressEvent(event)

    # ── Dark theme ────────────────────────────────────────────────────────────

    def _apply_dark_theme(self):
        p = QPalette()
        dark   = QColor(40, 40, 40)
        medium = QColor(55, 55, 55)
        light  = QColor(220, 220, 220)
        p.setColor(QPalette.Window,          dark)
        p.setColor(QPalette.WindowText,      light)
        p.setColor(QPalette.Base,            QColor(30, 30, 30))
        p.setColor(QPalette.AlternateBase,   medium)
        p.setColor(QPalette.ToolTipBase,     dark)
        p.setColor(QPalette.ToolTipText,     light)
        p.setColor(QPalette.Text,            light)
        p.setColor(QPalette.Button,          medium)
        p.setColor(QPalette.ButtonText,      light)
        p.setColor(QPalette.BrightText,      Qt.red)
        p.setColor(QPalette.Link,            QColor(42, 130, 218))
        p.setColor(QPalette.Highlight,       QColor(42, 130, 218))
        p.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(p)


# ─── Button style helper ──────────────────────────────────────────────────────

def _btn_style(normal: str, hover: str) -> str:
    return (
        f"QPushButton {{"
        f"  background: {normal}; color: white;"
        f"  font-size: 13px; font-weight: bold;"
        f"  border-radius: 4px; border: none;"
        f"}}"
        f"QPushButton:hover {{ background: {hover}; }}"
        f"QPushButton:disabled {{ background: #3a3a3a; color: #666; }}"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    rclpy.init(args=sys.argv)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    signals = RosSignals()
    node    = ZedNode(signals)

    window = MainWindow()
    signals.rgb_frame.connect(window.on_rgb_frame)
    signals.depth_frame.connect(window.on_depth_frame)
    signals.rgb_info.connect(window.on_rgb_info)
    signals.depth_info.connect(window.on_depth_info)

    # Spin ROS2 in a background daemon thread
    ros_thread = threading.Thread(
        target=lambda: rclpy.spin(node),
        daemon=True,
        name="ros2_spin",
    )
    ros_thread.start()

    window.show()
    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
