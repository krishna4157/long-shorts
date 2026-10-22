import os
import sys
import urllib.request
import traceback
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)


def ensure_yunet_model():
    """
    Ensures the 227KB YuNet ONNX deep-learning face model is available locally.
    Stores it in %LOCALAPPDATA%/VerticalReframer so PyInstaller --onefile builds
    always find it without bundling conflicts.
    """
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    cache_dir = os.path.join(local_appdata, "VerticalReframer")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, "face_detection_yunet_2023mar.onnx")

    if os.path.isfile(model_path) and os.path.getsize(model_path) > 100_000:
        return model_path

    # Also check next to the script/.exe
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    local_candidate = os.path.join(exe_dir, "face_detection_yunet_2023mar.onnx")
    if os.path.isfile(local_candidate) and os.path.getsize(local_candidate) > 100_000:
        return local_candidate

    # Download automatically from official OpenCV Zoo repository
    urllib.request.urlretrieve(YUNET_URL, model_path)
    return model_path


class VerticalCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("9:16 Vertical Reframer — YuNet DNN + Optical Flow Face Tracker")
        self.root.geometry("1220x820")
        self.root.configure(bg="#18181b")

        # Video state
        self.cap = None
        self.video_path = ""
        self.total_frames = 0
        self.fps = 30.0
        self.vid_w = 1920
        self.vid_h = 1080
        self.current_frame_idx = 0
        self.current_bgr_frame = None
        self.current_rgb_frame = None

        # Playback & Tracking toggles
        self.is_playing = False
        self.face_track_enabled = tk.BooleanVar(value=True)
        self.live_record = tk.BooleanVar(value=False)
        self.lock_vertical_pan = tk.BooleanVar(value=True)
        self.zoom_scale = tk.DoubleVar(value=1.0)
        self.follow_speed = tk.DoubleVar(value=0.55)  # Higher = tighter lock on fast left/right movement

        # Frame data:
        # keyframes: { frame_idx: (center_x, center_y, zoom) }
        # keyframe_types: { frame_idx: "manual" | "face" | "mouse" }
        self.keyframes = {}
        self.keyframe_types = {}

        # YuNet DNN + Lucas-Kanade Optical Flow state
        self.yunet = None
        self.yunet_status = "Initializing..."
        self._init_yunet_detector()

        self.detected_face_box = None       # (x, y, w, h, conf)
        self.detected_landmarks = []        # List of 5 (lx, ly) facial landmarks
        self.flow_points = None             # Nx1x2 float32 optical flow points on character
        self.prev_gray = None               # Previous grayscale frame for optical flow
        self.smooth_cx = None
        self.smooth_cy = None

        # Canvas display mapping
        self.canvas_w = 920
        self.canvas_h = 540
        self.disp_scale = 1.0
        self.off_x = 0
        self.off_y = 0

        self._build_ui()
        self._bind_shortcuts()

    def _init_yunet_detector(self):
        try:
            model_path = ensure_yunet_model()
            self.yunet = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (640, 360),
                0.55,  # Confidence threshold (catches side-profiles & angled faces)
                0.3,   # NMS threshold
                5000
            )
            self.yunet_status = "🧠 AI Ready: YuNet DNN + Optical Flow"
        except Exception as e:
            self.yunet = None
            self.yunet_status = f"⚠ Optical Flow Only ({str(e)[:28]})"

    def _build_ui(self):
        # Top Toolbar
        top_bar = tk.Frame(self.root, bg="#27272a", pady=8, padx=10)
        top_bar.pack(fill=tk.X)

        btn_style = {
            "bg": "#3f3f46", "fg": "white", "activebackground": "#52525b",
            "activeforeground": "white", "relief": tk.FLAT, "padx": 10, "pady": 5
        }

        tk.Button(top_bar, text="📂 Load Video", command=self.load_video, **btn_style).pack(side=tk.LEFT, padx=3)
        tk.Button(top_bar, text="🗑 Delete Frame Key", command=self.delete_current_keyframe, **btn_style).pack(side=tk.LEFT, padx=3)
        tk.Button(top_bar, text="🔄 Reset All", command=self.reset_all_keyframes, **btn_style).pack(side=tk.LEFT, padx=3)

        # Face Tracking Toggle
        face_chk = tk.Checkbutton(
            top_bar, text="🤖 Auto Face Track (Follow on Play)",
            variable=self.face_track_enabled, command=self._on_face_toggle,
            bg="#27272a", fg="#38bdf8", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="#38bdf8",
            font=("Segoe UI", 9, "bold")
        )
        face_chk.pack(side=tk.LEFT, padx=(12, 4))

        # Batch Auto-Track Button (tracks from current frame forward)
        tk.Button(
            top_bar, text="⚡ Auto-Track From Here", command=self.batch_track_from_current,
            bg="#0284c7", fg="white", activebackground="#0369a1",
            relief=tk.FLAT, padx=10, pady=5, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=4)

        # Horizontal-only Lock Toggle
        horiz_chk = tk.Checkbutton(
            top_bar, text="↔ Left/Right Only",
            variable=self.lock_vertical_pan,
            bg="#27272a", fg="#a1a1aa", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="white",
            font=("Segoe UI", 9)
        )
        horiz_chk.pack(side=tk.LEFT, padx=4)

        # Live Mouse Track Toggle
        rec_chk = tk.Checkbutton(
            top_bar, text="🔴 Live Mouse",
            variable=self.live_record, command=self._on_mouse_toggle,
            bg="#27272a", fg="#f87171", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="#ef4444",
            font=("Segoe UI", 9, "bold")
        )
        rec_chk.pack(side=tk.LEFT, padx=6)

        tk.Button(
            top_bar, text="🎬 Export 9:16 Video", command=self.export_video,
            bg="#2563eb", fg="white", activebackground="#1d4ed8",
            relief=tk.FLAT, padx=14, pady=5, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.RIGHT, padx=4)

        # Video Preview Canvas
        self.canvas = tk.Canvas(
            self.root, width=self.canvas_w, height=self.canvas_h,
            bg="#09090b", highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(pady=8, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_mouse)
        self.canvas.bind("<B1-Motion>", self.on_canvas_mouse)
        self.canvas.bind("<Motion>", self.on_canvas_hover)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Bottom Controls Panel
        ctrl_frame = tk.Frame(self.root, bg="#27272a", padx=14, pady=10)
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # Timeline Slider
        self.timeline_var = tk.IntVar(value=0)
        self.timeline = tk.Scale(
            ctrl_frame, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.timeline_var, command=self.on_timeline_scrub,
            bg="#27272a", fg="white", highlightthickness=0, troughcolor="#09090b",
            activebackground="#3b82f6"
        )
        self.timeline.pack(fill=tk.X, pady=(0, 8))

        # Transport & Sliders Row
        bottom_row = tk.Frame(ctrl_frame, bg="#27272a")
        bottom_row.pack(fill=tk.X)

        tk.Button(bottom_row, text="⏮ -10", command=lambda: self.step_frame(-10), **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_row, text="◀ -1 (A)", command=lambda: self.step_frame(-1), **btn_style).pack(side=tk.LEFT, padx=2)
        self.play_btn = tk.Button(bottom_row, text="▶ Play (Space)", command=self.toggle_play, width=14, **btn_style)
        self.play_btn.pack(side=tk.LEFT, padx=6)
        tk.Button(bottom_row, text="+1 (D) ▶", command=lambda: self.step_frame(1), **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_row, text="+10 ⏭", command=lambda: self.step_frame(10), **btn_style).pack(side=tk.LEFT, padx=2)

        # Zoom Slider
        tk.Label(bottom_row, text="Crop Size:", bg="#27272a", fg="#a1a1aa").pack(side=tk.LEFT, padx=(14, 4))
        zoom_slider = tk.Scale(
            bottom_row, from_=0.4, to=1.0, resolution=0.02, orient=tk.HORIZONTAL,
            variable=self.zoom_scale, command=self.on_zoom_change,
            bg="#27272a", fg="white", highlightthickness=0, length=100, showvalue=False
        )
        zoom_slider.pack(side=tk.LEFT)

        # Responsiveness Slider
        tk.Label(bottom_row, text="Track Speed:", bg="#27272a", fg="#a1a1aa").pack(side=tk.LEFT, padx=(14, 4))
        speed_slider = tk.Scale(
            bottom_row, from_=0.15, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self.follow_speed,
            bg="#27272a", fg="white", highlightthickness=0, length=100, showvalue=False
        )
        speed_slider.pack(side=tk.LEFT)

        self.status_lbl = tk.Label(
            bottom_row,
            text=f"{self.yunet_status} | Click character to lock target, then press Space",
            bg="#27272a", fg="#e4e4e7", font=("Consolas", 9)
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=6)

    def _on_face_toggle(self):
        if self.face_track_enabled.get():
            self.live_record.set(False)
            self._capture_character_anchor()

    def _on_mouse_toggle(self):
        if self.live_record.get():
            self.face_track_enabled.set(False)

    def _bind_shortcuts(self):
        self.root.bind("<Left>", lambda e: self.step_frame(-1))
        self.root.bind("<Right>", lambda e: self.step_frame(1))
        self.root.bind("a", lambda e: self.step_frame(-1))
        self.root.bind("d", lambda e: self.step_frame(1))
        self.root.bind("<Shift-Left>", lambda e: self.step_frame(-10))
        self.root.bind("<Shift-Right>", lambda e: self.step_frame(10))
        self.root.bind("<space>", lambda e: self.toggle_play())

    def on_canvas_resize(self, event):
        self.canvas_w = event.width
        self.canvas_h = event.height
        if self.current_rgb_frame is not None:
            self.render_preview()

    def load_video(self):
        path = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm"), ("All Files", "*.*")]
        )
        if not path:
            return

        if self.cap:
            self.cap.release()

        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.keyframes = {0: (self.vid_w / 2, self.vid_h / 2, 1.0)}
        self.keyframe_types = {0: "manual"}
        self.smooth_cx = self.vid_w / 2
        self.smooth_cy = self.vid_h / 2
        self.detected_face_box = None
        self.detected_landmarks = []
        self.flow_points = None
        self.prev_gray = None

        self.timeline.config(to=max(0, self.total_frames - 1))
        self.seek_frame(0)
        # Run initial face detection on Frame 0 so it immediately snaps to the character if visible
        self._auto_detect_initial_face()
        self._capture_character_anchor()
        self.render_preview()

    def _auto_detect_initial_face(self):
        """Finds the most prominent face on Frame 0 so the user starts locked onto the subject."""
        if self.current_bgr_frame is None or self.yunet is None:
            return
        faces = self._run_yunet(self.current_bgr_frame)
        if faces:
            # Pick largest / highest-confidence face near center
            best = max(faces, key=lambda f: f[2] * f[3] * f[14])
            fx, fy, fw, fh = best[0], best[1], best[2], best[3]
            nose_x, nose_y = best[8], best[9]
            cx = nose_x if (fx <= nose_x <= fx + fw) else (fx + fw / 2.0)
            cy = self.vid_h / 2.0 if self.lock_vertical_pan.get() else (fy + fh * 0.6)
            self.keyframes[0] = (cx, cy, 1.0)
            self.keyframe_types[0] = "manual"
            self.detected_face_box = (int(fx), int(fy), int(fw), int(fh), best[14])
            self.detected_landmarks = [
                (int(best[4]), int(best[5])),
                (int(best[6]), int(best[7])),
                (int(best[8]), int(best[9])),
                (int(best[10]), int(best[11])),
                (int(best[12]), int(best[13])),
            ]

    def get_crop_dimensions(self, zoom):
        target_aspect = 9.0 / 16.0
        if (self.vid_w / self.vid_h) >= target_aspect:
            base_h = self.vid_h
            base_w = self.vid_h * target_aspect
        else:
            base_w = self.vid_w
            base_h = self.vid_w / target_aspect

        crop_w = int(base_w * zoom)
        crop_h = int(base_h * zoom)
        crop_w = max(2, crop_w - (crop_w % 2))
        crop_h = max(2, crop_h - (crop_h % 2))
        return crop_w, crop_h

    def get_interpolated_state(self, frame_idx):
        if not self.keyframes:
            return self.vid_w / 2, self.vid_h / 2, 1.0

        if frame_idx in self.keyframes:
            return self.keyframes[frame_idx]

        sorted_keys = sorted(self.keyframes.keys())
        if frame_idx < sorted_keys[0]:
            return self.keyframes[sorted_keys[0]]
        if frame_idx > sorted_keys[-1]:
            return self.keyframes[sorted_keys[-1]]

        prev_k = max(k for k in sorted_keys if k < frame_idx)
        next_k = min(k for k in sorted_keys if k > frame_idx)

        ratio = (frame_idx - prev_k) / float(next_k - prev_k)
        x1, y1, z1 = self.keyframes[prev_k]
        x2, y2, z2 = self.keyframes[next_k]

        cx = x1 + (x2 - x1) * ratio
        cy = y1 + (y2 - y1) * ratio
        cz = z1 + (z2 - z1) * ratio
        return cx, cy, cz

    def get_clamped_box(self, cx, cy, zoom):
        crop_w, crop_h = self.get_crop_dimensions(zoom)
        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(cy - crop_h / 2))
        x1 = max(0, min(self.vid_w - crop_w, x1))
        y1 = max(0, min(self.vid_h - crop_h, y1))
        return x1, y1, crop_w, crop_h

    # =========================================================================
    # YUNET DNN + LUCAS-KANADE OPTICAL FLOW TRACKING ENGINE
    # =========================================================================
    def _run_yunet(self, bgr_frame):
        """Runs YuNet DNN on a BGR frame and returns scaled face detections."""
        if self.yunet is None:
            return []
        detect_w = 640
        scale = detect_w / float(self.vid_w)
        detect_h = max(32, int(round(self.vid_h * scale)))
        small_bgr = cv2.resize(bgr_frame, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)

        self.yunet.setInputSize((detect_w, detect_h))
        _, faces = self.yunet.detect(small_bgr)
        if faces is None or len(faces) == 0:
            return []

        results = []
        inv_scale = 1.0 / scale
        for row in faces:
            scaled = row.copy()
            scaled[0:14] *= inv_scale
            results.append(scaled)
        return results

    def _init_optical_flow_points(self, gray, center_x, center_y, box_w=None, box_h=None):
        """Initializes Shi-Tomasi corner points on the character's head/upper body for Optical Flow."""
        if box_w is None or box_h is None:
            box_w = max(80, int(self.vid_w * 0.14))
            box_h = max(100, int(self.vid_h * 0.22))

        x1 = max(0, int(center_x - box_w * 0.6))
        y1 = max(0, int(center_y - box_h * 0.6))
        x2 = min(self.vid_w, int(center_x + box_w * 0.6))
        y2 = min(self.vid_h, int(center_y + box_h * 0.8))

        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255

        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=48, qualityLevel=0.08, minDistance=6, mask=mask, blockSize=7
        )
        self.flow_points = pts

    def _capture_character_anchor(self):
        """Anchors both YuNet and Optical Flow to the character at the current crop center."""
        if self.current_bgr_frame is None:
            return
        cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
        self.smooth_cx = cx
        self.smooth_cy = cy

        gray = cv2.cvtColor(self.current_bgr_frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray

        # Check if YuNet sees a face right where the user clicked
        faces = self._run_yunet(self.current_bgr_frame)
        matched_face = self._select_character_face(faces, cx, cy, max_radius=self.vid_w * 0.25)
        if matched_face is not None:
            fx, fy, fw, fh = matched_face[0], matched_face[1], matched_face[2], matched_face[3]
            self.detected_face_box = (int(fx), int(fy), int(fw), int(fh), matched_face[14])
            self.detected_landmarks = [
                (int(matched_face[4]), int(matched_face[5])),
                (int(matched_face[6]), int(matched_face[7])),
                (int(matched_face[8]), int(matched_face[9])),
                (int(matched_face[10]), int(matched_face[11])),
                (int(matched_face[12]), int(matched_face[13])),
            ]
            self._init_optical_flow_points(gray, fx + fw / 2.0, fy + fh / 2.0, fw * 1.4, fh * 1.8)
        else:
            self.detected_face_box = None
            self.detected_landmarks = []
            self._init_optical_flow_points(gray, cx, cy)

    def _select_character_face(self, faces, ref_x, ref_y, max_radius):
        """Selects the face belonging to the tracked main character based on proximity and size."""
        if not faces:
            return None

        best_face = None
        best_cost = float("inf")

        for f in faces:
            fx, fy, fw, fh = f[0], f[1], f[2], f[3]
            conf = f[14]
            # Use nose tip (f[8], f[9]) or box center
            fcx = f[8] if (fx <= f[8] <= fx + fw) else (fx + fw / 2.0)
            fcy = f[9] if (fy <= f[9] <= fy + fh) else (fy + fh / 2.0)

            dist_x = abs(fcx - ref_x)
            dist_y = abs(fcy - ref_y) * 0.45  # Weight horizontal proximity higher
            dist = np.hypot(dist_x, dist_y)

            if dist <= max_radius:
                # Prefer close distance and high confidence
                cost = dist - (conf * 35.0)
                if cost < best_cost:
                    best_cost = cost
                    best_face = f

        # If only 1 clear high-confidence face is in the scene and we lost lock, re-acquire it
        if best_face is None and len(faces) == 1 and faces[0][14] >= 0.72:
            best_face = faces[0]

        return best_face

    def _track_character_on_current_frame(self):
        """
        Step 1: Predicts character motion using Lucas-Kanade Optical Flow.
        Step 2: Runs YuNet DNN Face Detection to lock onto exact facial landmarks (eyes/nose).
        Step 3: Smoothly updates the 9:16 crop center.
        """
        if self.current_bgr_frame is None:
            return

        prev_idx = max(0, self.current_frame_idx - 1)
        prev_cx, prev_cy, _ = self.get_interpolated_state(prev_idx)
        if self.smooth_cx is None:
            self.smooth_cx, self.smooth_cy = prev_cx, prev_cy

        gray = cv2.cvtColor(self.current_bgr_frame, cv2.COLOR_BGR2GRAY)

        # --- STEP 1: Pyramidal Lucas-Kanade Optical Flow ---
        flow_dx, flow_dy = 0.0, 0.0
        flow_valid = False
        if self.prev_gray is not None and self.flow_points is not None and len(self.flow_points) >= 6:
            lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
            )
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.flow_points, None, **lk_params)
            if p1 is not None and st is not None:
                good_new = p1[st == 1]
                good_old = self.flow_points[st == 1]
                if len(good_new) >= 5:
                    deltas = good_new - good_old
                    flow_dx = float(np.median(deltas[:, 0]))
                    flow_dy = float(np.median(deltas[:, 1]))
                    flow_valid = True
                    self.flow_points = good_new.reshape(-1, 1, 2)

        pred_cx = np.clip(self.smooth_cx + flow_dx, 0, self.vid_w)
        pred_cy = np.clip(self.smooth_cy + flow_dy, 0, self.vid_h)

        # --- STEP 2: YuNet Deep Learning Face Detection ---
        faces = self._run_yunet(self.current_bgr_frame)
        matched_face = self._select_character_face(faces, pred_cx, pred_cy, max_radius=self.vid_w * 0.35)

        target_x, target_y = None, None
        self.detected_face_box = None
        self.detected_landmarks = []

        if matched_face is not None:
            fx, fy, fw, fh = matched_face[0], matched_face[1], matched_face[2], matched_face[3]
            conf = matched_face[14]
            self.detected_face_box = (int(fx), int(fy), int(fw), int(fh), conf)

            # Extract 5 facial landmarks: right eye, left eye, nose tip, right mouth, left mouth
            self.detected_landmarks = [
                (int(matched_face[4]), int(matched_face[5])),
                (int(matched_face[6]), int(matched_face[7])),
                (int(matched_face[8]), int(matched_face[9])),
                (int(matched_face[10]), int(matched_face[11])),
                (int(matched_face[12]), int(matched_face[13])),
            ]

            # Center horizontally between the eyes and nose tip for rock-solid framing
            eye_mid_x = (matched_face[4] + matched_face[6]) / 2.0
            nose_x = matched_face[8]
            target_x = 0.5 * eye_mid_x + 0.5 * nose_x
            target_y = fy + fh * 0.62

            # Refresh optical flow points on the confirmed face + shoulders
            self._init_optical_flow_points(gray, target_x, fy + fh * 0.5, fw * 1.4, fh * 1.8)

        elif flow_valid:
            # Character turned away or face occluded: use Optical Flow displacement seamlessly
            target_x = pred_cx
            target_y = pred_cy
            if self.flow_points is None or len(self.flow_points) < 14:
                self._init_optical_flow_points(gray, target_x, target_y)

        self.prev_gray = gray

        # --- STEP 3: Responsive Camera Panning ---
        if target_x is not None:
            alpha = self.follow_speed.get()
            # Boost responsiveness automatically if the character makes a fast move
            dist_ratio = abs(target_x - self.smooth_cx) / float(self.vid_w)
            effective_alpha = min(1.0, alpha + dist_ratio * 1.5)

            deadzone_x = self.vid_w * 0.004
            if abs(target_x - self.smooth_cx) > deadzone_x:
                self.smooth_cx = self.smooth_cx + (target_x - self.smooth_cx) * effective_alpha

            if not self.lock_vertical_pan.get() and target_y is not None:
                deadzone_y = self.vid_h * 0.01
                if abs(target_y - self.smooth_cy) > deadzone_y:
                    self.smooth_cy = self.smooth_cy + (target_y - self.smooth_cy) * (effective_alpha * 0.5)

        self.keyframes[self.current_frame_idx] = (self.smooth_cx, self.smooth_cy, self.zoom_scale.get())
        self.keyframe_types[self.current_frame_idx] = "face"

    def batch_track_from_current(self):
        """Pre-tracks the character forward from the current frame until the end (or next manual keyframe)."""
        if not self.cap or self.current_bgr_frame is None:
            return
        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)")
        self._capture_character_anchor()

        start_f = self.current_frame_idx
        prog_win = tk.Toplevel(self.root)
        prog_win.title("AI Face Tracking in Progress...")
        prog_win.geometry("420x130")
        prog_win.configure(bg="#18181b")
        lbl = tk.Label(prog_win, text="Tracking character with YuNet DNN + Optical Flow...", bg="#18181b", fg="white")
        lbl.pack(pady=14)
        bar = ttk.Progressbar(prog_win, length=360, mode="determinate", maximum=self.total_frames - start_f)
        bar.pack(pady=4)

        cancel_flag = [False]
        tk.Button(
            prog_win, text="Stop Here", command=lambda: cancel_flag.__setitem__(0, True),
            bg="#ef4444", fg="white", relief=tk.FLAT, padx=10
        ).pack(pady=6)
        prog_win.update()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        for idx in range(start_f, self.total_frames):
            if cancel_flag[0]:
                break
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame_idx = idx
            self.current_bgr_frame = frame
            self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if idx > start_f:
                if self.keyframe_types.get(idx) == "manual":
                    self._capture_character_anchor()
                else:
                    self._track_character_on_current_frame()

            if (idx - start_f) % 15 == 0:
                bar["value"] = idx - start_f
                lbl.config(text=f"Tracking frame {idx} / {self.total_frames - 1}...")
                self.timeline_var.set(idx)
                self.render_preview()
                prog_win.update()

        prog_win.destroy()
        self.render_preview()

    def _retro_smooth_before_manual(self, manual_idx, new_cx, new_cy, new_zoom):
        """Smoothly blends the preceding ~12 auto-tracked frames into a manual pause correction."""
        blend_window = 12
        start_idx = max(0, manual_idx - blend_window)

        anchor_idx = start_idx
        for idx in range(manual_idx - 1, start_idx - 1, -1):
            if self.keyframe_types.get(idx) == "manual":
                anchor_idx = idx
                break

        if anchor_idx >= manual_idx - 1:
            return

        ax, ay, az = self.get_interpolated_state(anchor_idx)
        span = float(manual_idx - anchor_idx)
        for idx in range(anchor_idx + 1, manual_idx):
            if self.keyframe_types.get(idx) != "manual":
                t = (idx - anchor_idx) / span
                bx = ax + (new_cx - ax) * t
                by = ay + (new_cy - ay) * t
                bz = az + (new_zoom - az) * t
                self.keyframes[idx] = (bx, by, bz)

    # =========================================================================
    # SEEK, PREVIEW & INTERACTION
    # =========================================================================
    def seek_frame(self, idx):
        if not self.cap:
            return
        idx = max(0, min(self.total_frames - 1, idx))
        self.current_frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_bgr_frame = frame
            self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cx, cy, z = self.get_interpolated_state(idx)
            self.smooth_cx, self.smooth_cy = cx, cy
            self.zoom_scale.set(z)
            self.timeline_var.set(idx)
            self.detected_face_box = None
            self.detected_landmarks = []
            self.render_preview()

    def render_preview(self):
        if self.current_rgb_frame is None:
            return

        cx, cy, zoom = self.get_interpolated_state(self.current_frame_idx)
        x1, y1, cw, ch = self.get_clamped_box(cx, cy, zoom)
        x2, y2 = x1 + cw, y1 + ch

        preview = (self.current_rgb_frame * 0.32).astype(np.uint8)
        preview[y1:y2, x1:x2] = self.current_rgb_frame[y1:y2, x1:x2]

        # Draw Optical Flow tracking dots (green)
        if self.flow_points is not None:
            for pt in self.flow_points:
                px, py = int(pt[0][0]), int(pt[0][1])
                if 0 <= px < self.vid_w and 0 <= py < self.vid_h:
                    cv2.circle(preview, (px, py), max(2, int(self.vid_h / 320)), (74, 222, 128), -1)

        # Draw YuNet Face Bounding Box (yellow) + 5 Facial Landmarks (cyan)
        if self.detected_face_box is not None:
            fx, fy, fw, fh, conf = self.detected_face_box
            cv2.rectangle(preview, (fx, fy), (fx + fw, fy + fh), (250, 204, 21), max(2, int(self.vid_h / 340)))
            for (lx, ly) in self.detected_landmarks:
                cv2.circle(preview, (lx, ly), max(3, int(self.vid_h / 240)), (56, 189, 248), -1)

        k_type = self.keyframe_types.get(self.current_frame_idx, "interp")
        if k_type == "manual":
            border_color = (34, 197, 94)
            mode_tag = "🟢 MANUAL ADJUSTED"
        elif k_type == "face":
            border_color = (56, 189, 248)
            mode_tag = "🤖 YUNET FACE TRACKED"
        elif k_type == "mouse":
            border_color = (248, 113, 113)
            mode_tag = "🔴 MOUSE TRACKED"
        else:
            border_color = (99, 102, 241)
            mode_tag = "🔵 INTERPOLATED"

        cv2.rectangle(preview, (x1, y1), (x2, y2), border_color, max(2, int(self.vid_h / 260)))

        mid_x, mid_y = x1 + cw // 2, y1 + ch // 2
        cv2.drawMarker(preview, (mid_x, mid_y), border_color, cv2.MARKER_CROSS, 24, 2)

        self.disp_scale = min(self.canvas_w / self.vid_w, self.canvas_h / self.vid_h)
        disp_w = max(1, int(self.vid_w * self.disp_scale))
        disp_h = max(1, int(self.vid_h * self.disp_scale))
        self.off_x = (self.canvas_w - disp_w) // 2
        self.off_y = (self.canvas_h - disp_h) // 2

        resized = cv2.resize(preview, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        img = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.image = img
        self.canvas.create_image(self.off_x, self.off_y, anchor=tk.NW, image=img)

        self.status_lbl.config(
            text=f"Frame: {self.current_frame_idx}/{self.total_frames - 1} | {mode_tag} | {self.yunet_status}"
        )

    def _update_crop_from_mouse(self, event_x, event_y, source_type="manual"):
        if self.current_rgb_frame is None:
            return
        vx = (event_x - self.off_x) / self.disp_scale
        vy = (event_y - self.off_y) / self.disp_scale
        vx = max(0, min(self.vid_w, vx))
        vy = max(0, min(self.vid_h, vy))

        if self.lock_vertical_pan.get() and source_type == "manual":
            _, prev_cy, _ = self.get_interpolated_state(self.current_frame_idx)
            if abs(vy - prev_cy) < (self.vid_h * 0.08):
                vy = prev_cy

        zoom = self.zoom_scale.get()
        self.keyframes[self.current_frame_idx] = (vx, vy, zoom)
        self.keyframe_types[self.current_frame_idx] = source_type

        if source_type == "manual":
            self._retro_smooth_before_manual(self.current_frame_idx, vx, vy, zoom)
            self._capture_character_anchor()

        self.render_preview()

    def on_canvas_mouse(self, event):
        self._update_crop_from_mouse(event.x, event.y, source_type="manual")

    def on_canvas_hover(self, event):
        if self.is_playing and self.live_record.get():
            self._update_crop_from_mouse(event.x, event.y, source_type="mouse")

    def on_zoom_change(self, _):
        if self.current_rgb_frame is None:
            return
        cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
        self.keyframes[self.current_frame_idx] = (cx, cy, self.zoom_scale.get())
        self.keyframe_types[self.current_frame_idx] = "manual"
        self.render_preview()

    def on_timeline_scrub(self, val):
        idx = int(val)
        if idx != self.current_frame_idx:
            self.is_playing = False
            self.play_btn.config(text="▶ Play (Space)")
            self.seek_frame(idx)

    def step_frame(self, delta):
        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)")
        self.seek_frame(self.current_frame_idx + delta)

    def toggle_play(self):
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        self.play_btn.config(text="⏸ Pause (Space)" if self.is_playing else "▶ Play (Space)")
        if self.is_playing:
            self._capture_character_anchor()
            self._play_loop()

    def _play_loop(self):
        if not self.is_playing:
            return
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            ret, frame = self.cap.read()
            if ret:
                self.current_bgr_frame = frame
                self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.timeline_var.set(self.current_frame_idx)

                if self.face_track_enabled.get():
                    if self.keyframe_types.get(self.current_frame_idx) == "manual":
                        self._capture_character_anchor()
                    else:
                        self._track_character_on_current_frame()

                self.render_preview()
                delay = max(1, int(1000 / self.fps))
                self.root.after(delay, self._play_loop)
                return

        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)")

    def delete_current_keyframe(self):
        idx = self.current_frame_idx
        if idx in self.keyframes and len(self.keyframes) > 1:
            del self.keyframes[idx]
            self.keyframe_types.pop(idx, None)
            self.render_preview()

    def reset_all_keyframes(self):
        if self.cap and messagebox.askyesno("Reset", "Clear all tracked and manual frame positions?"):
            self.keyframes = {0: (self.vid_w / 2, self.vid_h / 2, 1.0)}
            self.keyframe_types = {0: "manual"}
            self.smooth_cx = self.vid_w / 2
            self.smooth_cy = self.vid_h / 2
            self.detected_face_box = None
            self.detected_landmarks = []
            self.flow_points = None
            self.render_preview()

    def export_video(self):
        if not self.cap:
            return
        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)")

        out_path = filedialog.asksaveasfilename(
            title="Save Vertical Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out_path:
            return

        out_w, out_h = 1080, 1920
        temp_video = out_path + ".temp_noaudio.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (out_w, out_h))

        prog_win = tk.Toplevel(self.root)
        prog_win.title("Rendering Vertical Video...")
        prog_win.geometry("420x120")
        prog_win.configure(bg="#18181b")
        prog_lbl = tk.Label(prog_win, text="Starting export...", bg="#18181b", fg="white")
        prog_lbl.pack(pady=15)
        prog_bar = ttk.Progressbar(prog_win, length=360, mode="determinate", maximum=self.total_frames)
        prog_bar.pack(pady=5)
        prog_win.update()

        export_cap = cv2.VideoCapture(self.video_path)
        for idx in range(self.total_frames):
            ret, frame = export_cap.read()
            if not ret:
                break

            cx, cy, zoom = self.get_interpolated_state(idx)
            x1, y1, cw, ch = self.get_clamped_box(cx, cy, zoom)
            cropped = frame[y1:y1 + ch, x1:x1 + cw]
            resized_vertical = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            writer.write(resized_vertical)

            if idx % 10 == 0:
                prog_bar["value"] = idx
                prog_lbl.config(text=f"Rendering frame {idx + 1} / {self.total_frames}...")
                prog_win.update()

        export_cap.release()
        writer.release()

        if shutil.which("ffmpeg"):
            prog_lbl.config(text="Merging original audio track via FFmpeg...")
            prog_win.update()
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_video,
                "-i", self.video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", out_path
            ]
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, creationflags=creation_flags
            )
            if os.path.exists(temp_video):
                os.remove(temp_video)
        else:
            os.replace(temp_video, out_path)

        prog_win.destroy()
        messagebox.showinfo("Done", f"Vertical video exported successfully:\n{out_path}")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = VerticalCropApp(root)
        root.mainloop()
    except Exception as e:
        err_root = tk.Tk()
        err_root.withdraw()
        messagebox.showerror("Application Startup Error", f"{e}\n\n{traceback.format_exc()}")