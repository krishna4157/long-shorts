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
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)


def ensure_ai_models():
    """
    Ensures both YuNet (face detector) and SFace (128-D biometric face recognizer)
    are downloaded into %LOCALAPPDATA%/VerticalReframer.
    """
    local_appdata = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    cache_dir = os.path.join(local_appdata, "VerticalReframer")
    os.makedirs(cache_dir, exist_ok=True)

    yunet_path = os.path.join(cache_dir, "face_detection_yunet_2023mar.onnx")
    sface_path = os.path.join(cache_dir, "face_recognition_sface_2021dec.onnx")

    if not (os.path.isfile(yunet_path) and os.path.getsize(yunet_path) > 100_000):
        urllib.request.urlretrieve(YUNET_URL, yunet_path)

    if not (os.path.isfile(sface_path) and os.path.getsize(sface_path) > 10_000_000):
        urllib.request.urlretrieve(SFACE_URL, sface_path)

    return yunet_path, sface_path


class VerticalCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("9:16 Vertical Reframer — Specific Person Identity Lock (YuNet + SFace)")
        self.root.geometry("1260x840")
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
        self.center_when_missing = tk.BooleanVar(value=True)  # Return to center if target face not found
        self.live_record = tk.BooleanVar(value=False)
        self.lock_vertical_pan = tk.BooleanVar(value=True)
        self.zoom_scale = tk.DoubleVar(value=1.0)
        self.follow_speed = tk.DoubleVar(value=0.50)
        self.sim_threshold = tk.DoubleVar(value=0.36)  # OpenCV SFace official cosine threshold is 0.363

        # Frame data:
        # keyframes: { frame_idx: (center_x, center_y, zoom) }
        # keyframe_types: { frame_idx: "manual" | "face" | "center_fallback" | "mouse" }
        self.keyframes = {}
        self.keyframe_types = {}

        # AI Models: YuNet Detector + SFace Recognizer
        self.yunet = None
        self.sface = None
        self.ai_status = "Initializing AI models..."
        self._init_ai_models()

        # Specific Person Identity State
        self.target_embedding = None      # 128-D SFace feature vector for the chosen main character
        self.target_thumb_tk = None       # Thumbnail image of locked person shown in toolbar
        self.missing_frames_count = 0     # How many consecutive frames the target face was absent
        self.frame_face_results = []      # [(x, y, w, h, score, is_match, landmarks), ...] for UI overlay
        self.smooth_cx = None
        self.smooth_cy = None

        # Canvas display mapping
        self.canvas_w = 940
        self.canvas_h = 540
        self.disp_scale = 1.0
        self.off_x = 0
        self.off_y = 0

        self._build_ui()
        self._bind_shortcuts()

    def _init_ai_models(self):
        try:
            yunet_path, sface_path = ensure_ai_models()
            self.yunet = cv2.FaceDetectorYN.create(
                yunet_path, "", (640, 360), 0.55, 0.3, 5000
            )
            self.sface = cv2.FaceRecognizerSF.create(sface_path, "")
            self.ai_status = "🧠 YuNet + SFace Identity Lock Ready"
        except Exception as e:
            self.ai_status = f"⚠ AI Load Error: {str(e)[:35]}"

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

        # Target Identity Lock Panel
        id_frame = tk.Frame(top_bar, bg="#18181b", padx=8, pady=3, highlightthickness=1, highlightbackground="#3f3f46")
        id_frame.pack(side=tk.LEFT, padx=10)

        self.thumb_lbl = tk.Label(id_frame, text="No Face\nLocked", bg="#09090b", fg="#a1a1aa", width=7, height=2, font=("Segoe UI", 8))
        self.thumb_lbl.pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            id_frame, text="🎯 Lock Target Face", command=self.lock_face_near_crop_center,
            bg="#16a34a", fg="white", activebackground="#15803d",
            relief=tk.FLAT, padx=8, pady=4, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            id_frame, text="✖ Clear ID", command=self.clear_locked_identity,
            bg="#3f3f46", fg="#f87171", activebackground="#52525b",
            relief=tk.FLAT, padx=6, pady=4, font=("Segoe UI", 8)
        ).pack(side=tk.LEFT, padx=2)

        # Face Tracking Toggle
        face_chk = tk.Checkbutton(
            top_bar, text="🤖 Track Locked Person",
            variable=self.face_track_enabled, command=self._on_face_toggle,
            bg="#27272a", fg="#38bdf8", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="#38bdf8",
            font=("Segoe UI", 9, "bold")
        )
        face_chk.pack(side=tk.LEFT, padx=4)

        # Auto-Center when Missing Toggle
        center_chk = tk.Checkbutton(
            top_bar, text="🎯 Center If Missing",
            variable=self.center_when_missing,
            bg="#27272a", fg="#fde047", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="#fde047",
            font=("Segoe UI", 9, "bold")
        )
        center_chk.pack(side=tk.LEFT, padx=4)

        # Batch Auto-Track Button
        tk.Button(
            top_bar, text="⚡ Auto-Track From Here", command=self.batch_track_from_current,
            bg="#0284c7", fg="white", activebackground="#0369a1",
            relief=tk.FLAT, padx=10, pady=5, font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT, padx=6)

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

        # Left/Right Lock
        tk.Checkbutton(
            bottom_row, text="↔ L/R Only", variable=self.lock_vertical_pan,
            bg="#27272a", fg="#a1a1aa", selectcolor="#18181b", activebackground="#27272a"
        ).pack(side=tk.LEFT, padx=8)

        # Zoom Slider
        tk.Label(bottom_row, text="Crop Size:", bg="#27272a", fg="#a1a1aa").pack(side=tk.LEFT, padx=(8, 2))
        tk.Scale(
            bottom_row, from_=0.4, to=1.0, resolution=0.02, orient=tk.HORIZONTAL,
            variable=self.zoom_scale, command=self.on_zoom_change,
            bg="#27272a", fg="white", highlightthickness=0, length=85, showvalue=False
        ).pack(side=tk.LEFT)

        # Match Strictness Slider
        tk.Label(bottom_row, text="ID Strictness:", bg="#27272a", fg="#a1a1aa").pack(side=tk.LEFT, padx=(10, 2))
        tk.Scale(
            bottom_row, from_=0.22, to=0.55, resolution=0.02, orient=tk.HORIZONTAL,
            variable=self.sim_threshold,
            bg="#27272a", fg="white", highlightthickness=0, length=85, showvalue=False
        ).pack(side=tk.LEFT)

        self.status_lbl = tk.Label(
            bottom_row,
            text=f"{self.ai_status} | Click character's face to lock their identity!",
            bg="#27272a", fg="#e4e4e7", font=("Consolas", 9)
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=6)

    def _on_face_toggle(self):
        if self.face_track_enabled.get():
            self.live_record.set(False)

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

        # Default to exact center
        self.keyframes = {0: (self.vid_w / 2.0, self.vid_h / 2.0, 1.0)}
        self.keyframe_types = {0: "manual"}
        self.smooth_cx = self.vid_w / 2.0
        self.smooth_cy = self.vid_h / 2.0
        self.missing_frames_count = 0
        self.frame_face_results = []

        self.timeline.config(to=max(0, self.total_frames - 1))
        self.seek_frame(0)

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
            return self.vid_w / 2.0, self.vid_h / 2.0, 1.0

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
    # SPECIFIC PERSON FACE RECOGNITION (YUNET + SFACE 128-D EMBEDDING)
    # =========================================================================
    def _run_yunet(self, bgr_frame):
        """Detects all faces in the frame and returns scaled YuNet rows (15 values each)."""
        if self.yunet is None or bgr_frame is None:
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
            scaled = row.copy().astype(np.float32)
            scaled[0:14] *= inv_scale
            results.append(scaled)
        return results

    def _extract_sface_embedding(self, bgr_frame, face_row):
        """Aligns the detected face using its 5 landmarks and extracts a 128-D biometric feature vector."""
        if self.sface is None:
            return None
        try:
            aligned_face = self.sface.alignCrop(bgr_frame, face_row)
            feat = self.sface.feature(aligned_face)
            return feat
        except Exception:
            return None

    def _compute_similarity(self, feat1, feat2):
        """Computes Cosine Similarity between two 128-D SFace embeddings (higher = same person)."""
        if self.sface is None or feat1 is None or feat2 is None:
            return 0.0
        try:
            return float(self.sface.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))
        except Exception:
            return 0.0

    def lock_face_near_point(self, target_vx, target_vy, silent_if_none=False):
        """
        Finds the face closest to (target_vx, target_vy), locks its 128-D identity embedding,
        and displays the locked face thumbnail in the top toolbar.
        """
        if self.current_bgr_frame is None:
            return False

        faces = self._run_yunet(self.current_bgr_frame)
        if not faces:
            if not silent_if_none:
                messagebox.showwarning("No Face Found", "No face detected on this frame. Step to a frame where your character's face is visible and click on it.")
            return False

        # Find the face whose bounding box contains or is closest to the clicked point
        best_face = None
        best_dist = float("inf")
        for f in faces:
            fx, fy, fw, fh = f[0], f[1], f[2], f[3]
            fcx, fcy = fx + fw / 2.0, fy + fh / 2.0
            dist = np.hypot(fcx - target_vx, fcy - target_vy)
            # Check if click was inside or reasonably near this face
            if dist < max(fw * 1.6, self.vid_w * 0.18) and dist < best_dist:
                best_dist = dist
                best_face = f

        if best_face is None:
            if not silent_if_none:
                # If user clicked "Lock Target Face" button without clicking directly on a face, pick closest to crop center
                best_face = min(faces, key=lambda f: np.hypot((f[0] + f[2] / 2.0) - target_vx, (f[1] + f[3] / 2.0) - target_vy))
            else:
                return False

        feat = self._extract_sface_embedding(self.current_bgr_frame, best_face)
        if feat is None:
            return False

        self.target_embedding = feat
        self.missing_frames_count = 0

        # Update toolbar thumbnail with the locked character's face
        fx, fy, fw, fh = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
        pad = int(max(fw, fh) * 0.15)
        x1 = max(0, fx - pad)
        y1 = max(0, fy - pad)
        x2 = min(self.vid_w, fx + fw + pad)
        y2 = min(self.vid_h, fy + fh + pad)
        face_crop_rgb = self.current_rgb_frame[y1:y2, x1:x2]
        if face_crop_rgb.size > 0:
            thumb_img = cv2.resize(face_crop_rgb, (44, 44), interpolation=cv2.INTER_AREA)
            self.target_thumb_tk = ImageTk.PhotoImage(Image.fromarray(thumb_img))
            self.thumb_lbl.config(image=self.target_thumb_tk, text="", width=44, height=44)

        # Evaluate faces on current frame immediately so green MATCH box shows up
        self._evaluate_faces_on_current_frame()
        return True

    def lock_face_near_crop_center(self):
        """Toolbar button handler: locks onto the face closest to the current crop center."""
        if self.current_bgr_frame is None:
            return
        cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
        if self.lock_face_near_point(cx, cy, silent_if_none=False):
            self.render_preview()

    def clear_locked_identity(self):
        self.target_embedding = None
        self.target_thumb_tk = None
        self.thumb_lbl.config(image="", text="No Face\nLocked", width=7, height=2)
        self.frame_face_results = []
        self.render_preview()

    def _evaluate_faces_on_current_frame(self):
        """
        Detects all faces on the current frame and checks which one matches `self.target_embedding`.
        Populates `self.frame_face_results` for visual preview and returns the matched face row (or None).
        """
        self.frame_face_results = []
        if self.current_bgr_frame is None:
            return None

        faces = self._run_yunet(self.current_bgr_frame)
        if not faces:
            return None

        thresh = self.sim_threshold.get()
        best_match_face = None
        best_match_sim = -1.0

        # If no specific person is locked yet, auto-lock onto the face closest to current crop center
        if self.target_embedding is None:
            cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
            self.lock_face_near_point(cx, cy, silent_if_none=True)

        # First pass: score every face against the locked target embedding
        scored_faces = []
        for f in faces:
            fx, fy, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            landmarks = [
                (int(f[4]), int(f[5])),
                (int(f[6]), int(f[7])),
                (int(f[8]), int(f[9])),
                (int(f[10]), int(f[11])),
                (int(f[12]), int(f[13])),
            ]
            sim = 0.0
            if self.target_embedding is not None:
                feat = self._extract_sface_embedding(self.current_bgr_frame, f)
                sim = self._compute_similarity(self.target_embedding, feat)

                # Slight spatial continuity bonus if the face is right where our tracked character was last frame
                if self.smooth_cx is not None and self.missing_frames_count < 8:
                    fcx = f[0] + f[2] / 2.0
                    dist_ratio = abs(fcx - self.smooth_cx) / float(self.vid_w)
                    if dist_ratio < 0.12:
                        sim += 0.04  # Helps during extreme head turns/talking

            scored_faces.append((f, fx, fy, fw, fh, sim, landmarks))
            if sim >= thresh and sim > best_match_sim:
                best_match_sim = sim
                best_match_face = f

        for (f, fx, fy, fw, fh, sim, landmarks) in scored_faces:
            is_match = (best_match_face is not None) and (f is best_match_face)
            self.frame_face_results.append((fx, fy, fw, fh, sim, is_match, landmarks))

        return best_match_face

    def _track_character_on_current_frame(self):
        """
        Tracks ONLY the locked character's face.
        If the locked face is NOT found, smoothly returns the crop to the exact center of the frame.
        """
        if self.current_bgr_frame is None:
            return

        prev_idx = max(0, self.current_frame_idx - 1)
        prev_cx, prev_cy, _ = self.get_interpolated_state(prev_idx)
        if self.smooth_cx is None:
            self.smooth_cx, self.smooth_cy = prev_cx, prev_cy

        matched_face = self._evaluate_faces_on_current_frame()
        alpha = self.follow_speed.get()

        if matched_face is not None:
            # Target person FOUND -> follow their face
            self.missing_frames_count = 0
            fx, fy, fw, fh = matched_face[0], matched_face[1], matched_face[2], matched_face[3]

            # Center horizontally between the eyes and nose tip
            eye_mid_x = (matched_face[4] + matched_face[6]) / 2.0
            nose_x = matched_face[8]
            target_x = 0.5 * eye_mid_x + 0.5 * nose_x
            target_y = (self.vid_h / 2.0) if self.lock_vertical_pan.get() else (fy + fh * 0.62)

            dist_ratio = abs(target_x - self.smooth_cx) / float(self.vid_w)
            effective_alpha = min(1.0, alpha + dist_ratio * 1.4)

            if abs(target_x - self.smooth_cx) > (self.vid_w * 0.004):
                self.smooth_cx = self.smooth_cx + (target_x - self.smooth_cx) * effective_alpha

            if not self.lock_vertical_pan.get():
                self.smooth_cy = self.smooth_cy + (target_y - self.smooth_cy) * (effective_alpha * 0.5)
            else:
                self.smooth_cy = self.vid_h / 2.0

            self.keyframes[self.current_frame_idx] = (self.smooth_cx, self.smooth_cy, self.zoom_scale.get())
            self.keyframe_types[self.current_frame_idx] = "face"

        else:
            # Target person NOT FOUND on this frame
            self.missing_frames_count += 1
            center_x = self.vid_w / 2.0
            center_y = self.vid_h / 2.0

            if self.center_when_missing.get():
                # Wait 3 frames before centering so a 1-frame blink doesn't jerk the camera,
                # then smoothly glide to the center of the frame
                if self.missing_frames_count >= 3:
                    center_speed = min(1.0, alpha * 0.45)
                    self.smooth_cx = self.smooth_cx + (center_x - self.smooth_cx) * center_speed
                    self.smooth_cy = self.smooth_cy + (center_y - self.smooth_cy) * center_speed
                    # Snap cleanly when very close to center
                    if abs(self.smooth_cx - center_x) < 3.0:
                        self.smooth_cx = center_x
                    if abs(self.smooth_cy - center_y) < 3.0:
                        self.smooth_cy = center_y

            self.keyframes[self.current_frame_idx] = (self.smooth_cx, self.smooth_cy, self.zoom_scale.get())
            self.keyframe_types[self.current_frame_idx] = "center_fallback"

    def batch_track_from_current(self):
        """Rapidly tracks the locked character from the current frame forward."""
        if not self.cap or self.current_bgr_frame is None:
            return
        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)")

        if self.target_embedding is None:
            cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
            if not self.lock_face_near_point(cx, cy, silent_if_none=False):
                return

        start_f = self.current_frame_idx
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Tracking Specific Person...")
        prog_win.geometry("440x130")
        prog_win.configure(bg="#18181b")
        lbl = tk.Label(prog_win, text="Matching locked face identity (YuNet + SFace)...", bg="#18181b", fg="white")
        lbl.pack(pady=14)
        bar = ttk.Progressbar(prog_win, length=380, mode="determinate", maximum=self.total_frames - start_f)
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
                    cx, cy, _ = self.keyframes[idx]
                    self.smooth_cx, self.smooth_cy = cx, cy
                    self.lock_face_near_point(cx, cy, silent_if_none=True)
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
            if self.target_embedding is not None:
                self._evaluate_faces_on_current_frame()
            else:
                self.frame_face_results = []
            self.render_preview()

    def render_preview(self):
        if self.current_rgb_frame is None:
            return

        cx, cy, zoom = self.get_interpolated_state(self.current_frame_idx)
        x1, y1, cw, ch = self.get_clamped_box(cx, cy, zoom)
        x2, y2 = x1 + cw, y1 + ch

        preview = (self.current_rgb_frame * 0.32).astype(np.uint8)
        preview[y1:y2, x1:x2] = self.current_rgb_frame[y1:y2, x1:x2]

        # Draw all detected faces: GREEN = Target Match, RED = Ignored Other Person
        font_scale = max(0.5, self.vid_h / 1400.0)
        thickness = max(2, int(self.vid_h / 340))
        for (fx, fy, fw, fh, sim, is_match, landmarks) in self.frame_face_results:
            if is_match:
                box_col = (34, 197, 94)  # Bright Green for locked person
                label = f"TARGET ({sim:.2f})"
                for (lx, ly) in landmarks:
                    cv2.circle(preview, (lx, ly), max(3, int(self.vid_h / 240)), (56, 189, 248), -1)
            else:
                box_col = (239, 68, 68)  # Red for ignored background/side people
                label = f"IGNORED ({sim:.2f})"

            cv2.rectangle(preview, (fx, fy), (fx + fw, fy + fh), box_col, thickness)
            cv2.putText(
                preview, label, (fx, max(20, fy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_col, thickness, cv2.LINE_AA
            )

        k_type = self.keyframe_types.get(self.current_frame_idx, "interp")
        if k_type == "manual":
            border_color = (34, 197, 94)
            mode_tag = "🟢 MANUAL / LOCKED"
        elif k_type == "face":
            border_color = (56, 189, 248)
            mode_tag = "🎯 TARGET FACE FOUND"
        elif k_type == "center_fallback":
            border_color = (250, 204, 21)
            mode_tag = "⚪ NO TARGET FACE -> CENTERED"
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

        id_state = "LOCKED" if self.target_embedding is not None else "NOT LOCKED (Click Character Face)"
        self.status_lbl.config(
            text=f"Frame: {self.current_frame_idx}/{self.total_frames - 1} | {mode_tag} | Identity: {id_state}"
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
        self.smooth_cx, self.smooth_cy = vx, vy

        if source_type == "manual":
            self._retro_smooth_before_manual(self.current_frame_idx, vx, vy, zoom)
            # If user clicks directly on a face while paused, automatically lock/update that person's identity!
            self.lock_face_near_point(vx, vy, silent_if_none=True)

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
            # Ensure we have a locked face identity if a face is under the current crop box
            if self.target_embedding is None:
                cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
                self.lock_face_near_point(cx, cy, silent_if_none=True)
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
                        cx, cy, _ = self.keyframes[self.current_frame_idx]
                        self.smooth_cx, self.smooth_cy = cx, cy
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
            self.keyframes = {0: (self.vid_w / 2.0, self.vid_h / 2.0, 1.0)}
            self.keyframe_types = {0: "manual"}
            self.smooth_cx = self.vid_w / 2.0
            self.smooth_cy = self.vid_h / 2.0
            self.missing_frames_count = 0
            self.frame_face_results = []
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