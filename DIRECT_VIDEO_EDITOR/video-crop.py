import os
import sys
import traceback
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


def find_cascade_file(filename):
    """Safely locates OpenCV cascade XML files inside both Python and PyInstaller .exe environments."""
    search_dirs = []

    # 1. PyInstaller bundle temp directory (_MEIPASS)
    if hasattr(sys, "_MEIPASS"):
        search_dirs.append(os.path.join(sys._MEIPASS, "cv2", "data"))
        search_dirs.append(sys._MEIPASS)

    # 2. Standard cv2.data.haarcascades path
    cv2_data = getattr(cv2, "data", None)
    if cv2_data is not None and hasattr(cv2_data, "haarcascades"):
        search_dirs.append(cv2_data.haarcascades)

    # 3. Relative to cv2 module location
    if hasattr(cv2, "__file__") and cv2.__file__:
        cv2_dir = os.path.dirname(cv2.__file__)
        search_dirs.append(os.path.join(cv2_dir, "data"))

    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


class VerticalCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("9:16 Vertical Reframer + Smart Face Tracker")
        self.root.geometry("1160x800")
        self.root.configure(bg="#18181b")

        # Video state
        self.cap = None
        self.video_path = ""
        self.total_frames = 0
        self.fps = 30.0
        self.vid_w = 1920
        self.vid_h = 1080
        self.current_frame_idx = 0
        self.current_rgb_frame = None

        # Playback & Tracking toggles
        self.is_playing = False
        self.face_track_enabled = tk.BooleanVar(value=True)
        self.live_record = tk.BooleanVar(value=False)
        self.lock_vertical_pan = tk.BooleanVar(value=True)
        self.zoom_scale = tk.DoubleVar(value=1.0)

        # Frame data:
        # keyframes: { frame_idx: (center_x, center_y, zoom) }
        # keyframe_types: { frame_idx: "manual" | "face" | "mouse" }
        self.keyframes = {}
        self.keyframe_types = {}

        # Face & Character Tracker state
        self.frontal_cascade = None
        self.profile_cascade = None
        self._init_cascades()
        self.detected_face_box = None
        self.char_template = None
        self.prev_gray_full = None
        self.smooth_cx = None
        self.smooth_cy = None

        # Canvas display mapping
        self.canvas_w = 900
        self.canvas_h = 540
        self.disp_scale = 1.0
        self.off_x = 0
        self.off_y = 0

        self._build_ui()
        self._bind_shortcuts()

    def _init_cascades(self):
        try:
            f_path = find_cascade_file("haarcascade_frontalface_default.xml")
            if f_path:
                c = cv2.CascadeClassifier(f_path)
                if not c.empty():
                    self.frontal_cascade = c

            p_path = find_cascade_file("haarcascade_profileface.xml")
            if p_path:
                c = cv2.CascadeClassifier(p_path)
                if not c.empty():
                    self.profile_cascade = c
        except Exception:
            pass

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
        face_chk.pack(side=tk.LEFT, padx=(14, 6))

        # Horizontal-only Lock Toggle
        horiz_chk = tk.Checkbutton(
            top_bar, text="↔ Left/Right Pan Only",
            variable=self.lock_vertical_pan,
            bg="#27272a", fg="#a1a1aa", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="white",
            font=("Segoe UI", 9)
        )
        horiz_chk.pack(side=tk.LEFT, padx=4)

        # Live Mouse Track Toggle
        rec_chk = tk.Checkbutton(
            top_bar, text="🔴 Live Mouse Track",
            variable=self.live_record, command=self._on_mouse_toggle,
            bg="#27272a", fg="#f87171", selectcolor="#18181b",
            activebackground="#27272a", activeforeground="#ef4444",
            font=("Segoe UI", 9, "bold")
        )
        rec_chk.pack(side=tk.LEFT, padx=10)

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

        # Transport & Status Row
        bottom_row = tk.Frame(ctrl_frame, bg="#27272a")
        bottom_row.pack(fill=tk.X)

        tk.Button(bottom_row, text="⏮ -10", command=lambda: self.step_frame(-10), **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_row, text="◀ -1 (A)", command=lambda: self.step_frame(-1), **btn_style).pack(side=tk.LEFT, padx=2)
        self.play_btn = tk.Button(bottom_row, text="▶ Play (Space)", command=self.toggle_play, width=14, **btn_style)
        self.play_btn.pack(side=tk.LEFT, padx=6)
        tk.Button(bottom_row, text="+1 (D) ▶", command=lambda: self.step_frame(1), **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_row, text="+10 ⏭", command=lambda: self.step_frame(10), **btn_style).pack(side=tk.LEFT, padx=2)

        # Zoom Slider
        tk.Label(bottom_row, text="Crop Size:", bg="#27272a", fg="#a1a1aa").pack(side=tk.LEFT, padx=(18, 4))
        zoom_slider = tk.Scale(
            bottom_row, from_=0.4, to=1.0, resolution=0.02, orient=tk.HORIZONTAL,
            variable=self.zoom_scale, command=self.on_zoom_change,
            bg="#27272a", fg="white", highlightthickness=0, length=130, showvalue=False
        )
        zoom_slider.pack(side=tk.LEFT)

        self.status_lbl = tk.Label(
            bottom_row,
            text="Load a video | Space: Play/Pause | Click while paused to re-center character & resume tracking",
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
        self.char_template = None
        self.prev_gray_full = None

        self.timeline.config(to=max(0, self.total_frames - 1))
        self.seek_frame(0)
        self._capture_character_anchor()

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
    # FACE + CHARACTER TRACKING ENGINE
    # =========================================================================
    def _capture_character_anchor(self):
        """Locks onto the character at the current crop center."""
        if self.current_rgb_frame is None:
            return
        cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
        self.smooth_cx = cx
        self.smooth_cy = cy

        gray = cv2.cvtColor(self.current_rgb_frame, cv2.COLOR_RGB2GRAY)
        self.prev_gray_full = gray
        patch_r = max(24, int(min(self.vid_w, self.vid_h) * 0.07))
        ix, iy = int(round(cx)), int(round(cy))
        x1 = max(0, ix - patch_r)
        y1 = max(0, iy - patch_r)
        x2 = min(self.vid_w, ix + patch_r)
        y2 = min(self.vid_h, iy + patch_r)
        if (x2 - x1) > 16 and (y2 - y1) > 16:
            self.char_template = gray[y1:y2, x1:x2].copy()

    def _track_character_on_current_frame(self):
        """Tracks the main character via Haar Face Cascades + Template/Optical Flow fallback."""
        if self.current_rgb_frame is None:
            return

        prev_idx = max(0, self.current_frame_idx - 1)
        prev_cx, prev_cy, _ = self.get_interpolated_state(prev_idx)
        if self.smooth_cx is None:
            self.smooth_cx, self.smooth_cy = prev_cx, prev_cy

        gray_full = cv2.cvtColor(self.current_rgb_frame, cv2.COLOR_RGB2GRAY)

        # Downscale frame for fast real-time face detection
        detect_w = 640
        scale = detect_w / float(self.vid_w)
        detect_h = max(1, int(self.vid_h * scale))
        small_gray = cv2.resize(gray_full, (detect_w, detect_h), interpolation=cv2.INTER_LINEAR)
        small_gray = cv2.equalizeHist(small_gray)

        candidates = []
        min_face = max(20, int(detect_h * 0.06))

        try:
            # 1. Frontal faces
            if self.frontal_cascade is not None:
                faces = self.frontal_cascade.detectMultiScale(
                    small_gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_face, min_face)
                )
                for (x, y, w, h) in faces:
                    candidates.append((x / scale, y / scale, w / scale, h / scale))

            # 2. Left & Right Profile faces
            if not candidates and self.profile_cascade is not None:
                prof_left = self.profile_cascade.detectMultiScale(
                    small_gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_face, min_face)
                )
                for (x, y, w, h) in prof_left:
                    candidates.append((x / scale, y / scale, w / scale, h / scale))

                flipped = cv2.flip(small_gray, 1)
                prof_right = self.profile_cascade.detectMultiScale(
                    flipped, scaleFactor=1.15, minNeighbors=5, minSize=(min_face, min_face)
                )
                for (x, y, w, h) in prof_right:
                    orig_x = detect_w - (x + w)
                    candidates.append((orig_x / scale, y / scale, w / scale, h / scale))
        except Exception:
            pass

        target_x, target_y = None, None
        self.detected_face_box = None

        if candidates:
            best_face = None
            best_score = float("inf")
            max_jump = self.vid_w * 0.38

            for (fx, fy, fw, fh) in candidates:
                fcx = fx + fw / 2.0
                fcy = fy + fh / 2.0
                dist = np.hypot(fcx - self.smooth_cx, (fcy - self.smooth_cy) * 0.5)
                if dist < max_jump and dist < best_score:
                    best_score = dist
                    best_face = (fx, fy, fw, fh)

            if best_face is None and len(candidates) == 1:
                best_face = candidates[0]

            if best_face is not None:
                fx, fy, fw, fh = best_face
                self.detected_face_box = (int(fx), int(fy), int(fw), int(fh))
                target_x = fx + fw / 2.0
                target_y = fy + fh * 0.65

                ix, iy, iw, ih = int(fx), int(fy), int(fw), int(fh)
                x1, y1 = max(0, ix), max(0, iy)
                x2, y2 = min(self.vid_w, ix + iw), min(self.vid_h, iy + ih)
                if (x2 - x1) > 16 and (y2 - y1) > 16:
                    self.char_template = gray_full[y1:y2, x1:x2].copy()

        # 3. Fallback: Template matching around character if face turned away
        if target_x is None and self.char_template is not None:
            th, tw = self.char_template.shape[:2]
            search_margin_x = int(self.vid_w * 0.16)
            search_margin_y = int(self.vid_h * 0.10)

            sx1 = max(0, int(self.smooth_cx - tw // 2 - search_margin_x))
            sy1 = max(0, int(self.smooth_cy - th // 2 - search_margin_y))
            sx2 = min(self.vid_w, int(self.smooth_cx + tw // 2 + search_margin_x))
            sy2 = min(self.vid_h, int(self.smooth_cy + th // 2 + search_margin_y))

            roi = gray_full[sy1:sy2, sx1:sx2]
            if roi.shape[0] > th and roi.shape[1] > tw:
                res = cv2.matchTemplate(roi, self.char_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val > 0.55:
                    mx = sx1 + max_loc[0] + tw / 2.0
                    my = sy1 + max_loc[1] + th / 2.0
                    target_x, target_y = mx, my
                    self.detected_face_box = (int(sx1 + max_loc[0]), int(sy1 + max_loc[1]), tw, th)

        self.prev_gray_full = gray_full

        # 4. Smooth Camera Panning
        if target_x is not None:
            deadzone_x = self.vid_w * 0.012
            if abs(target_x - self.smooth_cx) > deadzone_x:
                alpha_x = 0.28
                self.smooth_cx = self.smooth_cx + (target_x - self.smooth_cx) * alpha_x

            if not self.lock_vertical_pan.get() and target_y is not None:
                deadzone_y = self.vid_h * 0.02
                if abs(target_y - self.smooth_cy) > deadzone_y:
                    self.smooth_cy = self.smooth_cy + (target_y - self.smooth_cy) * 0.18

        self.keyframes[self.current_frame_idx] = (self.smooth_cx, self.smooth_cy, self.zoom_scale.get())
        self.keyframe_types[self.current_frame_idx] = "face"

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
            self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cx, cy, z = self.get_interpolated_state(idx)
            self.smooth_cx, self.smooth_cy = cx, cy
            self.zoom_scale.set(z)
            self.timeline_var.set(idx)
            self.detected_face_box = None
            self.render_preview()

    def render_preview(self):
        if self.current_rgb_frame is None:
            return

        cx, cy, zoom = self.get_interpolated_state(self.current_frame_idx)
        x1, y1, cw, ch = self.get_clamped_box(cx, cy, zoom)
        x2, y2 = x1 + cw, y1 + ch

        preview = (self.current_rgb_frame * 0.32).astype(np.uint8)
        preview[y1:y2, x1:x2] = self.current_rgb_frame[y1:y2, x1:x2]

        if self.detected_face_box is not None:
            fx, fy, fw, fh = self.detected_face_box
            cv2.rectangle(preview, (fx, fy), (fx + fw, fy + fh), (250, 204, 21), max(2, int(self.vid_h / 350)))

        k_type = self.keyframe_types.get(self.current_frame_idx, "interp")
        if k_type == "manual":
            border_color = (34, 197, 94)
            mode_tag = "🟢 MANUAL ADJUSTED"
        elif k_type == "face":
            border_color = (56, 189, 248)
            mode_tag = "🤖 FACE TRACKED"
        elif k_type == "mouse":
            border_color = (248, 113, 113)
            mode_tag = "🔴 MOUSE TRACKED"
        else:
            border_color = (99, 102, 241)
            mode_tag = "🔵 INTERPOLATED"

        cv2.rectangle(preview, (x1, y1), (x2, y2), border_color, max(2, int(self.vid_h / 280)))

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
            text=f"Frame: {self.current_frame_idx}/{self.total_frames - 1} | {mode_tag} | Saved Frames: {len(self.keyframes)}"
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