import os
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class VerticalCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("9:16 Frame-by-Frame Vertical Reframer")
        self.root.geometry("1100x780")
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

        # Playback & Recording state
        self.is_playing = False
        self.live_record = tk.BooleanVar(value=False)
        self.zoom_scale = tk.DoubleVar(value=1.0)  # 1.0 = full height 9:16

        # Keyframes dictionary: { frame_idx: (center_x, center_y, zoom) }
        # Stores manual frame customizations
        self.keyframes = {}

        # Canvas mapping state
        self.canvas_w = 880
        self.canvas_h = 520
        self.disp_scale = 1.0
        self.off_x = 0
        self.off_y = 0

        self._build_ui()
        self._bind_shortcuts()

    def _build_ui(self):
        # Top Toolbar
        top_bar = tk.Frame(self.root, bg="#27272a", pady=8, padx=12)
        top_bar.pack(fill=tk.X)

        btn_style = {"bg": "#3f3f46", "fg": "white", "activebackground": "#52525b",
                     "activeforeground": "white", "relief": tk.FLAT, "padx": 12, "pady": 5}

        tk.Button(top_bar, text="📂 Load Video", command=self.load_video, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(top_bar, text="🗑 Delete Frame Point", command=self.delete_current_keyframe, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(top_bar, text="🔄 Reset All Points", command=self.reset_all_keyframes, **btn_style).pack(side=tk.LEFT, padx=4)

        # Live Record Checkbox
        rec_chk = tk.Checkbutton(
            top_bar, text="🔴 Live Mouse Track (Record Every Frame on Play)",
            variable=self.live_record, bg="#27272a", fg="#f87171",
            selectcolor="#18181b", activebackground="#27272a", activeforeground="#ef4444",
            font=("Segoe UI", 9, "bold")
        )
        rec_chk.pack(side=tk.LEFT, padx=16)

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
        self.canvas.pack(pady=10, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Button-1>", self.on_canvas_mouse)
        self.canvas.bind("<B1-Motion>", self.on_canvas_mouse)
        self.canvas.bind("<Motion>", self.on_canvas_hover)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Controls Panel
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

        # Bottom Row: Transport + Zoom + Status
        bottom_row = tk.Frame(ctrl_frame, bg="#27272a")
        bottom_row.pack(fill=tk.X)

        tk.Button(bottom_row, text="⏮ -10", command=lambda: self.step_frame(-10), **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_row, text="◀ -1 (A)", command=lambda: self.step_frame(-1), **btn_style).pack(side=tk.LEFT, padx=2)
        self.play_btn = tk.Button(bottom_row, text="▶ Play (Space)", command=self.toggle_play, width=14, **btn_style)
        self.play_btn.pack(side=tk.LEFT, padx=6)
        tk.Button(bottom_row, text="+1 (D) ▶", command=lambda: self.step_frame(1), **btn_style).pack(side=tk.LEFT, padx=2)
        tk.Button(bottom_row, text="+10 ⏭", command=lambda: self.step_frame(10), **btn_style).pack(side=tk.LEFT, padx=2)

        # Zoom control
        tk.Label(bottom_row, text="Crop Size:", bg="#27272a", fg="#a1a1aa").pack(side=tk.LEFT, padx=(20, 4))
        zoom_slider = tk.Scale(
            bottom_row, from_=0.4, to=1.0, resolution=0.02, orient=tk.HORIZONTAL,
            variable=self.zoom_scale, command=self.on_zoom_change,
            bg="#27272a", fg="white", highlightthickness=0, length=140, showvalue=False
        )
        zoom_slider.pack(side=tk.LEFT)

        self.status_lbl = tk.Label(
            bottom_row, text="Load a video to start | Click/Drag to position crop | A/D or Arrows to step frames",
            bg="#27272a", fg="#e4e4e7", font=("Consolas", 9)
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=8)

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

        # Initialize default center keyframe at frame 0
        self.keyframes = {0: (self.vid_w / 2, self.vid_h / 2, 1.0)}
        self.timeline.config(to=max(0, self.total_frames - 1))
        self.seek_frame(0)

    def get_crop_dimensions(self, zoom):
        # Base 9:16 box that fits inside the video
        target_aspect = 9.0 / 16.0
        if (self.vid_w / self.vid_h) >= target_aspect:
            base_h = self.vid_h
            base_w = self.vid_h * target_aspect
        else:
            base_w = self.vid_w
            base_h = self.vid_w / target_aspect

        crop_w = int(base_w * zoom)
        crop_h = int(base_h * zoom)
        # Ensure even numbers for hardware encoders
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

        # Find surrounding keyframes for smooth linear interpolation
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

    def seek_frame(self, idx):
        if not self.cap:
            return
        idx = max(0, min(self.total_frames - 1, idx))
        self.current_frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, _, z = self.get_interpolated_state(idx)
            self.zoom_scale.set(z)
            self.timeline_var.set(idx)
            self.render_preview()

    def render_preview(self):
        if self.current_rgb_frame is None:
            return

        cx, cy, zoom = self.get_interpolated_state(self.current_frame_idx)
        x1, y1, cw, ch = self.get_clamped_box(cx, cy, zoom)
        x2, y2 = x1 + cw, y1 + ch

        # Dim outside area to highlight the 9:16 crop window
        preview = (self.current_rgb_frame * 0.32).astype(np.uint8)
        preview[y1:y2, x1:x2] = self.current_rgb_frame[y1:y2, x1:x2]

        # Draw border & rule-of-thirds grid
        is_manual = self.current_frame_idx in self.keyframes
        border_color = (34, 197, 94) if is_manual else (59, 130, 246)  # Green = manual keyframe, Blue = interpolated
        cv2.rectangle(preview, (x1, y1), (x2, y2), border_color, max(2, int(self.vid_h / 300)))

        # Center crosshair
        mid_x, mid_y = x1 + cw // 2, y1 + ch // 2
        cv2.drawMarker(preview, (mid_x, mid_y), border_color, cv2.MARKER_CROSS, 24, 2)

        # Scale to fit canvas
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

        mode_tag = "🟢 MANUAL FRAME" if is_manual else "🔵 AUTO-INTERPOLATED"
        self.status_lbl.config(
            text=f"Frame: {self.current_frame_idx}/{self.total_frames - 1} | {mode_tag} | Keys set: {len(self.keyframes)}"
        )

    def _update_crop_from_mouse(self, event_x, event_y):
        if self.current_rgb_frame is None:
            return
        vx = (event_x - self.off_x) / self.disp_scale
        vy = (event_y - self.off_y) / self.disp_scale
        vx = max(0, min(self.vid_w, vx))
        vy = max(0, min(self.vid_h, vy))
        self.keyframes[self.current_frame_idx] = (vx, vy, self.zoom_scale.get())
        self.render_preview()

    def on_canvas_mouse(self, event):
        self._update_crop_from_mouse(event.x, event.y)

    def on_canvas_hover(self, event):
        if self.is_playing and self.live_record.get():
            self._update_crop_from_mouse(event.x, event.y)

    def on_zoom_change(self, _):
        if self.current_rgb_frame is None:
            return
        cx, cy, _ = self.get_interpolated_state(self.current_frame_idx)
        self.keyframes[self.current_frame_idx] = (cx, cy, self.zoom_scale.get())
        self.render_preview()

    def on_timeline_scrub(self, val):
        idx = int(val)
        if idx != self.current_frame_idx:
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
                self.render_preview()
                delay = max(1, int(1000 / self.fps))
                self.root.after(delay, self._play_loop)
                return
        self.is_playing = False
        self.play_btn.config(text="▶ Play (Space)")

    def delete_current_keyframe(self):
        if self.current_frame_idx in self.keyframes and len(self.keyframes) > 1:
            del self.keyframes[self.current_frame_idx]
            self.render_preview()

    def reset_all_keyframes(self):
        if self.cap and messagebox.askyesno("Reset", "Clear all custom frame positions?"):
            self.keyframes = {0: (self.vid_w / 2, self.vid_h / 2, 1.0)}
            self.render_preview()

    def export_video(self):
        if not self.cap:
            return
        self.is_playing = False

        out_path = filedialog.asksaveasfilename(
            title="Save Vertical Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")]
        )
        if not out_path:
            return

        # Standard 1080x1920 vertical HD output
        out_w, out_h = 1080, 1920
        temp_video = out_path + ".temp_noaudio.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video, fourcc, self.fps, (out_w, out_h))

        # Progress Popup
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

        # Merge original audio if FFmpeg is installed on system PATH
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
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(temp_video):
                os.remove(temp_video)
        else:
            os.replace(temp_video, out_path)

        prog_win.destroy()
        messagebox.showinfo("Done", f"Vertical video exported successfully:\n{out_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = VerticalCropApp(root)
    root.mainloop()