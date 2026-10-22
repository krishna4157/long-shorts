"""
track-crop.py - Landscape -> 9:16 using a smoothed, face-tracking crop window.

Two passes:
  1. Detect faces + scene cuts for the whole video.
  2. Smooth the camera path (median + Gaussian, looking ahead and behind),
     then render with sub-pixel cropping.

Usage:
    python track-crop.py -i input.mp4 -o output.mp4
    python track-crop.py -i input.mp4 -o output.mp4 --smooth-seconds 1.0

Needs: pip install opencv-python numpy, ffmpeg on PATH, and
haarcascade_frontalface_default.xml next to this script.
"""
import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np


def load_cascade():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "haarcascade_frontalface_default.xml",
    )
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise SystemExit(
            f"Could not load face cascade: {path}\n"
            "Download it with:\n  curl -L -o haarcascade_frontalface_default.xml "
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
            "haarcascade_frontalface_default.xml"
        )
    return cascade


def median_filter(a, window):
    if len(a) < 3 or window < 3:
        return a
    r = window // 2
    p = np.pad(a, r, mode="edge")
    return np.array([np.median(p[i:i + window]) for i in range(len(a))])


def gaussian_smooth(a, sigma):
    if len(a) < 3 or sigma <= 0:
        return a
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode="edge"), k, mode="valid")


def analyze(path, cascade, every, src_w):
    """Pass 1: return (face_x per frame with NaN gaps, list of scene-cut frame indices, fps, n_frames)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    det_scale = 640 / src_w if src_w > 640 else 1.0

    xs, cuts, prev_hist, i = [], [0], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=det_scale, fy=det_scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Scene cut detection via histogram correlation
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.6:
            cuts.append(i)
        prev_hist = hist

        x = np.nan
        if i % every == 0:
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces):
                fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                x = (fx + fw / 2) / det_scale
        xs.append(x)
        i += 1
    cap.release()
    return np.array(xs, dtype=float), cuts, fps, i


def build_path(xs, cuts, fps, src_w, crop_w, smooth_seconds):
    """Turn sparse detections into a smooth per-frame camera center."""
    n = len(xs)
    centers = np.zeros(n)
    bounds = cuts + [n]
    last_cx = src_w / 2

    for s, e in zip(bounds[:-1], bounds[1:]):
        seg = xs[s:e]
        idx = np.where(~np.isnan(seg))[0]
        if len(idx) == 0:
            path = np.full(e - s, last_cx)  # no face in this shot: hold position
        else:
            vals = median_filter(seg[idx], 5)               # remove outlier detections
            path = np.interp(np.arange(e - s), idx, vals)   # fill every frame
            path = gaussian_smooth(path, smooth_seconds * fps)
        centers[s:e] = path
        last_cx = path[-1]

    return np.clip(centers, crop_w / 2, src_w - crop_w / 2)


# def render(path, out_tmp, centers, crop_w, out_w, out_h, fps, src_h):
#     cap = cv2.VideoCapture(path)
#     writer = cv2.VideoWriter(out_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
#     scale = out_h / src_h
#     for i in range(len(centers)):
#         ok, frame = cap.read()
#         if not ok:
#             break
#         left = centers[i] - crop_w / 2
#         # Sub-pixel crop + scale in one affine transform (no integer snapping)
#         M = np.float32([[scale, 0, -left * scale], [0, scale, 0]])
#         out = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
#                              borderMode=cv2.BORDER_REPLICATE)
#         writer.write(out)
#     cap.release()
#     writer.release()
def render(path, out_tmp, centers, crop_w, crop_h, top, out_w, out_h, fps):
    cap = cv2.VideoCapture(path)
    writer = cv2.VideoWriter(out_tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    scale = out_h / crop_h  # same as out_w / crop_w
    for i in range(len(centers)):
        ok, frame = cap.read()
        if not ok:
            break
        left = centers[i] - crop_w / 2
        # Sub-pixel crop + zoom in one affine transform
        M = np.float32([[scale, 0, -left * scale],
                        [0, scale, -top * scale]])
        out = cv2.warpAffine(frame, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
        writer.write(out)
    cap.release()
    writer.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--smooth-seconds", type=float, default=0.6,
                    help="Camera smoothing strength (higher = slower, smoother pans)")
    ap.add_argument("--every", type=int, default=2,
                    help="Detect faces every N frames")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="Manual zoom factor (>=1.0). Higher = tighter crop, removes black bars")
    ap.add_argument("--y-center", type=float, default=0.5,
                    help="Vertical center of the zoomed window (0=top, 1=bottom)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.input}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if args.zoom < 1.0:
        raise SystemExit("--zoom must be >= 1.0")

    base_crop_w = src_h * args.width / args.height
    if base_crop_w >= src_w:
        raise SystemExit("Source is already narrower than the target aspect ratio.")

    # Zoom shrinks the window in both dimensions, keeping the 9:16 shape
    crop_w = base_crop_w / args.zoom
    crop_h = src_h / args.zoom
    top = float(np.clip(args.y_center * src_h - crop_h / 2, 0, src_h - crop_h))


    cascade = load_cascade()
    print("Pass 1/2: analyzing...")
    xs, cuts, fps, n = analyze(args.input, cascade, args.every, src_w)
    print(f"  {n} frames, {len(cuts) - 1} scene cut(s), {int(np.sum(~np.isnan(xs)))} face detections")

    centers = build_path(xs, cuts, fps, src_w, crop_w, args.smooth_seconds)

    print("Pass 2/2: rendering...")
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    render(args.input, tmp, centers, crop_w, crop_h, top, args.width, args.height, fps)

    cmd = [
        "ffmpeg", "-y", "-i", tmp, "-i", args.input,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-c:a", "aac", "-shortest", args.output,
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        os.remove(tmp)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()