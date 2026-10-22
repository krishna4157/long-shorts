"""
track_crop.py

Landscape -> 9:16 using a smoothed, face-tracking crop window.

Can be used in two ways:

1. From command line:
   python track_crop.py -i input.mp4 -o output.mp4

2. Imported from another Python script:
   from track_crop import process_video

   process_video(
       input_path="input.mp4",
       output_path="output.mp4"
   )

Needs:
    pip install opencv-python numpy

Also requires:
    ffmpeg on PATH

And:
    haarcascade_frontalface_default.xml
    next to this script.
"""

import argparse
import os
import subprocess
import tempfile

import cv2
import numpy as np
from insightface.app import FaceAnalysis

def load_scrfd(det_size=(640, 640)):
    detector = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],
        providers=["CPUExecutionProvider"],
    )

    detector.prepare(
        ctx_id=-1,
        det_size=det_size,
    )

    return detector

def load_cascade():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "haarcascade_frontalface_default.xml",
    )

    cascade = cv2.CascadeClassifier(path)

    if cascade.empty():
        raise RuntimeError(
            f"Could not load face cascade: {path}\n"
            "Download it with:\n"
            "curl -L -o haarcascade_frontalface_default.xml "
            "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
            "haarcascade_frontalface_default.xml"
        )

    return cascade


def median_filter(a, window):
    if len(a) < 3 or window < 3:
        return a

    r = window // 2
    p = np.pad(a, r, mode="edge")

    return np.array([
        np.median(p[i:i + window])
        for i in range(len(a))
    ])


def gaussian_smooth(a, sigma):
    if len(a) < 3 or sigma <= 0:
        return a

    r = int(3 * sigma) + 1

    k = np.exp(
        -0.5 * (np.arange(-r, r + 1) / sigma) ** 2
    )

    k /= k.sum()

    return np.convolve(
        np.pad(a, r, mode="edge"),
        k,
        mode="valid"
    )


def analyze(
    path,
    detector,
    every,
    src_w,
    min_face_score=0.45):
    """
    Pass 1:
    Detect faces and scene cuts.

    Returns:
        xs
        cuts
        fps
        n_frames
    """

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    det_scale = 640 / src_w if src_w > 640 else 1.0

    xs = []
    cuts = [0]

    prev_hist = None
    i = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        small = cv2.resize(
            frame,
            None,
            fx=det_scale,
            fy=det_scale
        )

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        # Scene cut detection
        hist = cv2.calcHist(
            [gray],
            [0],
            None,
            [64],
            [0, 256]
        )

        cv2.normalize(hist, hist)

        if (
            prev_hist is not None
            and cv2.compareHist(
                prev_hist,
                hist,
                cv2.HISTCMP_CORREL
            ) < 0.6
        ):
            cuts.append(i)

        prev_hist = hist

        x = np.nan

        # Face detection
        # SCRFD face detection
    # ---------------------------------------------------------
    # SCRFD FACE DETECTION
    # Track up to TWO faces simultaneously
    # ---------------------------------------------------------
        if i % every == 0:

            faces = detector.get(small)

            # Keep only reasonably confident faces
            faces = [
                face
                for face in faces
                if float(face.det_score) >= min_face_score
            ]

            if faces:

                # Sort by face area, largest first
                faces = sorted(
                    faces,
                    key=lambda face: (
                        float(face.det_score)
                        *
                        np.sqrt(
                            max(
                                1.0,
                                float(face.bbox[2] - face.bbox[0])
                            )
                            *
                            max(
                                1.0,
                                float(face.bbox[3] - face.bbox[1])
                            )
                        )
                    ),
                    reverse=True
                )

                # Keep maximum 2 faces
                tracked_faces = faces[:2]

                centers_x = []

                for face in tracked_faces:

                    x1, y1, x2, y2 = face.bbox

                    center_x = (
                        float(x1) + float(x2)
                    ) / 2.0

                    centers_x.append(center_x)

                # -------------------------------------------------
                # ONE FACE
                # -------------------------------------------------

                if len(centers_x) == 1:

                    x = centers_x[0] / det_scale

                # -------------------------------------------------
                # TWO FACES
                # -------------------------------------------------

                else:

                    # Center between both faces
                    group_center_x = (
                        centers_x[0] + centers_x[1]
                    ) / 2.0

                    x = group_center_x / det_scale

        xs.append(x)

        i += 1

    cap.release()

    return (
        np.array(xs, dtype=float),
        cuts,
        fps,
        i
    )


def build_path(
    xs,
    cuts,
    fps,
    src_w,
    crop_w,
    smooth_seconds
):
    """
    Build a smooth per-frame horizontal camera path.
    """

    n = len(xs)

    centers = np.zeros(n)

    bounds = cuts + [n]

    last_cx = src_w / 2

    for s, e in zip(
        bounds[:-1],
        bounds[1:]
    ):

        seg = xs[s:e]

        idx = np.where(
            ~np.isnan(seg)
        )[0]

        if len(idx) == 0:

            path = np.full(
                e - s,
                last_cx
            )

        else:

            vals = median_filter(
                seg[idx],
                5
            )

            path = np.interp(
                np.arange(e - s),
                idx,
                vals
            )

            path = gaussian_smooth(
                path,
                smooth_seconds * fps
            )

        centers[s:e] = path

        last_cx = path[-1]

    return np.clip(
        centers,
        crop_w / 2,
        src_w - crop_w / 2
    )


def render(
    path,
    out_tmp,
    centers,
    crop_w,
    crop_h,
    top,
    out_w,
    out_h,
    fps
):

    cap = cv2.VideoCapture(path)

    writer = cv2.VideoWriter(
        out_tmp,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h)
    )

    scale = out_h / crop_h

    for i in range(len(centers)):

        ok, frame = cap.read()

        if not ok:
            break

        left = centers[i] - crop_w / 2

        M = np.float32([
            [
                scale,
                0,
                -left * scale
            ],
            [
                0,
                scale,
                -top * scale
            ]
        ])

        out = cv2.warpAffine(
            frame,
            M,
            (out_w, out_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        writer.write(out)

    cap.release()
    writer.release()


def process_video(
    input_path,
    output_path,
    width=1080,
    height=1920,
    smooth_seconds=0.6,
    every=2,
    zoom=1.0,
    y_center=0.5
):
    """
    Main reusable function.

    Converts a landscape video into a 9:16
    face-tracked vertical video.

    Parameters
    ----------
    input_path:
        Source MP4.

    output_path:
        Final vertical MP4.

    width:
        Output width. Default 1080.

    height:
        Output height. Default 1920.

    smooth_seconds:
        Camera movement smoothing.

    every:
        Detect faces every N frames.

    zoom:
        Crop zoom.

    y_center:
        Vertical crop position.
    """

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    print()
    print("========================================")
    print("Vertical Face-Tracking Crop")
    print("========================================")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print()

    if not os.path.isfile(input_path):
        raise FileNotFoundError(
            f"Input video does not exist: {input_path}"
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    src_w = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    src_h = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    if zoom < 1.0:
        raise ValueError(
            "zoom must be >= 1.0"
        )

    # Base 9:16 crop width
    base_crop_w = (
        src_h * width / height
    )

    if base_crop_w >= src_w:
        raise ValueError(
            "Source is already narrower than "
            "the target aspect ratio."
        )

    # Zoom
    crop_w = base_crop_w / zoom
    crop_h = src_h / zoom

    top = float(
        np.clip(
            y_center * src_h - crop_h / 2,
            0,
            src_h - crop_h
        )
    )

    cascade = load_cascade()

    print("Pass 1/2: analyzing...")
    detector = load_scrfd()

    xs, cuts, fps, n = analyze(
        input_path,
        detector,
        every,
        src_w,
        min_face_score=0.45,
    )

    print(
        f"  {n} frames, "
        f"{len(cuts) - 1} scene cut(s), "
        f"{int(np.sum(~np.isnan(xs)))} "
        f"face detections"
    )

    centers = build_path(
        xs,
        cuts,
        fps,
        src_w,
        crop_w,
        smooth_seconds
    )

    print("Pass 2/2: rendering...")

    tmp = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    ).name

    try:

        render(
            input_path,
            tmp,
            centers,
            crop_w,
            crop_h,
            top,
            width,
            height,
            fps
        )

        # Add original audio
        cmd = [
            "ffmpeg",
            "-y",

            "-i",
            tmp,

            "-i",
            input_path,

            "-map",
            "0:v:0",

            "-map",
            "1:a:0?",

            "-c:v",
            "libx264",

            "-pix_fmt",
            "yuv420p",

            "-crf",
            "20",

            "-c:a",
            "aac",

            "-shortest",

            output_path
        ]

        subprocess.run(
            cmd,
            check=True
        )

    finally:

        if os.path.exists(tmp):
            os.remove(tmp)

    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"Vertical output was not created: "
            f"{output_path}"
        )

    print()
    print(f"Done: {output_path}")
    print()

    return output_path


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-i",
        "--input",
        required=True
    )

    ap.add_argument(
        "-o",
        "--output",
        required=True
    )

    ap.add_argument(
        "--width",
        type=int,
        default=1080
    )

    ap.add_argument(
        "--height",
        type=int,
        default=1920
    )

    ap.add_argument(
        "--smooth-seconds",
        type=float,
        default=0.6
    )

    ap.add_argument(
        "--every",
        type=int,
        default=2
    )

    ap.add_argument(
        "--zoom",
        type=float,
        default=1.0
    )

    ap.add_argument(
        "--y-center",
        type=float,
        default=0.5
    )

    args = ap.parse_args()
    input_path = str(input_path)
    output_path = str(output_path)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if os.path.getsize(input_path) == 0:
        raise RuntimeError(f"Input video is empty: {input_path}")

    print(f"[TRACK-CROP] Input: {input_path}")
    print(f"[TRACK-CROP] Size: {os.path.getsize(input_path) / (1024 * 1024):.2f} MB")

    process_video(
        input_path=args.input,
        output_path=args.output,
        width=args.width,
        height=args.height,
        smooth_seconds=args.smooth_seconds,
        every=args.every,
        zoom=args.zoom,
        y_center=args.y_center
    )


if __name__ == "__main__":
    main()