# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
import subprocess
from pathlib import Path
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.genai import types
import static_ffmpeg

# Initialize ffmpeg binary paths
static_ffmpeg.add_paths()

import yt_dlp

# OUTPUT_DIR = Path("/config/Desktop/Session1/test-short/output_shorts")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path(r"D:\GIT\long-shorts\output-shorts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Saving to:", OUTPUT_DIR.resolve())

COOKIES_FILE = Path(r"D:\GIT\long-shorts\cookies.txt")
COOKIES_BROWSER = "firefox"  # used only if cookies.txt doesn't exist

def yt_auth_opts() -> dict:
    """Cookie options for the yt_dlp Python API."""
    if COOKIES_FILE.exists():
        return {"cookiefile": str(COOKIES_FILE)}
    return {"cookiesfrombrowser": (COOKIES_BROWSER,)}

def yt_auth_args() -> list:
    """Cookie flags for the yt_dlp command line."""
    if COOKIES_FILE.exists():
        return ["--cookies", str(COOKIES_FILE)]
    return ["--cookies-from-browser", COOKIES_BROWSER]

def format_seconds(seconds: float) -> str:
    """Formats seconds into MM:SS format."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def format_timestamp_window(start_sec: float, end_sec: float) -> str:
    """Formats start and end seconds into string timestamp range (e.g. '01:30 - 01:45')."""
    return f"{format_seconds(start_sec)} - {format_seconds(end_sec)}"


def gather_top_trending_videos(count: int = 10) -> str:
    """Gathers top trending or popular viral YouTube videos.

    Args:
        count: Number of top trending videos to fetch (default: 10).

    Returns:
        JSON string containing the list of top trending videos with titles, URLs, and IDs.
    """
    search_query = f"ytsearch{count}:trending viral videos"
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        # "js_runtimes": {"node": {"path": "/usr/bin/node"}},
        "remote_components": ["ejs:github"],
    }

    results = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            entries = info.get("entries", [])
            for idx, entry in enumerate(entries[:count], 1):
                vid_id = entry.get("id")
                title = entry.get("title")
                url = f"https://www.youtube.com/watch?v={vid_id}"
                results.append({
                    "rank": idx,
                    "title": title,
                    "id": vid_id,
                    "url": url,
                    "uploader": entry.get("uploader", "Unknown"),
                })
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error gathering trending videos: {str(e)}"

# def get_most_replayed_scenes(video_url: str, max_scenes: int = 3, min_score: float = 0.0) -> str:
#     """Extracts the YouTube 'most replayed' heatmap and returns the top distinct scenes.

#     Args:
#         video_url: Full YouTube video URL.
#         max_scenes: Maximum number of scenes to return (default: 3).
#         min_score: Optional minimum heatmap value (default: 0.0, no threshold).

#     Returns:
#         JSON string with scenes (start/end seconds, score, chapter).
#     """
#     ydl_opts = {
#         "quiet": True,
#         "skip_download": True,
#         **yt_auth_opts(),
#     }

#     MIN_GAP = 30     # minimum seconds between chosen scenes
#     CLIP_LEN = 30    # length of each short in seconds

#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             info = ydl.extract_info(video_url, download=False)
#             heatmap = info.get("heatmap") or []
#             duration = info.get("duration", 0)
#             title = info.get("title", "")
#             chapters = info.get("chapters") or []

#             if not heatmap:
#                 return json.dumps({"video": title, "url": video_url, "scenes": []})

#             skip_sec = min(15, duration * 0.10)
#             candidates = [p for p in heatmap if p.get("start_time", 0) >= skip_sec] or heatmap
#             candidates = [p for p in candidates if p.get("value", 0) >= min_score]
#             candidates.sort(key=lambda p: p.get("value", 0), reverse=True)

#             picked = []
#             for p in candidates:
#                 center = (p.get("start_time", 0) + p.get("end_time", 0)) / 2
#                 if all(abs(center - c) >= MIN_GAP for c, _ in picked):
#                     picked.append((center, p))
#                 if len(picked) >= max_scenes:
#                     break

#             scenes = []
#             for center, p in sorted(picked, key=lambda x: x[0]):
#                 st = max(0, center - CLIP_LEN / 2)
#                 et = min(duration, st + CLIP_LEN)
#                 chapter_title = "N/A"
#                 for c in chapters:
#                     if c.get("start_time", 0) <= st <= c.get("end_time", 0):
#                         chapter_title = c.get("title", "N/A")
#                         break
#                 scenes.append({
#                     "start_sec": round(st, 1),
#                     "end_sec": round(et, 1),
#                     "time_window": format_timestamp_window(st, et),
#                     "heatmap_score": round(p.get("value", 0), 2),
#                     "chapter": chapter_title,
#                 })

#             return json.dumps({
#                 "video": title,
#                 "url": video_url,
#                 "duration": format_seconds(duration),
#                 "scenes": scenes,
#             }, indent=2)

#     except Exception as e:
#         return f"Error analyzing heatmap for video {video_url}: {str(e)}"

def get_most_replayed_scenes(video_url: str, max_scenes: int = 3, min_score: float = 0.25) -> str:
    """Extracts YouTube audience retention heatmap data for a video and identifies peak most replayed scenes.

    Args:
        video_url: Full YouTube video URL.
        max_scenes: Maximum number of peak replayed scenes to return per video (default: 3).
        min_score: Minimum relative heatmap score threshold for a peak (default: 0.25).

    Returns:
        JSON string listing the detected peak replayed scenes with start/end seconds, scores, and chapters.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        # "js_runtimes": {"node": {"path": "/usr/bin/node"}},
        # "remote_components": ["ejs:github"],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            heatmap = info.get("heatmap") or []
            duration = info.get("duration", 0)
            title = info.get("title", "")
            chapters = info.get("chapters") or []

            if not heatmap:
                return json.dumps({"video": title, "url": video_url, "scenes": []})

            # Filter out initial video intro baseline (first 15 seconds or 10% of video)
            skip_sec = min(15, duration * 0.10)
            filtered = [p for p in heatmap if p.get("start_time", 0) >= skip_sec]

            if not filtered:
                filtered = heatmap

            # Detect local maxima peaks
            local_peaks = []
            for i in range(len(filtered)):
                val = filtered[i].get("value", 0)
                prev_val = filtered[i - 1].get("value", 0) if i > 0 else 0
                next_val = filtered[i + 1].get("value", 0) if i < len(filtered) - 1 else 0

                if val >= min_score and val >= prev_val and val >= next_val:
                    local_peaks.append(filtered[i])

            # Fallback to overall highest point if no local peaks above threshold
            if not local_peaks:
                local_peaks = [max(filtered, key=lambda x: x.get("value", 0))]

            # Sort peaks by score descending
            local_peaks.sort(key=lambda x: x.get("value", 0), reverse=True)

            # Deduplicate peaks closer than 20 seconds
            deduped_peaks = []
            for peak in local_peaks:
                t = peak.get("start_time", 0)
                if not any(abs(t - d.get("start_time", 0)) < 20 for d in deduped_peaks):
                    deduped_peaks.append(peak)
                if len(deduped_peaks) >= max_scenes:
                    break

            scenes = []
            for p in deduped_peaks:
                st = p.get("start_time", 0)
                et = p.get("end_time", st + 15)
                # Ensure minimum 10s scene window for shorts clip
                if (et - st) < 10:
                    et = st + 10

                # Match chapter title if available
                chapter_title = "N/A"
                for c in chapters:
                    if c.get("start_time", 0) <= st <= c.get("end_time", 0):
                        chapter_title = c.get("title", "N/A")
                        break

                scenes.append({
                    "start_sec": st,
                    "end_sec": et,
                    "time_window": format_timestamp_window(st, et),
                    "heatmap_score": round(p.get("value", 0), 2),
                    "chapter": chapter_title
                })

            return json.dumps({
                "video": title,
                "url": video_url,
                "duration": format_seconds(duration),
                "scenes": scenes
            }, indent=2)

    except Exception as e:
        return f"Error analyzing heatmap for video {video_url}: {str(e)}"


# def crop_replayed_scene_to_mp4(video_url: str, start_time: float, end_time: float, output_filename: str) -> str:
#     """Crops a specific timestamp scene from a YouTube video into an MP4 file.

#     Args:
#         video_url: Full YouTube video URL.
#         start_time: Start timestamp in seconds.
#         end_time: End timestamp in seconds.
#         output_filename: Output MP4 filename (saved under test-short/output_shorts/).

#     Returns:
#         Status message with the path to the cropped MP4 file.
#     """
#     import sys
#     import shutil
#     import static_ffmpeg

#     static_ffmpeg.add_paths()
#     ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
#     output_path = OUTPUT_DIR / output_filename
#     temp_video = OUTPUT_DIR / f"temp_{output_path.stem}.mp4"

#     # Step 1: Download video stream with fast 360p format filter & quick timeouts
#     dl_cmd = [
#         sys.executable,
#         "-m",
#         "yt_dlp",
#         "--quiet",
#         "--no-warnings",
#         "--retries", "2",
#         "--fragment-retries", "1",
#         "--socket-timeout", "10",
#         "--format", "18/134+140/395+140/best[height<=360]/best",
#         "-o", str(temp_video),
#         video_url
#     ]

#     try:
#         subprocess.run(dl_cmd, check=True)

#         # Locate downloaded temp file (might have .mp4, .webm, or .mkv extension)
#         matching_temps = list(OUTPUT_DIR.glob(f"temp_{output_path.stem}*"))
#         if not matching_temps or matching_temps[0].stat().st_size == 0:
#             return f"Failed to download base video for {video_url}"

#         actual_temp = matching_temps[0]

#         # Step 2: Crop scene segment locally using ffmpeg
#         duration = max(1.0, end_time - start_time)
#         crop_cmd = [
#             ffmpeg_bin,
#             "-y",
#             "-ss", str(start_time),
#             "-t", str(duration),
#             "-i", str(actual_temp),
#             "-c", "copy",
#             str(output_path)
#         ]
#         subprocess.run(crop_cmd, check=True, capture_output=True)

#         # Cleanup temporary full video download files
#         for tmp in matching_temps:
#             if tmp.exists():
#                 try:
#                     tmp.unlink()
#                 except Exception:
#                     pass

#         if output_path.exists() and output_path.stat().st_size > 0:
#             size_mb = output_path.stat().st_size / (1024 * 1024)
#             return f"Successfully created MP4 short: {output_path} (Size: {size_mb:.2f} MB)"
#         else:
#             return f"Failed to verify cropped MP4 file at {output_path}"
#     except Exception as e:
#         matching_temps = list(OUTPUT_DIR.glob(f"temp_{output_path.stem}*"))
#         for tmp in matching_temps:
#             if tmp.exists():
#                 try:
#                     tmp.unlink()
#                 except Exception:
#                     pass
#         return f"Error cropping scene to MP4 for {video_url}: {str(e)}"
def crop_replayed_scene_to_mp4(video_url: str, start_time: float, end_time: float, output_filename: str, max_height: int = 1080) -> str:
    """Crops a specific timestamp scene from a YouTube video into a high-quality MP4 file.

    Args:
        video_url: Full YouTube video URL.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        output_filename: Output MP4 filename.
        max_height: Maximum video height to download (default: 1080).

    Returns:
        Status message with the path to the cropped MP4 file.
    """
    import sys
    import shutil

    static_ffmpeg.add_paths()
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
    output_path = OUTPUT_DIR / output_filename
    temp_stem = f"temp_{output_path.stem}"
    temp_template = str(OUTPUT_DIR / f"{temp_stem}.%(ext)s")

    # Best video + best audio up to max_height, merged into mp4
    fmt = (
        f"bv*[height<={max_height}]+ba/"
        f"b[height<={max_height}]/best"
    )

    # Step 1: download ONLY the needed section (much faster than the full video)
    dl_cmd = [
        sys.executable, "-m",
        "yt_dlp",
        *yt_auth_args(),    
        "--quiet",
        "--no-warnings",
        "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", "15",
        "--format", 
        fmt,
        "--merge-output-format", "mkv",   # mkv container accepts any codec combo
        "--download-sections", f"*{start_time}-{end_time}",
        "-o", temp_template,
        video_url,
    ]

    def cleanup():
        for tmp in OUTPUT_DIR.glob(f"{temp_stem}*"):
            try:
                tmp.unlink()
            except Exception:
                pass

    try:
        subprocess.run(dl_cmd, check=True)

        matching_temps = list(OUTPUT_DIR.glob(f"{temp_stem}*"))
        if not matching_temps or matching_temps[0].stat().st_size == 0:
            return f"Failed to download base video for {video_url}"
        actual_temp = matching_temps[0]

        # Step 2: re-encode to a widely compatible, high-quality MP4
        encode_cmd = [
            ffmpeg_bin, "-y",
            "-i", str(actual_temp),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]
        subprocess.run(encode_cmd, check=True, capture_output=True)
        cleanup()

        if output_path.exists() and output_path.stat().st_size > 0:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            return f"Successfully created MP4 short: {output_path} (Size: {size_mb:.2f} MB)"
        return f"Failed to verify cropped MP4 file at {output_path}"
    except Exception as e:
        cleanup()
        return f"Error cropping scene to MP4 for {video_url}: {str(e)}"

def create_shorts_from_video_url(video_url: str, max_scenes: int = 3) -> str:
    """Processes a specific YouTube video URL, extracts its peak most replayed scenes from the YouTube heatmap,
    and crops those scenes into MP4 format video shorts.

    Args:
        video_url: Full YouTube video URL or ID.
        max_scenes: Maximum number of replayed scenes to crop for this video (default: 3).

    Returns:
        Structured text report detailing all generated MP4 video shorts for the provided URL.
    """
    if not video_url.startswith("http"):
        video_url = f"https://www.youtube.com/watch?v={video_url}"

    scenes_data_str = get_most_replayed_scenes(video_url, max_scenes=max_scenes)
    try:
        scenes_info = json.loads(scenes_data_str)
        v_title = scenes_info.get("video", "YouTube_Video")
        scenes = scenes_info.get("scenes", [])
    except Exception as e:
        return f"Failed to analyze scenes for {video_url}: {str(e)}"

    if not scenes:
        scenes = [{
            "start_sec": 10,
            "end_sec": 25,
            "time_window": "00:10 - 00:25",
            "heatmap_score": 0.0,
            "chapter": "Fallback Section"
        }]

    # Extract video ID for clean output naming
    v_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
    v_id = v_id_match.group(1) if v_id_match else "custom_vid"

    shorts_created = []
    for s_idx, scene in enumerate(scenes, 1):
        st = scene["start_sec"]
        et = scene["end_sec"]
        filename = f"short_url_{v_id}_{s_idx:02d}_{int(st)}s_{int(et)}s.mp4"

        crop_res = crop_replayed_scene_to_mp4(
            video_url=video_url,
            start_time=st,
            end_time=et,
            output_filename=filename
        )

        out_path = OUTPUT_DIR / filename
        file_exists = out_path.exists()
        file_size_mb = (out_path.stat().st_size / (1024 * 1024)) if file_exists else 0.0

        shorts_created.append({
            "video_title": v_title,
            "video_url": video_url,
            "scene_index": s_idx,
            "time_window": scene["time_window"],
            "heatmap_score": scene["heatmap_score"],
            "chapter": scene.get("chapter", "N/A"),
            "mp4_file": str(out_path),
            "filename": filename,
            "size_mb": round(file_size_mb, 2),
            "status": "Success" if file_exists else "Failed"
        })

    report_lines = [
        f"# Most Replayed Video Shorts Report",
        f"Processed video **{v_title}** (`{video_url}`) and generated **{len(shorts_created)}** MP4 video shorts.\n",
        "## Summary of Cropped Shorts (MP4 Format):",
    ]

    for item in shorts_created:
        report_lines.append(
            f"- **{item['video_title']}**\n"
            f"  - **Scene #{item['scene_index']} Window**: {item['time_window']} (Score: {item['heatmap_score']})\n"
            f"  - **Chapter**: {item['chapter']}\n"
            f"  - **Output MP4 File**: `{item['filename']}` ({item['size_mb']} MB)\n"
            f"  - **Full Path**: `{item['mp4_file']}`\n"
        )

    return "\n".join(report_lines)


def create_most_replayed_video_shorts(video_url: str = None, top_video_count: int = 10, max_scenes_per_video: int = 2) -> str:
    """Master workflow tool to gather top 10 trending videos OR process a provided YouTube URL,
    find all most replayed scenes for each video, and crop those scenes into MP4 format shorts.

    Args:
        video_url: Optional specific YouTube video URL. If provided, processes this specific video instead of trending videos.
        top_video_count: Number of top trending videos to process if video_url is not provided (default: 10).
        max_scenes_per_video: Maximum number of replayed scenes to crop per video (default: 2).

    Returns:
        Structured text report detailing all generated MP4 video shorts.
    """
    if video_url:
        return create_shorts_from_video_url(video_url=video_url, max_scenes=max_scenes_per_video)

    # Step 1: Gather top 10 trending videos
    trending_data_str = gather_top_trending_videos(count=top_video_count)
    try:
        trending_videos = json.loads(trending_data_str)
    except Exception:
        return f"Failed to parse trending videos list: {trending_data_str}"

    if not isinstance(trending_videos, list) or not trending_videos:
        return "No trending videos found."

    total_shorts_created = []

    # Step 2 & 3: Check heatmap and crop replayed scenes for each video
    for v_idx, video in enumerate(trending_videos, 1):
        v_url = video["url"]
        v_title = video.get("title", f"Video_{v_idx}")
        v_id = video.get("id", f"id_{v_idx}")

        scenes_data_str = get_most_replayed_scenes(v_url, max_scenes=max_scenes_per_video)
        try:
            scenes_info = json.loads(scenes_data_str)
            scenes = scenes_info.get("scenes", [])
        except Exception:
            scenes = []

        if not scenes:
            # Fallback default 15s short if no heatmap
            scenes = [{
                "start_sec": 10,
                "end_sec": 25,
                "time_window": "00:10 - 00:25",
                "heatmap_score": 0.0,
                "chapter": "Fallback Section"
            }]

        for s_idx, scene in enumerate(scenes, 1):
            st = scene["start_sec"]
            et = scene["end_sec"]
            clean_title = re.sub(r"[^\w\-]", "_", v_title)[:30]
            filename = f"short_{v_idx:02d}_{s_idx:02d}_{v_id}_{int(st)}s_{int(et)}s.mp4"
            
            crop_res = crop_replayed_scene_to_mp4(
                video_url=v_url,
                start_time=st,
                end_time=et,
                output_filename=filename
            )

            out_path = OUTPUT_DIR / filename
            file_exists = out_path.exists()
            file_size_mb = (out_path.stat().st_size / (1024 * 1024)) if file_exists else 0.0

            total_shorts_created.append({
                "video_rank": v_idx,
                "video_title": v_title,
                "video_url": v_url,
                "scene_index": s_idx,
                "time_window": scene["time_window"],
                "heatmap_score": scene["heatmap_score"],
                "chapter": scene.get("chapter", "N/A"),
                "mp4_file": str(out_path),
                "filename": filename,
                "size_mb": round(file_size_mb, 2),
                "status": "Success" if file_exists else "Failed"
            })

    # Step 4: Build final report
    report_lines = [
        f"# Most Replayed Video Shorts Report",
        f"Processed **{len(trending_videos)}** trending videos and generated **{len(total_shorts_created)}** MP4 video shorts.\n",
        "## Summary of Cropped Shorts (MP4 Format):",
    ]

    for item in total_shorts_created:
        report_lines.append(
            f"- **[Video #{item['video_rank']}] {item['video_title']}**\n"
            f"  - **Scene #{item['scene_index']} Window**: {item['time_window']} (Score: {item['heatmap_score']})\n"
            f"  - **Chapter**: {item['chapter']}\n"
            f"  - **Output MP4 File**: `{item['filename']}` ({item['size_mb']} MB)\n"
            f"  - **Full Path**: `{item['mp4_file']}`\n"
        )

    return "\n".join(report_lines)


root_agent = Agent(
    name="root_agent",
    model=Gemini(
        model="gemini-3.7-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an AI Video Shorts Creator agent specialized in extracting and cropping the most replayed scenes from YouTube videos. "
        "Your workflow:\n"
        "1. If the user provides a specific YouTube video URL (or link), process that specific video URL using `create_shorts_from_video_url` or `create_most_replayed_video_shorts(video_url=...)`.\n"
        "2. If the user requests to 'create most replayed video shorts' without a specific URL, gather top 10 trending videos and process all of them.\n"
        "3. Check audience retention / player heatmap data to detect all peak replayed scenes for the video(s).\n"
        "4. Crop those exact replayed scenes into separate MP4 format video files under output_shorts/.\n"
        "5. Return a complete, clear list of all cropped MP4 shorts created."
    ),
    tools=[
        create_most_replayed_video_shorts,
        create_shorts_from_video_url,
        gather_top_trending_videos,
        get_most_replayed_scenes,
        crop_replayed_scene_to_mp4,
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
