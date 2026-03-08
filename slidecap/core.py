from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse

import cv2
import whisper
import yt_dlp
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


@dataclass
class Slide:
    slide_num: int
    timestamp: float
    image_path: str
    transcript: str
    youtube_link: str


@dataclass
class PipelineResult:
    status: str
    url: str
    video_id: str
    output_markdown: str
    images_dir: str
    slide_count: int
    image_files: list[str]
    downloaded_resolution: str
    downloaded_fps: str
    format_note: str
    download_format: str
    similarity_threshold: float
    sample_rate: float
    whisper_model: str
    language: str | None
    started_at: str
    completed_at: str
    duration_seconds: float
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "url": self.url,
            "video_id": self.video_id,
            "output_markdown": self.output_markdown,
            "images_dir": self.images_dir,
            "slide_count": self.slide_count,
            "image_files": self.image_files,
            "downloaded_resolution": self.downloaded_resolution,
            "downloaded_fps": self.downloaded_fps,
            "format_note": self.format_note,
            "download_format": self.download_format,
            "similarity_threshold": self.similarity_threshold,
            "sample_rate": self.sample_rate,
            "whisper_model": self.whisper_model,
            "language": self.language,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "warnings": self.warnings,
        }


def _check_runtime_dependencies() -> None:
    missing = [cmd for cmd in ("yt-dlp", "ffmpeg") if shutil.which(cmd) is None]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required system binaries: {missing_list}")


def _extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)
    host = parsed.netloc.lower()

    if "youtu.be" in host:
        candidate = parsed.path.strip("/").split("/")[0]
        if candidate:
            return candidate

    if "youtube.com" in host:
        params = parse_qs(parsed.query)
        video_ids = params.get("v", [])
        if video_ids:
            return video_ids[0]

        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
            return parts[1]

    return "unknown_video"


def _canonical_watch_url(youtube_url: str, video_id: str) -> str:
    if video_id != "unknown_video":
        return f"https://www.youtube.com/watch?v={video_id}"
    parsed = urlparse(youtube_url)
    path = parsed.path
    if not path:
        return youtube_url
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def generate_youtube_timestamp_url(youtube_url: str, timestamp_seconds: float) -> str:
    video_id = _extract_video_id(youtube_url)
    base = _canonical_watch_url(youtube_url, video_id)
    return f"{base}&t={int(timestamp_seconds)}s" if "?" in base else f"{base}?t={int(timestamp_seconds)}s"


def build_1080p_format_string(allow_lower_quality: bool) -> str:
    strict_1080 = "bestvideo[height=1080]+bestaudio/best[height=1080]"
    if not allow_lower_quality:
        return strict_1080

    fallback = (
        "bestvideo[height<=720]+bestaudio/best[height<=720]/"
        "bestvideo[height<=480]+bestaudio/best[height<=480]/"
        "bestvideo[height<=360]+bestaudio/best[height<=360]/best"
    )
    return f"{strict_1080}/{fallback}"


def _run_command(cmd: Iterable[str]) -> subprocess.CompletedProcess[str]:
    LOGGER.debug("Running command: %s", " ".join(cmd))
    return subprocess.run(list(cmd), capture_output=True, text=True)


def _make_progress_hook() -> callable:
    """Return a yt-dlp progress_hooks callback that drives a tqdm bar."""
    pbar = None

    def hook(d: dict) -> None:
        nonlocal pbar
        status = d.get("status")

        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)

            if pbar is None and total:
                pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc="Downloading",
                    file=sys.stderr,
                )
            if pbar is not None:
                if total and pbar.total != total:
                    pbar.total = total
                    pbar.refresh()
                pbar.n = downloaded
                pbar.refresh()

        elif status == "finished":
            if pbar is not None:
                pbar.close()
                pbar = None

    return hook


def _download_video(youtube_url: str, video_path: Path, allow_lower_quality: bool) -> tuple[Path, str]:
    format_string = build_1080p_format_string(allow_lower_quality)
    ydl_opts = {
        "format": format_string,
        "merge_output_format": "mp4",
        "writeinfojson": True,
        "noplaylist": True,
        "outtmpl": {"default": str(video_path)},
        "progress_hooks": [_make_progress_hook()],
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except yt_dlp.DownloadError as exc:
        raise RuntimeError(f"yt-dlp download failed:\n{exc}") from exc
    LOGGER.info("Video downloaded: %s", video_path)
    return video_path, format_string


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    cmd = ["ffmpeg", "-i", str(video_path), "-q:a", "0", "-map", "a", str(audio_path), "-y"]
    result = _run_command(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr.strip()}")
    LOGGER.info("Audio extracted: %s", audio_path)


def detect_slides(video_path: Path, similarity_threshold: float = 0.85, sample_rate: float = 1.0) -> list[tuple[float, Image.Image]]:
    """
    Reuses the slide detection behavior from the existing app:
    - SSIM frame comparison on sampled grayscale frames.
    - Dynamic threshold adjustment based on video height.
    - Extra blur for 1080p+ input.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Failed to read video FPS; cannot perform slide detection.")

    LOGGER.info("Video info: %sx%s, %.1f FPS, %s frames", width, height, fps, total_frames)

    if height >= 1080:
        adjusted_threshold = similarity_threshold * 1.05
        LOGGER.info("High quality video detected. Adjusted threshold: %.3f", adjusted_threshold)
    elif height >= 720:
        adjusted_threshold = similarity_threshold * 1.02
        LOGGER.info("Medium quality video detected. Adjusted threshold: %.3f", adjusted_threshold)
    else:
        adjusted_threshold = similarity_threshold
        LOGGER.info("Standard quality video detected. Threshold: %.3f", adjusted_threshold)

    slides: list[tuple[float, Image.Image]] = []
    prev_frame = None
    frame_count = 0

    sample_interval = int(fps * sample_rate)
    sample_interval = max(sample_interval, 1)

    if height >= 1080:
        sample_interval = max(sample_interval, int(fps * 0.5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % sample_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if height >= 1080:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if prev_frame is not None:
            similarity = ssim(prev_frame, gray)
            if similarity < adjusted_threshold:
                timestamp = frame_count / fps
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                slides.append((timestamp, pil_img))
                LOGGER.info(
                    "Slide detected at %.1fs (similarity %.3f < threshold %.3f)",
                    timestamp,
                    similarity,
                    adjusted_threshold,
                )

        prev_frame = gray

    cap.release()

    if not slides:
        fallback_cap = cv2.VideoCapture(str(video_path))
        fallback_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"mp4v"))
        fallback_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = fallback_cap.read()
        if ret:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            slides.insert(0, (0.0, pil_img))
        fallback_cap.release()

    return slides


def _transcribe_audio(audio_path: Path, whisper_model: str, language: str | None) -> tuple[str, list[dict]]:
    LOGGER.info("Loading Whisper model: %s", whisper_model)
    model = whisper.load_model(whisper_model)
    kwargs = {"language": language} if language else {}
    LOGGER.info("Transcribing audio...")
    result = model.transcribe(str(audio_path), **kwargs)
    return result.get("text", "").strip(), result.get("segments", [])


def _align_transcript_to_slides(segments: list[dict], slide_timestamps: list[float]) -> list[str]:
    """Assign Whisper segments to slides based on actual segment start times."""
    if not slide_timestamps:
        return []

    chunks: list[str] = []
    for i, timestamp in enumerate(slide_timestamps):
        next_timestamp = slide_timestamps[i + 1] if i + 1 < len(slide_timestamps) else float("inf")

        if i == 0:
            # First slide gets all segments that start before the second slide
            slide_segs = [seg for seg in segments if seg["start"] < next_timestamp]
        else:
            slide_segs = [seg for seg in segments if timestamp <= seg["start"] < next_timestamp]

        chunks.append(" ".join(seg["text"].strip() for seg in slide_segs))

    return chunks


def _write_slide_images(
    slides: list[tuple[float, Image.Image]],
    images_dir: Path,
    video_id: str,
    image_format: str,
    image_quality: int,
) -> list[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    pil_format = "JPEG" if image_format == "jpg" else "PNG"
    extension = "jpg" if image_format == "jpg" else "png"

    for idx, (timestamp, image) in enumerate(slides, start=1):
        file_name = f"yt_{video_id}_slide_{idx:03d}_t{int(timestamp):06d}.{extension}"
        out_path = images_dir / file_name
        if image_format == "jpg":
            if image.mode in ("RGBA", "LA", "P"):
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
                image = rgb_image
            image.save(out_path, format=pil_format, quality=image_quality, optimize=True)
        else:
            image.save(out_path, format=pil_format, optimize=True)
        output_paths.append(out_path)

    return output_paths


def _anchor_for_slide(slide_num: int, timestamp: float) -> str:
    ts = f"{timestamp:.1f}".replace(".", "")
    return f"slide-{slide_num}-{ts}s"


def _build_markdown(
    youtube_url: str,
    video_id: str,
    slide_records: list[Slide],
    format_string: str,
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# YouTube Slide Transcript",
        "",
        f"- Source: {youtube_url}",
        f"- Video ID: {video_id}",
        f"- Generated: {generated}",
        f"- Slide Count: {len(slide_records)}",
        f"- Download Format: `{format_string}`",
        "",
        "## Table of Contents",
    ]

    for slide in slide_records:
        anchor = _anchor_for_slide(slide.slide_num, slide.timestamp)
        lines.append(f"- [Slide {slide.slide_num} ({slide.timestamp:.1f}s)](#{anchor})")

    lines.extend(["", "---", ""])

    for slide in slide_records:
        anchor = _anchor_for_slide(slide.slide_num, slide.timestamp)
        lines.extend(
            [
                f"## Slide {slide.slide_num} ({slide.timestamp:.1f}s)",
                f'<a id="{anchor}"></a>',
                "",
                f"YouTube Link: {slide.youtube_link}",
                "",
                f"![Slide {slide.slide_num}]({slide.image_path})",
                "",
                "Transcript:",
                slide.transcript or "_No transcript text for this segment._",
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def _read_download_info(video_path: Path) -> dict:
    info_path = video_path.with_suffix(".info.json")
    if not info_path.exists():
        return {}
    try:
        with info_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _fetch_video_title(url: str) -> str | None:
    """Fetch video title via yt-dlp without downloading the video."""
    result = _run_command(["yt-dlp", "--print", "title", "--no-playlist", url])
    if result.returncode == 0:
        title = result.stdout.strip()
        if title:
            return title
    return None


def _sanitize_filename(name: str, max_length: int = 100) -> str:
    """Remove filesystem-invalid characters and truncate."""
    sanitized = re.sub(r'[\\/:*?"<>|]', "", name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip().strip(".")
    return (sanitized[:max_length].rstrip() if len(sanitized) > max_length else sanitized) or "untitled"


def resolve_output_paths(url: str, out_md: str | None, images_dir: str | None) -> tuple[Path, Path]:
    """Return resolved (out_md, images_dir) paths, filling in defaults when not provided."""
    base_dir = Path.cwd() / "slidecap"

    if out_md is None:
        title = _fetch_video_title(url)
        filename = (_sanitize_filename(title) if title else _extract_video_id(url)) + ".md"
        resolved_out_md = base_dir / filename
    else:
        resolved_out_md = Path(out_md).expanduser().resolve()

    if images_dir is None:
        resolved_images_dir = base_dir / "slides"
    else:
        resolved_images_dir = Path(images_dir).expanduser().resolve()

    return resolved_out_md, resolved_images_dir


def run_pipeline(args: argparse.Namespace) -> PipelineResult:
    _check_runtime_dependencies()
    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    start_time = time.time()

    out_md, images_dir = resolve_output_paths(args.url, args.out_md, args.images_dir)

    if out_md.exists() and not args.overwrite:
        raise FileExistsError(f"Output markdown already exists: {out_md}. Use --overwrite.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="slidecap_"))
        LOGGER.info("Keeping temp directory: %s", temp_dir)
        should_cleanup = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="slidecap_"))
        should_cleanup = True

    try:
        video_path = temp_dir / "video.mp4"
        audio_path = temp_dir / "audio.mp3"

        downloaded_video, format_string = _download_video(args.url, video_path, args.allow_lower_quality)
        info = _read_download_info(downloaded_video)
        if info:
            LOGGER.info(
                "Downloaded resolution: %sx%s, fps: %s, format: %s",
                info.get("width", "unknown"),
                info.get("height", "unknown"),
                info.get("fps", "unknown"),
                info.get("format_note", "unknown"),
            )

        _extract_audio(video_path, audio_path)
        slides = detect_slides(video_path, similarity_threshold=args.similarity_threshold, sample_rate=args.sample_rate)
        if not slides:
            raise RuntimeError("No slides detected.")

        _transcript, segments = _transcribe_audio(audio_path, args.whisper_model, args.language)
        slide_timestamps = [s[0] for s in slides]
        transcript_chunks = _align_transcript_to_slides(segments, slide_timestamps)
        video_id = _extract_video_id(args.url)
        image_paths = _write_slide_images(
            slides,
            images_dir,
            video_id=video_id,
            image_format=args.image_format,
            image_quality=args.image_quality,
        )

        slide_records: list[Slide] = []
        for idx, ((timestamp, _image), transcript_chunk, image_path) in enumerate(
            zip(slides, transcript_chunks, image_paths), start=1
        ):
            relative_img_path = os.path.relpath(image_path, start=out_md.parent).replace(os.sep, "/")
            slide_records.append(
                Slide(
                    slide_num=idx,
                    timestamp=timestamp,
                    image_path=relative_img_path,
                    transcript=transcript_chunk,
                    youtube_link=generate_youtube_timestamp_url(args.url, timestamp),
                )
            )

        markdown = _build_markdown(
            youtube_url=args.url,
            video_id=video_id,
            slide_records=slide_records,
            format_string=format_string,
        )
        out_md.write_text(markdown, encoding="utf-8")
        LOGGER.info("Markdown written: %s", out_md)
        LOGGER.info("Slides written to: %s", images_dir)

        width = info.get("width")
        height = info.get("height")
        downloaded_resolution = (
            f"{width}x{height}" if isinstance(width, int) and isinstance(height, int) else "unknown"
        )
        downloaded_fps = str(info.get("fps", "unknown"))
        format_note = str(info.get("format_note", "unknown"))

        completed_at = datetime.now(timezone.utc).isoformat()
        duration_seconds = round(time.time() - start_time, 3)

        return PipelineResult(
            status="ok",
            url=args.url,
            video_id=video_id,
            output_markdown=str(out_md),
            images_dir=str(images_dir),
            slide_count=len(slide_records),
            image_files=[str(path) for path in image_paths],
            downloaded_resolution=downloaded_resolution,
            downloaded_fps=downloaded_fps,
            format_note=format_note,
            download_format=format_string,
            similarity_threshold=args.similarity_threshold,
            sample_rate=args.sample_rate,
            whisper_model=args.whisper_model,
            language=args.language,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            warnings=[],
        )
    finally:
        if should_cleanup:
            shutil.rmtree(temp_dir, ignore_errors=True)
