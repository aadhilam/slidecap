# slidecap

CLI tool that turns a YouTube video into slide-aligned markdown notes for agent/LLM workflows.

## What It Does

1. Downloads a YouTube video at 1080p (strict by default).
2. Detects slide changes using SSIM-based frame comparison (same behavior as your existing app).
3. Transcribes audio with Whisper.
4. Aligns transcript chunks to detected slide timestamps.
5. Writes:
   - A markdown file with slide sections + transcript text.
   - Slide images to one shared images folder.

## Install

```bash
pip install slidecap
```

System dependencies:
- `ffmpeg` in PATH
- `yt-dlp` in PATH

## Usage

Minimal — just pass a URL:

```bash
slidecap --url "https://www.youtube.com/watch?v=abc123"
```

This creates:
```
./slidecap/
  My Video Title.md
  slides/
    yt_abc123_slide_001_t000005.jpg
    ...
```

Custom output paths:

```bash
slidecap \
  --url "https://www.youtube.com/watch?v=abc123" \
  --out-md "/path/to/vault/notes/abc123.md" \
  --images-dir "/path/to/vault/assets/youtube-slides"
```

JSON output (for agents):

```bash
slidecap --url "https://www.youtube.com/watch?v=abc123" --json
```

## Flags

- `--url` (required): YouTube URL.
- `--out-md` (optional): Output markdown path. Defaults to `./slidecap/<video title>.md`.
- `--images-dir` (optional): Slide images folder. Defaults to `./slidecap/slides/`.
- `--similarity-threshold` (default `0.85`): Slide detection threshold.
- `--sample-rate` (default `1.0`): Frame sampling interval in seconds.
- `--whisper-model` (default `base`): Whisper model name. See [Whisper Models](#whisper-models) below.
- `--language`: Optional transcription language code (e.g. `en`, `de`, `ja`).
- `--image-format` (default `jpg`): `jpg` or `png`.
- `--image-quality` (default `90`): JPEG quality.
- `--allow-lower-quality`: Fallback below 1080p if exact 1080p is unavailable.
- `--overwrite`: Overwrite existing markdown output.
- `--keep-temp`: Keep temp downloads for debugging.
- `--log-level` (default `info`): `debug|info|warn|error`.
- `--json`: Print structured JSON result (success or error) for agent workflows.

## Whisper Models

| Model | Speed | Accuracy | Notes |
|---|---|---|---|
| `tiny` | Fastest | Lowest | Quick drafts |
| `base` | Fast | Good | **Default** |
| `small` | Moderate | Better | Good general choice |
| `medium` | Slow | Strong | |
| `large` | Slowest | Best | |
| `large-v2` | Slowest | Best | Improved large |
| `large-v3` | Slowest | Best | Latest multilingual |
| `turbo` | Fast | Very good | Efficient alternative to large |

English-only variants (`tiny.en`, `base.en`, `small.en`, `medium.en`) are faster than their multilingual counterparts when transcribing English content.

## Output Format

The markdown file includes:
- source metadata
- table of contents
- per-slide sections with:
  - timestamped YouTube link
  - relative markdown image link
  - transcript chunk aligned to that slide

Images are named to avoid collisions:
- `yt_<video-id>_slide_<nnn>_t<seconds>.<ext>`

## JSON Response Shape

Success:

```json
{
  "status": "ok",
  "url": "https://www.youtube.com/watch?v=abc123",
  "video_id": "abc123",
  "output_markdown": "/path/to/vault/notes/abc123.md",
  "images_dir": "/path/to/vault/assets/youtube-slides",
  "slide_count": 12,
  "image_files": [],
  "downloaded_resolution": "1920x1080",
  "downloaded_fps": "30",
  "format_note": "1080p",
  "download_format": "bestvideo[height=1080]+bestaudio/best[height=1080]",
  "similarity_threshold": 0.85,
  "sample_rate": 1.0,
  "whisper_model": "base",
  "language": "en",
  "started_at": "2026-02-23T00:00:00+00:00",
  "completed_at": "2026-02-23T00:03:00+00:00",
  "duration_seconds": 180.123,
  "warnings": []
}
```

Error:

```json
{
  "status": "error",
  "error": "message",
  "error_type": "RuntimeError"
}
```
