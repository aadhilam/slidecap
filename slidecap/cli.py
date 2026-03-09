from __future__ import annotations

import argparse
import json
import logging
import sys

try:
    from .core import run_pipeline
except ImportError:  # pragma: no cover - enables direct script invocation
    from core import run_pipeline  # type: ignore[no-redef]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slidecap",
        description="Generate slide-aligned markdown notes from a YouTube URL.",
    )
    parser.add_argument("--url", required=True, help="YouTube URL to process.")
    parser.add_argument(
        "--out-md",
        default=None,
        help="Output markdown file path. Defaults to ./slidecap/<video title>.md",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory where slide images are written. Defaults to ./slidecap/slides/",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Slide detection sensitivity threshold (default: 0.85).",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Frame sampling rate in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help=(
            "Whisper model name. Default: medium. "
            "Multilingual: tiny, base, small, medium, large, large-v2, large-v3, turbo. "
            "English-only (faster): tiny.en, base.en, small.en, medium.en."
        ),
    )
    parser.add_argument("--language", default=None, help="Optional transcription language code (e.g. en).")
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Image format for saved slides (default: jpg).",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=90,
        help="JPEG quality 1-100 (used only for jpg output). Default: 90.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output markdown if it exists.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary downloaded files for debugging.")
    parser.add_argument(
        "--allow-lower-quality",
        action="store_true",
        help="Fallback to sub-1080p quality if exact 1080p is not available.",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warn", "error"],
        help="Logging verbosity (default: info).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON output for agent workflows.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not (0.5 <= args.similarity_threshold <= 0.95):
        raise ValueError("--similarity-threshold must be between 0.5 and 0.95.")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be > 0.")
    if not (1 <= args.image_quality <= 100):
        raise ValueError("--image-quality must be between 1 and 100.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(level=level_map[args.log_level], format="%(levelname)s: %(message)s")

    try:
        _validate_args(args)
        result = run_pipeline(args)
    except Exception as exc:  # noqa: BLE001
        if args.json:
            error_payload = {
                "status": "error",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            print(json.dumps(error_payload, ensure_ascii=True))
        else:
            logging.error(str(exc))
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=True))
    else:
        print(result.output_markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
