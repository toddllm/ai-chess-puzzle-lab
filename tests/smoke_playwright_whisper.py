#!/usr/bin/env python3
"""
Headless smoke test:
- Starts the local Flask server (server.py) with a lightweight TTS engine.
- Drives the UI with Playwright (Chromium, headless).
- Downloads two generated commentary clips and transcribes them with Whisper.
- Asserts that the transcripts contain key phrases and that puzzle navigation works.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import requests
import whisper
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "playwright_runs"
RUN_DIR.mkdir(exist_ok=True)


def normalize_text(text: str) -> str:
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace())


def wait_for_health(base_url: str, timeout: float = 40.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{base_url}/health", timeout=2)
            if res.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Server did not become healthy in time.")


@contextmanager
def run_server(port: int, engine: str):
    cmd = [sys.executable, "server.py", "--port", str(port), "--tts-engine", engine]
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        wait_for_health(f"http://127.0.0.1:{port}")
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


_WHISPER_MODEL = None


def run_whisper(path: Path) -> str:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    res = _WHISPER_MODEL.transcribe(str(path), fp16=False)
    return normalize_text(res.get("text", ""))


def capture_audio_clip(page, action, base_url: str, label: str) -> Path:
    with page.expect_response("**/api/tts") as resp_info:
        action()
    resp = resp_info.value
    data = resp.json()
    audio_url = base_url + data["url"]
    clip_path = RUN_DIR / f"{label}.{audio_url.rsplit('.', 1)[-1]}"
    clip_bytes = requests.get(audio_url, timeout=30).content
    clip_path.write_bytes(clip_bytes)
    return clip_path


def main():
    parser = argparse.ArgumentParser(description="Smoke test: Playwright + Whisper.")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--engine", default="dia", help="TTS engine to use for the server.")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    results = []

    with run_server(args.port, args.engine):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(base_url, wait_until="networkidle")

            # Puzzle 1, move 1
            page.click("#next-move")
            clip1 = capture_audio_clip(page, lambda: page.click("#play-commentary"), base_url, "puzzle1_move1")
            transcript1 = run_whisper(clip1)
            rook_keywords = ["double rook sac", "double rooksac", "double rooksack", "double rook "]
            ok1 = any(kw in transcript1 for kw in rook_keywords)
            results.append(
                {
                    "clip": str(clip1),
                    "transcript": transcript1,
                    "expect": "double rook sac",
                    "ok": ok1,
                }
            )
            if not ok1:
                raise AssertionError("Transcript for puzzle 1 did not include the double rook sac line.")

            # Go to next puzzle, move 1
            page.click("#next-puzzle")
            page.click("#next-move")
            clip2 = capture_audio_clip(page, lambda: page.click("#play-commentary"), base_url, "puzzle2_move1")
            transcript2 = run_whisper(clip2)
            ok2 = "deflect" in transcript2
            results.append(
                {
                    "clip": str(clip2),
                    "transcript": transcript2,
                    "expect": "deflects",
                    "ok": ok2,
                }
            )
            if not ok2:
                raise AssertionError("Transcript for puzzle 2 did not mention 'deflect'.")

            browser.close()

    summary_path = RUN_DIR / "smoke_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"Smoke test passed. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
