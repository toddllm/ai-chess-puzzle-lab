#!/usr/bin/env python3
"""
Minimal Flask server for the AI Chess Puzzle Lab with a pluggable TTS backend.

Default behavior:
- Serves the static frontend (index.html, puzzles.json).
- Exposes /api/tts to synthesize commentary audio for the current move.
- Stores generated audio under generated_audio/ and reuses cached clips.

Engines:
- gtts (default if Dia is not installed)
- dia (requires the Dia package + torch)
- dummy (short tone; useful when testing without network/audio)
"""
from __future__ import annotations

import argparse
import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
from flask import Flask, jsonify, request, send_from_directory


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE_DIR = ROOT / "generated_audio"


def safe_slug(value: str) -> str:
    """Normalize IDs for filenames."""
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "puzzle"


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


class BaseEngine:
    name = "base"
    ext = "wav"

    def synthesize(self, text: str, dest: Path) -> Tuple[Path, Optional[float]]:
        raise NotImplementedError


class DummyEngine(BaseEngine):
    """Generates a short tone; useful for offline smoke."""

    name = "dummy"
    ext = "wav"

    def synthesize(self, text: str, dest: Path) -> Tuple[Path, Optional[float]]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        sr = 16_000
        duration = 1.2 + min(len(text) / 60, 1.6)
        t = np.linspace(0, duration, int(sr * duration), False)
        tone = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(dest, tone, sr)
        return dest, duration


class GttsEngine(BaseEngine):
    name = "gtts"
    ext = "mp3"

    def __init__(self, lang: str = "en"):
        try:
            from gtts import gTTS  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("gtts is not installed. Run `pip install gTTS`.") from exc
        self._gtts_cls = gTTS
        self.lang = lang

    def synthesize(self, text: str, dest: Path) -> Tuple[Path, Optional[float]]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tts = self._gtts_cls(text=text, lang=self.lang)
        tts.save(dest)
        return dest, None


class DiaEngine(BaseEngine):
    name = "dia"
    ext = "wav"

    def __init__(self, model_name: str, max_tokens: int):
        try:
            import torch  # type: ignore
            from dia.model import DEFAULT_SAMPLE_RATE, Dia  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "Dia is not installed. Install from the dia repo (`pip install -e ../dia`) or pip."
            ) from exc

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = "float16" if device.type == "cuda" else "float32"
        print(f"[dia] Loading model {model_name} on {device} ({dtype})")
        self.model = Dia.from_pretrained(model_name, compute_dtype=dtype, device=device)
        self.sample_rate = DEFAULT_SAMPLE_RATE
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

    def synthesize(self, text: str, dest: Path) -> Tuple[Path, Optional[float]]:
        import numpy as np

        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            audio = self.model.generate(
                text,
                max_tokens=self.max_tokens,
                cfg_scale=3.0,
                temperature=1.6,
                top_p=0.90,
                use_torch_compile=False,
                cfg_filter_top_k=50,
                verbose=False,
            )
        if isinstance(audio, list):
            audio = audio[0]
        audio = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio * (0.9 / peak)
        sf.write(dest, audio, self.sample_rate)
        seconds = float(len(audio) / self.sample_rate) if self.sample_rate else None
        return dest, seconds


def build_engine(engine_name: str, model_name: str, max_tokens: int) -> BaseEngine:
    if engine_name == "dia":
        return DiaEngine(model_name=model_name, max_tokens=max_tokens)
    if engine_name == "dummy":
        return DummyEngine()
    return GttsEngine()


def build_cache_filename(puzzle_id: str, move_index: int, text: str, ext: str) -> str:
    slug = safe_slug(puzzle_id)
    h = hash_text(text)
    return f"{slug}-m{move_index:02d}-{h}.{ext}"


def create_app(engine_name: str, model_name: str, max_tokens: int, cache_dir: Path) -> Flask:
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine(engine_name, model_name, max_tokens)
    print(f"[server] Using TTS engine: {engine.name} (cache={cache_dir})")

    app = Flask(__name__, static_folder=str(ROOT))
    lock = threading.Lock()

    @app.route("/", methods=["GET"])
    def index():
        return send_from_directory(ROOT, "index.html")

    @app.route("/puzzles.json", methods=["GET"])
    def puzzles():
        return send_from_directory(ROOT, "puzzles.json")

    @app.route("/audio_manifest.json", methods=["GET"])
    def audio_manifest():
        manifest_path = ROOT / "audio_manifest.json"
        if manifest_path.exists():
            return send_from_directory(ROOT, "audio_manifest.json")
        return jsonify({"error": "audio manifest not found"}), 404

    @app.route("/audio/<path:filename>", methods=["GET"])
    def audio(filename: str):
        path = cache_dir / filename
        if not path.exists():
            return jsonify({"error": "audio not found"}), 404
        return send_from_directory(cache_dir, filename)

    @app.route("/api/tts", methods=["POST"])
    def tts():
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Add light Dia-friendly markup; harmless for other engines.
        if not text.startswith("[S1]") and not text.startswith("[S2]"):
            text = f"[S1] {text} [S1]"

        puzzle_id = safe_slug(str(data.get("puzzle_id") or "puzzle"))
        move_index = int(data.get("move_index") or 0)
        force = bool(data.get("force"))

        fname = build_cache_filename(puzzle_id, move_index, text, engine.ext)
        fpath = cache_dir / fname

        if fpath.exists() and not force:
            return jsonify(
                {
                    "url": f"/audio/{fname}",
                    "cached": True,
                    "engine": engine.name,
                    "audio_seconds": None,
                }
            )

        start = time.time()
        try:
            with lock:
                out_path, seconds = engine.synthesize(text, fpath)
        except Exception as exc:
            print(f"[server] TTS failure: {exc}")
            return jsonify({"error": str(exc)}), 500

        elapsed = time.time() - start
        print(
            f"[server] gen ok | engine={engine.name} move={move_index} puzzle={puzzle_id} "
            f"time={elapsed:.2f}s saved={out_path.name}"
        )
        return jsonify(
            {
                "url": f"/audio/{out_path.name}",
                "cached": False,
                "engine": engine.name,
                "audio_seconds": seconds,
            }
        )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "engine": engine.name})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the AI Chess Puzzle Lab with TTS commentary.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--tts-engine", choices=["dia", "gtts", "dummy"], default="dia")
    parser.add_argument("--model", default="nari-labs/Dia-1.6B-0626", help="Dia model name (if using dia engine).")
    parser.add_argument("--max-tokens", type=int, default=900, help="Max tokens for Dia generation.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    app = create_app(args.tts_engine, args.model, args.max_tokens, args.cache_dir)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
