#!/usr/bin/env python3
"""
generate_moves.py

Utility to turn a JSON move list into per-move, TTS-ready prompts, dummy audio
files, and a manifest. Plug in your actual TTS where noted.
"""
import argparse
import json
import math
import re
import wave
from pathlib import Path
from typing import Dict, List, Any


DEFAULT_SETTINGS = {
    "temperature": 1.2,
    "top_p": 0.90,
    "cfg_scale": 3.0,
    "cfg_filter_top_k": 50,
    "max_tokens": 1200,
    "audio_format": "wav",
    "sample_rate": 44100,
    "channels": 1,
    "normalization_peak": 0.9,
}

FILLER = "That continues the line."


def ensure_punctuation(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[-1] not in ".?!":
        text += "."
    return text


def needs_filler(text: str, min_words: int = 12) -> bool:
    return len(text.split()) < min_words


def build_prompt(commentary: str, speaker: str = "S1") -> str:
    core = ensure_punctuation(commentary)
    if needs_filler(core):
        core = f"{core} {FILLER}"
    return f"[{speaker}] {core} [{speaker}]"


def write_silence_wav(path: Path, duration_seconds: float, sample_rate: int = 44100, channels: int = 1):
    nframes = int(duration_seconds * sample_rate)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * nframes)
    return duration_seconds


def process_moves(game_id: str, moves: List[Dict[str, Any]], out_dir: Path, settings: Dict[str, Any]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_moves = []
    for move in sorted(moves, key=lambda m: m["ply"]):
        speaker = move.get("speaker", "S1")
        raw_text = move.get("commentary", "").strip()
        prompt = build_prompt(raw_text, speaker)
        ply = move["ply"]
        audio_filename = f"{game_id}_move{ply}.{settings['audio_format']}"
        audio_path = out_dir / audio_filename
        # Placeholder audio: 1.2s silence
        duration = write_silence_wav(audio_path, duration_seconds=1.2, sample_rate=settings["sample_rate"], channels=settings["channels"]) \
            if settings["audio_format"] == "wav" else 0.0
        manifest_moves.append({
            "game_id": game_id,
            "ply": ply,
            "move_san": move.get("move_san", ""),
            "speaker": speaker,
            "text_used": prompt,
            "audio_file": audio_filename,
            "duration_seconds": duration,
            "peak_level": 0.0,  # silence; replace with real peak after TTS + normalize
            "gen_time_seconds": 0.0,
            "settings_used": {
                "temperature": settings["temperature"],
                "top_p": settings["top_p"],
                "cfg_scale": settings["cfg_scale"],
                "cfg_filter_top_k": settings["cfg_filter_top_k"],
                "max_tokens": settings["max_tokens"],
                "audio_format": settings["audio_format"],
                "sample_rate": settings["sample_rate"],
                "channels": settings["channels"],
                "normalization_peak": settings["normalization_peak"],
            },
        })
    return {
        "moves": manifest_moves,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate per-move TTS prompts and placeholder audio.")
    parser.add_argument("--game-id", required=True, help="Game identifier.")
    parser.add_argument("--moves", required=True, help="Path to moves.json (see sample format).")
    parser.add_argument("--out", default="generated", help="Output base directory.")
    parser.add_argument("--audio-format", choices=["wav", "mp3"], default="wav", help="Output audio format (wav recommended).")
    parser.add_argument("--temp", type=float, help="Override temperature.")
    parser.add_argument("--top-p", type=float, help="Override top_p.")
    parser.add_argument("--cfg-scale", type=float, help="Override cfg_scale.")
    parser.add_argument("--cfg-filter-top-k", type=int, help="Override cfg_filter_top_k.")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens.")
    args = parser.parse_args()

    settings = DEFAULT_SETTINGS.copy()
    settings["audio_format"] = args.audio_format
    if args.temp is not None:
        settings["temperature"] = args.temp
    if args.top_p is not None:
        settings["top_p"] = args.top_p
    if args.cfg_scale is not None:
        settings["cfg_scale"] = args.cfg_scale
    if args.cfg_filter_top_k is not None:
        settings["cfg_filter_top_k"] = args.cfg_filter_top_k
    if args.max_tokens is not None:
        settings["max_tokens"] = args.max_tokens

    with open(args.moves, "r", encoding="utf-8") as f:
        data = json.load(f)

    game_id = args.game_id
    game_info = {
        "game_id": game_id,
        "event": data.get("event"),
        "white": data.get("white"),
        "black": data.get("black"),
        "site": data.get("site"),
        "date": data.get("date"),
    }
    moves = data["moves"]

    game_dir = Path(args.out) / game_id
    game_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "game": game_info,
        "settings": settings,
    }
    manifest.update(process_moves(game_id, moves, game_dir, settings))

    manifest_path = game_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote manifest to {manifest_path}")
    print(f"Audio files in {game_dir}")
    print("NOTE: Replace the placeholder TTS call in process_moves with your real TTS to generate voiced clips.")


if __name__ == "__main__":
    main()
