#!/usr/bin/env python3
"""
Generate Dia-spoken commentary clips for every puzzle entry.

Outputs:
- audio/<slug>-mXX.wav files (Dia voiced)
- audio_manifest.json with metadata per puzzle/move

This uses the Dia model directly, no other TTS backends.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from dia.model import DEFAULT_SAMPLE_RATE, Dia

ROOT = Path(__file__).resolve().parent
PUZZLES_PATH = ROOT / "puzzles.json"
AUDIO_DIR = ROOT / "audio"
MANIFEST_PATH = ROOT / "audio_manifest.json"


def safe_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "puzzle"


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text[-1] not in ".?!":
        text += "."
    return text


def build_prompt(text: str) -> str:
    core = ensure_sentence(text)
    return f"[S1] {core} [S1]"


def load_model(model_name: str, device: torch.device) -> Dia:
    dtype = "float16" if device.type == "cuda" else "float32"
    print(f"Loading Dia '{model_name}' on {device} ({dtype})")
    return Dia.from_pretrained(model_name, compute_dtype=dtype, device=device)


def render_commentary(model: Dia, text: str, max_tokens: int = 240) -> np.ndarray:
    audio = model.generate(
        build_prompt(text),
        max_tokens=max_tokens,
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
    return audio


def main():
    puzzles: List[Dict] = json.loads(PUZZLES_PATH.read_text())
    AUDIO_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("nari-labs/Dia-1.6B-0626", device)

    manifest = []
    total = 0
    for puzzle in puzzles:
        slug = safe_slug(puzzle["id"])
        clips = []
        for idx, comment in enumerate(puzzle.get("commentary", []), start=1):
            text = comment.strip()
            if not text:
                continue
            fname = f"{slug}-m{idx:02d}.wav"
            fpath = AUDIO_DIR / fname
            if fpath.exists():
                print(f"[{puzzle['id']}] move {idx}: exists, skipping")
            else:
                print(f"[{puzzle['id']}] move {idx}: generating...")
                audio = render_commentary(model, text)
                sf.write(fpath, audio, DEFAULT_SAMPLE_RATE)
            clips.append(
                {
                    "move_index": idx,
                    "file": f"audio/{fname}",
                    "text": text,
                }
            )
            total += 1
        manifest.append(
            {
                "id": puzzle["id"],
                "slug": slug,
                "clips": clips,
            }
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Done. Wrote {total} clips to {AUDIO_DIR} and manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
