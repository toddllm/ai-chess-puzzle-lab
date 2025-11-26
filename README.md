# AI Chess Puzzle Lab

A tiny web demo that showcases the DeepMind “double rook sacrifice” puzzle (and several related mates) with per-move commentary, optional narrated audio, and simple navigation.

## Live Demo

Hosted on GitHub Pages (static only):  
https://toddllm.github.io/ai-chess-puzzle-lab/

Audio buttons are disabled on the static site because there is no backend there—run locally with the Flask server to enable speech.

## Running Locally (silent/static)

1. Clone the repo.
2. Start a static server in the repo root, e.g.:
   ```bash
   python -m http.server 8010
   ```
3. Open `http://localhost:8010` in your browser.

## Running Locally with TTS audio

We ship a lightweight Flask backend (`server.py`) that serves the static files and generates commentary audio on demand. Pick your engine:

- `--tts-engine dia` to use the Dia model (install from the sibling repo: `pip install -e ../dia`; requires torch)
- `--tts-engine gtts` (default if Dia is not installed) for a fast, lightweight network TTS
- `--tts-engine dummy` if you just want a tone placeholder

Quick start:

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium  # first run only, for smoke tests

# run the server with the lightweight TTS engine
python server.py --port 8010 --tts-engine gtts
# open http://localhost:8010
```

Audio is generated live by Dia and cached under `generated_audio/` to avoid repeated synthesis. There is no static fallback; run the server to hear commentary.

## Controls

- Next / Prev buttons or Arrow Right / Arrow Left to step through the solution.
- Commentary updates per move.
- Play / Regenerate commentary audio (when the backend is available).
- Reset returns to the initial position.

## Puzzle Data

- Puzzles live in `puzzles.json` (FEN, SAN lines, themes, commentary).
- Current content: the DeepMind double-rook-sac puzzle plus six mate-in-2s from the DeepMind paper appendix.

## Smoke Test (Playwright + Whisper)

The `tests/smoke_playwright_whisper.py` script:
- starts the Flask server (headless)
- steps through two puzzles with Playwright (chromium, headless)
- downloads the generated audio clips and transcribes them with Whisper
- asserts key phrases are present and writes a summary to `playwright_runs/smoke_summary.json`

Run it locally:

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
python tests/smoke_playwright_whisper.py --engine gtts --port 8010
```

Switch `--engine dia` to smoke test the Dia backend instead.

## Development Notes

- Frontend is plain HTML/JS with `chess.js` and `chessboard.js` from CDN.
- `server.py` is a tiny Flask wrapper that adds `/api/tts` to the static bundle.
- Playwright (chromium) + Whisper are used for the optional headless smoke test.
