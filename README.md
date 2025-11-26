# AI Chess Puzzle Lab

A tiny web demo that showcases the DeepMind “double rook sacrifice” puzzle with per-move commentary and simple navigation.

## Live Demo

Hosted on GitHub Pages:  
https://toddllm.github.io/ai-chess-puzzle-lab/

## Running Locally

1. Clone the repo.
2. Start a static server in the repo root, e.g.:
   ```bash
   python -m http.server 8010
   ```
3. Open `http://localhost:8010` in your browser.

## Controls

- Next / Prev buttons or Arrow Right / Arrow Left to step through the solution.
- Commentary updates per move.
- Reset returns to the initial position.

## TTS Pipeline (optional)

- `generate_moves.py` builds per-move prompts, dummy audio (silence), and a manifest.
- Sample input: `sample_moves_double_rook_sac.json`.
- Usage:
  ```bash
  python generate_moves.py --game-id double_rook_sac --moves sample_moves_double_rook_sac.json --out generated/
  ```
- Replace the placeholder TTS call in `generate_moves.py` with your TTS; files land in `generated/<game_id>/` with `manifest.json`.

## Puzzle Data

- Puzzles live in `puzzles.json` (FEN, SAN lines, themes, commentary).
- Current content: the DeepMind double-rook-sac puzzle with full mate line.

## Development Notes

- Frontend is plain HTML/JS with `chess.js` and `chessboard.js` from CDN.
- Playwright (chromium) is included as a dev dep for headless smoke tests.
