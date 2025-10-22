# Anime Frame Optimizer (AFOptimizer)

## Overview
AFOptimizer removes redundant frames from anime and stylised video sources. The project ships with two entry points:
- A Flask web dashboard (`app.py`) that lets you upload footage, watch live progress, and download processed clips.
- A flexible command-line runner (`cli.py`) with per-method fine-tuning flags for automation and batch work.

Under the hood, AFOptimizer bundles four complementary pruning strategies—Optical Flow, Frame Difference, SSIM, and an advanced Unsupervised Deduplication pipeline—so you can mix accuracy, speed, and perceptual quality to fit each project.

## Key Features
- Multi-method optimisation including dense motion analysis, adaptive pixel differencing, perceptual similarity, and a three-stage unsupervised deduper.
- Browser workflow powered by Flask: background job queue, live progress updates, downloadable artefacts, and automatic upload cleanup.
- Command-line execution with reusable defaults, opt-in presets, and granular override flags for every method parameter.
- Automatic H.264 fast-start transcode plus optional `--encoding-*` overrides to control ffmpeg CRF and preset from the CLI.
- Built-in safeguards such as safety windows between keyframes and adaptive thresholding to limit over-pruning.

## Setup
1. Create and activate a Python environment (3.9+ recommended).
2. Install dependencies: `pip install -r requirements.txt`.
3. Install `ffmpeg` and ensure it is discoverable on your `PATH` (needed for final H.264 output).

## Running the Flask App
1. Launch the server: `python3 app.py` (defaults to `http://0.0.0.0:5000`).
2. Upload a supported video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`).
3. Pick a method, tweak thresholds (including unsupervised presets: gentle/balanced/aggressive), and submit.
4. Monitor live progress, FPS estimates, and ETAs on the dashboard; completed jobs expose direct download links.

The web backend saves uploads to `uploads/`, moves processed artefacts into `outputs/`, and removes temporary files automatically once a job finishes.

## Command-Line Interface
Run `python3 cli.py --help` to see the full reference. The general pattern is:

```
python3 cli.py [GLOBAL OPTIONS] METHOD [METHOD OPTIONS]
```

### Global options
- `-o, --output PATH` – place the processed clip at `PATH` (defaults to `input_stem` + method suffix).
- `--encoding-crf VALUE` – re-encode the final file with ffmpeg CRF `VALUE` (lower means higher quality, default 18 when used).
- `--encoding-preset NAME` – ffmpeg preset for the re-encode (e.g. `veryslow`, defaults to `medium` when used).

### Methods and flags
For each method the CLI exposes both plain-language guidance and technical knobs:

**Optical Flow (`optical-flow`)**
- General: keeps frames that show meaningful pixel-wise motion between consecutive frames; good for action-heavy sequences.
- Technical: runs Farnebäck dense optical flow, computes mean vector magnitude, and writes frames whose magnitude exceeds the set threshold.
- Flags: `--flow-mag-threshold FLOAT` (default `0.4`). Lower values keep more frames; higher values prune subtle motion.

Example:
```bash
python3 cli.py ~/videos/episode01.mp4 optical-flow --flow-mag-threshold 0.35 -o ~/outputs/episode01_of.mp4
```

**Frame Difference (`frame-difference`)**
- General: compares brightness changes between frames and drops segments with negligible deltas; fast and effective for static scenes.
- Technical: adapts the supplied base threshold using an initial sampling window, then counts high-difference pixels to decide whether to preserve a frame.
- Flags: `--base-threshold FLOAT` (default `10.0`). Increase to demand larger pixel swings before keeping a frame.

Example:
```bash
python3 cli.py ~/videos/episode01.mp4 frame-difference --base-threshold 14
```

**SSIM (`ssim`)**
- General: focuses on perceptual similarity—removes frames that look virtually identical to the previous one.
- Technical: calculates grayscale Structural Similarity (SSIM) for each frame pair and writes frames whose SSIM falls below the cutoff; always appends the final frame.
- Flags: `--ssim-threshold FLOAT` (default `0.9587`). Lower thresholds enforce stricter pruning; higher thresholds retain more visually similar frames.

Example:
```bash
python3 cli.py ~/videos/episode01.mp4 ssim --ssim-threshold 0.97 -o ~/outputs/episode01_ssim.mp4 --encoding-crf 20
```

**Unsupervised Deduplication (`unsupervised-dedup`)**
- General: combines perceptual hashing, feature clustering, and motion analysis to suppress redundant footage without labelled data; ideal for long episodic sources.
- Technical: stages include Walsh–Hadamard hashing with ordinal texture checks, ORB-based feature clustering, and motion gating using downscaled optical flow with safety keyframe spacing.
- Preset: `--profile {gentle|balanced|aggressive}` loads tuned defaults for different aggressiveness levels (default `balanced`).
- Fine-tune flags (all optional; override preset values):
  - `--hash-threshold INT` – Hamming distance for hash matches.
  - `--ordinal-footrule-threshold FLOAT` – maximum footrule distance between ordinal signatures.
  - `--feature-similarity FLOAT` – ORB match ratio required to collapse a cluster.
  - `--flow-static-threshold FLOAT` – mean flow magnitude treated as static.
  - `--flow-low-ratio FLOAT` – fraction of low-motion pixels necessary for static gating.
  - `--pan-orientation-std FLOAT` – orientation spread threshold for detecting pans.
  - `--safety-keep-seconds FLOAT` – minimum seconds before forcing a keyframe to avoid long drops.

Example:
```bash
python3 cli.py ~/videos/season01.mkv unsupervised-dedup --profile aggressive --safety-keep-seconds 2.0 -o ~/outputs/season01_dedup.mp4
```

All methods emit H.264 MP4 files. Use `--encoding-crf`/`--encoding-preset` when you need tighter control over bitrate/quality trade-offs.

## Method Notes
- Optical Flow and Unsupervised Dedup produce the most accurate cuts for complex motion but run slower.
- Frame Difference is lightweight and pairs well with footage that has obvious mid-frame changes.
- SSIM offers a perceptually aware middle ground with modest runtime cost.

Sample throughput from earlier benchmarks (4 vCPU, 8 GB RAM environment):
- SSIM ≈ 2.1 frames/s
- Frame Difference ≈ 37.1 frames/s
- Optical Flow ≈ 1.2 frames/s

Actual speeds vary with resolution, codec, and hardware.

## Contributing
Issues and pull requests are welcome. Focus areas include new pruning heuristics, improved progress reporting, and GPU-accelerated kernels.

## Support
Questions or feedback? Reach out at `hello@karanprasad.com`.
