# Anime Frame Optimizer (AFOptimizer)

## Overview
AFOptimizer removes redundant frames from anime and stylised video sources. The project ships with two entry points:
- **Web Dashboard** (`app.py`): A Flask-based web interface that lets you upload footage, watch live progress, and download processed clips.
- **Command-Line Interface** (`cli.py`): A flexible CLI with per-method fine-tuning flags for automation and batch work.

Under the hood, AFOptimizer bundles four complementary pruning strategies-Optical Flow, Frame Difference, SSIM, and an advanced Unsupervised Deduplication pipeline-so you can mix accuracy, speed, and perceptual quality to fit each project.

## Key Features
- **Multi-method optimisation**: Dense motion analysis, adaptive pixel differencing, perceptual similarity, and a three-stage unsupervised deduper.
- **GPU acceleration**: Automatic detection and utilization of NVIDIA, AMD, Intel, and Apple Silicon GPUs for faster processing.
- **Hardware-accelerated encoding**: Uses NVENC, AMF, QuickSync, or VideoToolbox when available for faster video encoding.
- **Web interface**: Browser workflow with background job queue, live progress updates, downloadable artefacts, and automatic upload cleanup.
- **Command-line execution**: Reusable defaults, opt-in presets, and granular override flags for every method parameter.
- **Video encoding**: Automatic H.264 fast-start transcode plus optional encoding controls for CRF and preset.
- **Built-in safeguards**: Safety windows between keyframes and adaptive thresholding to limit over-pruning.

## Supported Formats
Input formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`  
Output format: H.264 MP4 (compatible with all modern players)

## Setup
1. Create and activate a Python environment (Python 3.9+ recommended):
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `ffmpeg` and ensure it is discoverable on your `PATH` (required for video encoding):
   - **Linux**: `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. **(Optional) GPU Acceleration Setup**:
   AFOptimizer automatically detects and uses GPU acceleration when available. No configuration needed!
   
   **For NVIDIA GPUs (CUDA)**:
   - Install CUDA toolkit and drivers
   - Install OpenCV with CUDA support (or build from source)
   - Optional: `pip install cupy-cuda11x` or `cupy-cuda12x` (matching your CUDA version)
   
   **For AMD GPUs**:
   - Install AMD drivers and OpenCL runtime
   - Optional: `pip install pyopencl`
   
   **For Intel integrated GPUs**:
   - Install Intel OpenCL drivers
   - Optional: `pip install pyopencl`
   
   **For Apple Silicon (M1/M2/M3)**:
   - No additional setup needed! VideoToolbox is automatically used if available.
   
   The system gracefully falls back to CPU processing if GPU is not available. You can test GPU detection by running:
   ```bash
   python3 test_gpu_detection.py
   ```

## Web Dashboard (Flask App)

### Starting the Server
Launch the Flask application:
```bash
python3 app.py
```

The server will start on `http://0.0.0.0:5000` (accessible at `http://localhost:5000` in your browser).

### Using the Web Interface

1. **Upload Video**: 
   - Drag and drop a video file or click to browse
   - Supported formats: MP4, MOV, AVI, MKV, WEBM

2. **Choose Method**:
   - **Optical Flow**: Detects motion by tracking pixel flow for smooth transitions
   - **Frame Difference**: Flags static frames by comparing brightness deltas
   - **SSIM**: Keeps only frames with meaningful structural change
   - **Unsupervised Dedup**: Three-stage hashing, features, and motion flow to drop redundant frames

3. **Fine-Tune Parameters**:
   - **Optical Flow**: Adjust flow magnitude threshold (default: 0.4)
   - **Frame Difference**: Adjust base threshold (default: 10)
   - **SSIM**: Adjust SSIM threshold (default: 0.9587)
   - **Unsupervised Dedup**: Choose profile (Gentle/Balanced/Aggressive)

4. **Monitor Progress**:
   - Real-time progress bar with percentage completion
   - Live FPS (frames per second) processing rate
   - Elapsed time and estimated time remaining (ETA)
   - Current processing stage information

5. **Download Results**:
   - Once processing completes, a download link appears
   - Processed videos are saved in the `outputs/` directory
   - Original uploads are automatically cleaned up after processing

### Web Interface Features
- **Background Processing**: Jobs run in separate threads, allowing multiple uploads
- **Live Updates**: Progress updates via AJAX polling every few seconds
- **Automatic Cleanup**: Temporary upload files are removed after job completion
- **Error Handling**: Clear error messages displayed if processing fails
- **Responsive Design**: Works on desktop and mobile browsers

### Directory Structure
- `uploads/`: Temporary storage for uploaded videos (auto-cleaned after processing)
- `outputs/`: Final processed videos (persistent, available for download)

## Command-Line Interface

The CLI provides full control over all optimization methods with fine-grained parameter tuning. It's ideal for batch processing, automation, and integration into larger workflows.

### Basic Usage
Run `python3 cli.py --help` to see the full reference. The general pattern is:

```bash
python3 cli.py [GLOBAL OPTIONS] INPUT_VIDEO METHOD [METHOD OPTIONS]
```

### Global Options
- `-o, --output PATH` – Specify the output file path (defaults to `input_stem` + method suffix in the same directory)
- `--encoding-crf VALUE` – Re-encode the final file with ffmpeg CRF value (lower = higher quality, default 18 when used)
  - Recommended range: 18-28 (18 = high quality, 23 = default, 28 = smaller file)
- `--encoding-preset NAME` – ffmpeg encoding preset (default `medium` when used)
  - Options: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`, `slower`, `veryslow`
  - Faster presets = larger files, slower presets = smaller files

### Methods and Options

#### Optical Flow (`optical-flow`)
**Description**: Keeps frames that show meaningful pixel-wise motion between consecutive frames. Best for action-heavy sequences with smooth motion.

**How it works**: Runs Farnebäck dense optical flow, computes mean vector magnitude, and writes frames whose magnitude exceeds the threshold.

**Options**:
- `--flow-mag-threshold FLOAT` (default: `0.4`)
  - Lower values = keep more frames (less aggressive)
  - Higher values = prune subtle motion (more aggressive)
  - Recommended range: 0.2-0.6

**Example**:
```bash
python3 cli.py ~/videos/episode01.mp4 optical-flow --flow-mag-threshold 0.35 -o ~/outputs/episode01_of.mp4
```

#### Frame Difference (`frame-difference`)
**Description**: Compares brightness changes between frames and drops segments with negligible deltas. Fast and effective for static scenes.

**How it works**: Adapts the supplied base threshold using an initial sampling window, then counts high-difference pixels to decide whether to preserve a frame.

**Options**:
- `--base-threshold FLOAT` (default: `10.0`)
  - Increase to demand larger pixel swings before keeping a frame
  - Recommended range: 5-20

**Example**:
```bash
python3 cli.py ~/videos/episode01.mp4 frame-difference --base-threshold 14 -o ~/outputs/episode01_fd.mp4
```

#### SSIM (`ssim`)
**Description**: Focuses on perceptual similarity-removes frames that look virtually identical to the previous one. Good balance between quality and speed.

**How it works**: Calculates grayscale Structural Similarity (SSIM) for each frame pair and writes frames whose SSIM falls below the cutoff. Always appends the final frame.

**Options**:
- `--ssim-threshold FLOAT` (default: `0.9587`)
  - Lower thresholds = stricter pruning (more frames removed)
  - Higher thresholds = retain more visually similar frames
  - Recommended range: 0.90-0.99

**Example**:
```bash
python3 cli.py ~/videos/episode01.mp4 ssim --ssim-threshold 0.97 -o ~/outputs/episode01_ssim.mp4 --encoding-crf 20
```

#### Unsupervised Deduplication (`unsupervised-dedup`)
**Description**: Combines perceptual hashing, feature clustering, and motion analysis to suppress redundant footage without labeled data. Ideal for long episodic sources.

**How it works**: Three-stage pipeline:
1. **Walsh–Hadamard hashing** with ordinal texture checks
2. **ORB-based feature clustering** for visual similarity
3. **Motion gating** using downscaled optical flow with safety keyframe spacing

**Presets**:
- `--profile {gentle|balanced|aggressive}` (default: `balanced`)
  - `gentle`: Keeps more frames, less aggressive pruning
  - `balanced`: Recommended mix for most videos
  - `aggressive`: More aggressive pruning, smaller output files

**Fine-tune Options** (all optional; override preset values):
- `--hash-threshold INT` – Hamming distance for hash matches (default varies by profile)
- `--ordinal-footrule-threshold FLOAT` – Maximum footrule distance between ordinal signatures
- `--feature-similarity FLOAT` – ORB match ratio required to treat frames as equivalent (0.0-1.0)
- `--flow-static-threshold FLOAT` – Mean flow magnitude treated as static
- `--flow-low-ratio FLOAT` – Fraction of low-motion pixels necessary for static gating (0.0-1.0)
- `--pan-orientation-std FLOAT` – Orientation spread threshold for detecting pans
- `--safety-keep-seconds FLOAT` – Minimum seconds between forced keyframes to avoid over-pruning

**Examples**:
```bash
# Using preset
python3 cli.py ~/videos/season01.mkv unsupervised-dedup --profile aggressive -o ~/outputs/season01_dedup.mp4

# Customizing preset parameters
python3 cli.py ~/videos/season01.mkv unsupervised-dedup --profile balanced --safety-keep-seconds 2.0 --hash-threshold 10 -o ~/outputs/season01_custom.mp4
```

### Output
All methods produce H.264 MP4 files. Use `--encoding-crf` and `--encoding-preset` for tighter control over bitrate/quality trade-offs.

## Method Comparison

### Performance Characteristics
- **Optical Flow**: Most accurate for complex motion but slower. Best for action sequences with smooth camera movement.
- **Frame Difference**: Fastest method, lightweight. Best for static scenes with obvious changes.
- **SSIM**: Good balance between quality and speed. Perceptually aware, suitable for most content.
- **Unsupervised Dedup**: Most sophisticated, slower but most accurate for long-form content. Best for episodic videos with repeated scenes.

### Performance Benchmarks
Sample throughput from benchmarks (4 vCPU, 8 GB RAM environment):
- **Frame Difference**: ≈ 37.1 frames/s (fastest)
- **SSIM**: ≈ 2.1 frames/s (moderate)
- **Optical Flow**: ≈ 1.2 frames/s (slower)
- **Unsupervised Dedup**: ≈ 0.8-1.5 frames/s (slowest, varies by profile)

**Note**: Actual speeds vary significantly with:
- Video resolution (1080p vs 4K)
- Source codec and bitrate
- Hardware (CPU cores, RAM, disk I/O)
- Video length and complexity

### Choosing the Right Method

| Use Case | Recommended Method | Why |
|----------|-------------------|-----|
| Fast processing needed | Frame Difference | Fastest method, good for quick previews |
| Action sequences | Optical Flow | Best motion detection |
| General purpose | SSIM | Good balance of quality and speed |
| Long episodes/series | Unsupervised Dedup | Most accurate for removing repeated content |
| Static scenes | Frame Difference | Efficient for minimal motion |
| High quality needed | SSIM or Optical Flow | Better perceptual quality |

## Workflow Examples

### Web Interface Workflow
1. Start the server: `python3 app.py`
2. Open browser to `http://localhost:5000`
3. Upload video file
4. Select optimization method
5. Adjust parameters (or use defaults)
6. Click "Process Video"
7. Monitor progress in real-time
8. Download completed video when ready

### CLI Batch Processing Example
```bash
# Process multiple videos with SSIM
for video in ~/videos/*.mp4; do
    python3 cli.py "$video" ssim --ssim-threshold 0.96 -o ~/outputs/$(basename "$video" .mp4)_optimized.mp4
done

# Process with custom encoding settings
python3 cli.py input.mp4 optical-flow --flow-mag-threshold 0.4 \
    --encoding-crf 20 --encoding-preset slow -o output.mp4
```

## Troubleshooting

### Common Issues

**"ffmpeg not found" error**
- Ensure ffmpeg is installed and available in your PATH
- Test with: `ffmpeg -version`
- On Linux, you may need to install: `sudo apt-get install ffmpeg`

**"No video file provided" (Web Interface)**
- Ensure you've selected a file before clicking "Process Video"
- Check that the file format is supported (MP4, MOV, AVI, MKV, WEBM)

**Processing is very slow**
- This is normal for high-resolution videos or complex methods
- Try Frame Difference for faster processing
- Consider reducing video resolution before processing
- Check available system resources (CPU, RAM)

**Output file is too large/small**
- Adjust method-specific thresholds (lower = keep more frames)
- Use `--encoding-crf` in CLI to control file size (higher CRF = smaller file)
- Try different methods to find the right balance

**Job fails or shows error**
- Check that input video is not corrupted
- Ensure sufficient disk space in `uploads/` and `outputs/` directories
- Review error message in web interface or CLI output
- Try a different optimization method

## Technical Details

### Architecture
- **Backend**: Flask web server with threading for background job processing
- **Video Processing**: OpenCV for frame extraction and analysis
- **Encoding**: ffmpeg for H.264 encoding and transcoding
- **Progress Tracking**: Real-time updates via in-memory job queue

### File Structure
```
AFOptimizer/
├── app.py                 # Flask web application
├── cli.py                 # Command-line interface
├── frame_optimization_methods/
│   ├── opticalFlow.py     # Optical flow method
│   ├── frameDifference.py # Frame difference method
│   ├── ssim.py            # SSIM method
│   ├── unsupervised_dedup.py  # Unsupervised deduplication
│   └── video_encoding.py  # H.264 encoding utilities
├── templates/
│   └── index.html         # Web interface template
├── static/
│   ├── css/style.css      # Stylesheet
│   └── js/main.js         # Frontend JavaScript
├── uploads/               # Temporary upload storage
└── outputs/               # Processed video output
```

## GPU Acceleration

AFOptimizer includes comprehensive GPU acceleration support that automatically detects and utilizes available hardware:

### Supported GPUs
- **NVIDIA GPUs**: CUDA acceleration for frame processing, NVENC for encoding
- **AMD GPUs**: OpenCL/ROCm support, AMF encoder for video encoding
- **Intel integrated GPUs**: QuickSync Video (QSV) encoding, OpenCL support
- **Apple Silicon**: Metal acceleration, VideoToolbox encoding

### How It Works
1. **Automatic Detection**: On startup, AFOptimizer detects available GPU hardware
2. **Smart Fallback**: If GPU is unavailable or fails, the system automatically falls back to CPU
3. **Hardware Encoding**: Video encoding uses hardware encoders (NVENC, AMF, QSV, VideoToolbox) when available
4. **GPU-Accelerated Operations**: Frame processing operations (color conversion, resizing, blur, optical flow preprocessing) use GPU when available

### Performance Benefits
GPU acceleration can provide significant speedups:
- **Frame processing**: 2-5x faster on supported GPUs
- **Video encoding**: 5-10x faster with hardware encoders
- **Overall workflow**: 3-7x faster end-to-end processing

### Testing GPU Detection
Run the included test script to check your GPU setup:
```bash
python3 test_gpu_detection.py
```

This will show:
- Detected GPU backend (CUDA, OpenCL, Metal, etc.)
- Device name
- Availability status
- Hardware encoder support

### Troubleshooting GPU Issues
- **"Using CPU processing" message**: This is normal if no GPU is available or GPU libraries aren't installed
- **GPU detection fails**: Ensure GPU drivers are installed and OpenCV has GPU support
- **Encoding fails with hardware encoder**: System automatically falls back to CPU encoding
- **Performance not improved**: Some operations (like SSIM calculation) still use CPU; overall speedup depends on workload

## Contributing
Issues and pull requests are welcome! Focus areas include:
- New pruning heuristics and optimization methods
- Improved progress reporting and UI enhancements
- Additional GPU acceleration optimizations
- Performance optimizations
- Documentation improvements

## License
See [LICENSE](LICENSE) file for details.

## Support
Questions or feedback? Reach out at `hello@karanprasad.com`.
