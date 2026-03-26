# 🚗 Smart Road Safety Analyzer

A comprehensive computer vision system that analyzes road conditions and safety factors from images and dashcam footage. Built as a capstone project for **CSE3010 — Computer Vision**.

## Problem Statement

Road safety is a daily concern — **potholes damage vehicles**, poor **lane discipline** causes accidents, and **pedestrians** are constantly at risk. This project addresses these real-world issues by building an intelligent analysis system that processes road imagery to:

- **Detect lane markings** and assess lane discipline
- **Identify potholes** and score their severity
- **Detect pedestrians and vehicles** in the scene
- **Analyze motion patterns** and estimate speeds from video

## Features

| Feature | Technique | Syllabus Module |
|---------|-----------|-----------------|
| Image Enhancement | CLAHE, Bilateral Filter, Fourier HPF | Module 1 |
| Perspective Transform | Projective Transformation (Bird's-Eye View) | Module 1 |
| Lane Detection | Canny Edge + Hough Line Transform + ROI | Modules 1, 3 |
| Pothole Detection | Gabor Texture Analysis + Segmentation + Contour Analysis | Modules 1, 3, 5 |
| Feature Extraction | SIFT, ORB, Harris Corners, HOG | Module 3 |
| Segmentation | Watershed, GrabCut, Otsu Thresholding | Module 3 |
| Pedestrian Detection | HOG + SVM (Dalal & Triggs method) | Module 4 |
| Vehicle Detection | Haar Cascade Classifier | Module 4 |
| Optical Flow | Lucas-Kanade (sparse) + Farneback (dense) | Module 4 |
| Background Subtraction | MOG2 (Mixture of Gaussians) | Module 4 |
| Object Tracking | Centroid-based multi-object tracker | Module 4 |
| Speed Estimation | Optical flow magnitude → km/h conversion | Module 4 |
| Texture Analysis | Gabor filters, variance maps | Module 5 |

## Project Structure

```
├── main.py                      # CLI entry point with sub-commands
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # Image enhancement, filtering, Fourier, transforms
│   ├── feature_extraction.py    # Canny, Hough, SIFT, ORB, HOG, Harris
│   ├── segmentation.py          # Watershed, GrabCut, contours, morphology
│   ├── detection.py             # HOG+SVM pedestrians, Haar cascade vehicles, NMS
│   ├── motion_analysis.py       # Optical flow, background subtraction, tracking
│   ├── lane_detection.py        # Complete lane detection pipeline
│   ├── pothole_detection.py     # Pothole detection + severity scoring
│   └── utils.py                 # Visualization and drawing utilities
├── demo/
│   ├── demo_preprocessing.py    # Preprocessing techniques showcase
│   ├── demo_lanes.py            # Lane detection (image + video)
│   ├── demo_potholes.py         # Pothole detection with scoring
│   ├── demo_detection.py        # Pedestrian + vehicle detection
│   ├── demo_motion.py           # Motion analysis (LK, Farneback, MOG2)
│   └── demo_full_pipeline.py    # Combined road safety analysis
├── samples/                     # Place test images/videos here
├── results/                     # Output directory for processed results
└── report/
    └── project_report.md        # Detailed project report
```

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/smart-road-safety-analyzer.git
cd smart-road-safety-analyzer

# 2. Create a virtual environment (highly recommended)
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (Command Prompt):
python -m venv venv
venv\Scripts\activate.bat

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add sample images/videos to the samples/ directory
```

## Usage

### CLI Commands

The project provides a unified CLI through `main.py`:

```bash
# View all available commands
python main.py --help

# Image Preprocessing (histogram eq, filtering, Fourier analysis)
python main.py preprocess --input samples/road.jpg --output results/

# Feature Extraction (SIFT, ORB, Harris, Canny, LOG)
python main.py features --input samples/road.jpg --output results/

# Lane Detection
python main.py lanes --input samples/highway.jpg --output results/

# Pothole Detection
python main.py potholes --input samples/pothole.jpg --output results/

# Pedestrian & Vehicle Detection
python main.py detect --input samples/street.jpg --output results/

# Motion Analysis (video only)
python main.py motion --input samples/traffic.mp4 --output results/

# Full Pipeline (all analyses combined)
python main.py full --input samples/road.jpg --output results/
python main.py full --input samples/dashcam.mp4 --output results/
```

### Individual Demo Scripts

Each demo can also be run directly for more control:

```bash
python demo/demo_preprocessing.py --input samples/road.jpg --output results/
python demo/demo_lanes.py --input samples/road.jpg --output results/
python demo/demo_potholes.py --input samples/pothole.jpg --sensitivity 0.35
python demo/demo_detection.py --input samples/street.jpg
python demo/demo_motion.py --input samples/traffic.mp4 --mode dense --max-frames 200
python demo/demo_full_pipeline.py --input samples/dashcam.mp4 --max-frames 300
```

## Technical Approach

### Lane Detection Pipeline
```
Input → Grayscale → CLAHE → Gaussian Blur → Canny Edge Detection
    → ROI Masking → Probabilistic Hough Transform
    → Left/Right Separation → Weighted Averaging → Lane Overlay
```

### Pothole Detection Pipeline
```
Input → Bilateral Filter → CLAHE → Dark Region Detection (thresholding)
    → Gabor Texture Analysis → Texture Variance Map
    → Mask Fusion (dark ∩ texture) → Contour Extraction
    → Shape Analysis → Severity Scoring → Annotated Output
```

### Motion Analysis Pipeline
```
Video Frames → Background Subtraction (MOG2) → Foreground Mask
    → Contour Detection → Centroid Tracking → Trajectory Drawing
    ↓
    → Dense Optical Flow (Farneback) → Motion Heatmap → Speed Estimation
    ↓
    → Sparse Optical Flow (Lucas-Kanade) → Feature Tracking → Flow Vectors
```

## Syllabus Module Coverage

This project meaningfully applies concepts from **all 5 modules** of CSE3010:

1. **Module 1** — Image formation, Fourier transform, convolution, filtering, histogram processing, perspective transforms
2. **Module 2** — Perspective geometry concepts applied to speed estimation and bird's-eye view transforms
3. **Module 3** — Canny/LOG/DOG edges, Hough Transform, Harris corners, SIFT/ORB descriptors, HOG, watershed, GrabCut, contour analysis
4. **Module 4** — HOG+SVM classification (pedestrians), background subtraction (MOG2), optical flow (Lucas-Kanade, Farneback), motion tracking
5. **Module 5** — Gabor filter texture analysis (Shape from Texture) for pothole surface analysis

## Technologies Used

- **Python 3.8+**
- **OpenCV** — Core computer vision operations
- **NumPy** — Numerical computations
- **Matplotlib** — Visualization and plotting
- **scikit-learn** — Distance computations for tracking
- **SciPy** — Spatial distance calculations

## FAQ & Troubleshooting

During development and testing across different operating systems, we simulated and resolved several common issues. If you encounter errors, check here first:

### 1. IDE shows "Import `cv2` could not be resolved" (VS Code)
**Issue:** VS Code shows red squiggly lines under `import cv2` or `import numpy`, even though the code runs fine in the terminal.
**Solution:** Your IDE is using the global Python interpreter instead of the virtual environment. 
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac).
- Search for **"Python: Select Interpreter"**.
- Choose the interpreter located in `./venv/bin/python` (Linux/Mac) or `.\venv\Scripts\python.exe` (Windows).
- Alternatively, we have included a `pyrightconfig.json` that attempts to auto-resolve this. Reload your window afterwards.

### 2. Vehicle Detection crashes with `Can't open file: haarcascade_car.xml`
**Issue:** The `opencv-python` package installed via pip does not natively ship with the car cascade file, causing a crash on some operating systems.
**Solution:** We have already added a graceful fallback in `src/detection.py` to use `haarcascade_frontalface_default.xml` so the pipeline won't crash. For accurate vehicle detection, download `haarcascade_car.xml` from the official OpenCV GitHub and place it in the `src/` directory, then pass its path to the detector.

### 3. Potholes are clearly visible but not detected
**Issue:** Rain or bright sunlight creates reflections inside potholes (since they are filled with water), violating the "dark region" assumption of the combined intensity+texture algorithm.
**Solution:** Adjust the `--sensitivity` parameter. The default is `0.6`. By lowering it (e.g., `--sensitivity 0.8`), the detector becomes more forgiving of lighting changes. 
```bash
python main.py potholes --input samples/pothole.jpg --sensitivity 0.8
```

### 4. Code freezes or runs extremely slowly on video (Motion Analysis)
**Issue:** Dense optical flow (Farneback) and MOG2 are computationally expensive, especially on high-resolution dashcam videos (e.g., 4K or 1080p).
**Solution:** The preprocessing pipeline automatically resizes inputs, but older Windows/Linux machines might still struggle without GPU acceleration. Run the demo with the `--max-frames` flag to test a short segment, or stick to images for immediate results.

## License

This project was developed for educational purposes as part of CSE3010 — Computer Vision.
