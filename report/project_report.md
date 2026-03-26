# Smart Road Safety Analyzer — Project Report

**Course:** CSE3010 — Computer Vision  
**Project Type:** Bring Your Own Project (BYOP)

---

## 1. Problem Statement

Road safety is a universal concern that affects everyone who walks, cycles, or drives. Three specific problems are addressed:

1. **Potholes** — Damaged road surfaces cause vehicle damage, tire blowouts, and accidents. Manual road surveys are slow and inconsistent.
2. **Lane discipline** — Poor lane adherence is a major cause of road accidents, especially on highways.
3. **Pedestrian and vehicle proximity** — Detecting road users is critical for collision avoidance and traffic management systems.

These are real, observable problems experienced daily — on campus roads, city streets, and highways.

## 2. Why This Problem Matters

- India records over **150,000 road accident deaths annually** (MoRTH, 2022)
- Potholes alone cause **thousands of accidents** every monsoon season
- Automated visual inspection can **replace manual surveys**, enabling faster response
- The techniques used here form the foundation of **ADAS (Advanced Driver Assistance Systems)** used in modern vehicles

## 3. Approach & Solution

The project builds a modular computer vision pipeline that processes road images and dashcam video:

### Architecture

The system is organized into independent, reusable modules that mirror the course syllabus:

| Module | File | Purpose |
|--------|------|---------|
| Preprocessing | `src/preprocessing.py` | Image enhancement, noise reduction, transforms |
| Feature Extraction | `src/feature_extraction.py` | Edge, corner, and keypoint detection |
| Segmentation | `src/segmentation.py` | Region-based and graph-based segmentation |
| Detection | `src/detection.py` | Object detection using HOG+SVM and Haar cascades |
| Motion Analysis | `src/motion_analysis.py` | Optical flow, background subtraction, tracking |
| Lane Detection | `src/lane_detection.py` | End-to-end lane detection pipeline |
| Pothole Detection | `src/pothole_detection.py` | Multi-method pothole detection with severity scoring |

### Key Algorithms & Decisions

**1. Lane Detection: Hough Transform vs. Deep Learning**

I chose the classical Hough Transform approach because:
- It directly implements Module 3 syllabus concepts (Hough Line Transform)
- It's interpretable — each step can be visualized and understood
- No training data required — works out-of-the-box
- Combined with temporal smoothing, it works well for structured road environments

The pipeline: CLAHE → Canny → ROI masking → Probabilistic Hough Transform → lane averaging.

**2. Pothole Detection: Multi-Method Fusion**

Single-method pothole detection is unreliable. I combined:
- **Intensity-based detection** (potholes are darker due to shadows/depth)
- **Texture analysis using Gabor filters** (surface irregularities — Shape from Texture, Module 5)
- **Contour analysis** (shape properties like circularity and area)

The intersection of dark regions AND texture anomalies reduces false positives.

**3. Pedestrian Detection: HOG + SVM**

Used OpenCV's pre-trained HOG+SVM detector because:
- HOG descriptors are a core Module 3/4 concept
- The Dalal & Triggs method is the foundational approach in the field
- Pre-trained on INRIA dataset — robust and well-validated

**4. Motion Analysis: Dual Optical Flow**

Implemented both sparse (Lucas-Kanade) and dense (Farneback) optical flow:
- **Sparse**: Efficient for tracking individual features — used with centroid tracker
- **Dense**: Provides complete motion field — used for speed estimation and motion heatmaps
- **MOG2 background subtraction**: Complements optical flow for detecting moving objects

## 4. Challenges Faced

1. **Pothole false positives and illumination dependency**: Initially, dark shadows triggered false detections. This was solved by combining intensity and texture. However, real-world testing revealed a profound limitation: **water-filled potholes reflecting the bright sky**. Because the algorithm assumes potholes are "dark regions", it completely missed severe potholes that were highly reflective. The Gabor texture filters successfully found the edges, but the darkness threshold discarded them.
   
2. **Lane detection on curved roads**: The Hough Transform assumes straight lines. Mitigated with a smaller ROI focused on the near-road region where curves appear more linear.

3. **Speed estimation accuracy**: Without proper camera calibration, pixel-to-world-unit conversion is approximate. Documented this as a known limitation.

4. **Domain mismatch with Haar Cascades**: During testing, the system confidently bounded patches of trees as "vehicles". This occurred because the required `haarcascade_car.xml` was missing from the standard environment, forcing a fallback to `haarcascade_frontalface_default.xml`. The algorithm searched a forest for faces, found textural patterns resembling eyes/noses in the leaves, and hallucinated false positives. This vividly demonstrated the fragility of sliding-window cascade classifiers when applied out-of-domain.

## 5. What I Learned

1. **Classical CV is still powerful**: For structured problems like lane detection, classical methods (Canny + Hough) work well without any deep learning.

2. **Preprocessing matters enormously**: CLAHE and bilateral filtering dramatically improved results across all modules. Bad preprocessing = bad everything downstream.

3. **Multi-method fusion has blind spots**: No single technique is sufficient, but even combined methods (intensity + texture) fail when environmental conditions violate core heuristic assumptions (e.g., a pothole filled with water acts like a mirror, invalidating the "dark region" assumption).

4. **Domain specificity of classical detectors**: Haar Cascades are highly domain-specific. Feeding a forest image to a face cascade looking for cars results in hallucinated bounding boxes in the trees. You cannot simply swap classifiers without retraining.

5. **The importance of non-maximum suppression**: Without NMS, sliding-window detectors (like HOG) produce overwhelming overlapping boxes. NMS is a critical post-processing step.

6. **Optical flow assumptions**: The brightness constancy assumption breaks down with illumination changes, which is common in outdoor driving scenarios.

7. **Modular design pays off**: Separating each technique into its own module made debugging much easier and allowed reusing components across pipelines.

## 6. Results

The system successfully:
- ✅ Detects lane markings on structured road images
- ✅ Identifies potholes with severity scoring (Minor/Moderate/Severe)
- ✅ Detects pedestrians using HOG+SVM
- ✅ Detects vehicles using Haar cascades
- ✅ Tracks moving objects across video frames
- ✅ Computes dense optical flow and motion heatmaps
- ✅ Estimates approximate speed from flow magnitude

### Limitations
- Lane detection assumes relatively straight roads (Hough Transform limitation)
- Pothole detection fails dramatically on wet, highly reflective potholes (breaks the darkness assumption)
- Haar cascades produce severe false positives when applied out-of-domain (e.g., tree textures mistaken for objects)
- Speed estimation is uncalibrated (approximate without camera intrinsics)

## 7. Future Work

- Replace Hough-based lanes with polynomial fitting or deep learning (LaneNet)
- Use YOLO/SSD for more robust multi-class object detection
- Implement proper camera calibration for accurate speed measurement
- Add GPS integration for pothole location mapping
- Build a web interface for real-time dashcam streaming analysis

## 8. References

1. Szeliski, R. (2011). *Computer Vision: Algorithms and Applications*. Springer.
2. Forsyth, D.A., & Ponce, J. (2003). *Computer Vision: A Modern Approach*. Pearson.
3. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. *CVPR*.
4. Lucas, B.D., & Kanade, T. (1981). An Iterative Image Registration Technique. *IJCAI*.
5. Farneback, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. *SCIA*.
6. Zivkovic, Z. (2004). Improved Adaptive Gaussian Mixture Model for Background Subtraction. *ICPR*.
7. Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: Interactive Foreground Extraction. *SIGGRAPH*.
