
 Mini Project: Real-Time Multi-Object Detection
 Roll No.: 2116230701020
 Algorithm C: cv2.CascadeClassifier (Haar/LBP)
 Subject: AD23B31 — Image Processing and CV
 Academic Year: 2025–2026
===================================================

SETUP
-----
1. pip install opencv-python numpy

2. Run setup (copies cascade XMLs from OpenCV):
   python main.py

   This auto-copies Haar/LBP cascade XMLs into cascades/

3. Add your dataset images to:
   dataset/images/
   dataset/annotations/instances_val.json   (COCO format)

USAGE
-----
# Detect on all images in dataset/images/
python main.py images

# Real-time webcam detection
python main.py webcam

# Haar vs LBP speed/accuracy experiment
python main.py experiment

# Evaluate Precision / Recall / F1 (needs annotations JSON)
python evaluate.py

OUTPUT
------
Annotated images are saved to output/det_<filename>.jpg

FILE LISTING
------------
main.py          — main script (image batch + webcam + experiment modes)
preprocess.py    — resize + denoise + histogram equalisation pipeline
multi_cascade.py — cascade loader and multi-class detect/draw wrapper
evaluate.py      — IoU-based Precision / Recall / F1 calculator
cascades/        — Haar and LBP XML files (auto-populated by main.py)
dataset/         — place your images and annotations here
output/          — annotated output images written here


# Real-Time Multi-Object Detection using Haar & LBP Cascades 👁️📷

## Project Overview
This mini project demonstrates a real-time multi-object detection system using classical computer vision techniques in OpenCV. The application detects multiple object categories such as faces, eyes, full body, smile, and upper body using Haar Cascade and LBP Cascade classifiers.

The system supports:
- Static image detection
- Real-time webcam detection
- Haar vs LBP performance comparison
- Precision / Recall / F1 evaluation
- Preprocessing for improved detection quality

This project is developed as part of the subject:

AD23B31 — Image Processing and Computer Vision  
Academic Year: 2024–2025

--------------------------------------------------

# Objectives 🎯

- To understand classical object detection techniques
- To implement Haar and LBP cascade classifiers
- To perform real-time object detection using webcam feed
- To compare speed and accuracy of Haar and LBP methods
- To evaluate detection performance using IoU metrics
- To apply preprocessing techniques for better results

--------------------------------------------------

# Features ✨

✅ Multi-object detection  
✅ Real-time webcam support  
✅ Batch image processing  
✅ Image preprocessing pipeline  
✅ Bounding box visualization  
✅ Performance evaluation metrics  
✅ Haar vs LBP comparison experiment  
✅ Automatic cascade XML setup  
✅ COCO annotation support  
✅ Output image saving

--------------------------------------------------

# Technologies Used 🛠️

- Python
- OpenCV
- NumPy
- Haar Cascades
- LBP Cascades
- COCO Dataset Format

--------------------------------------------------

# System Workflow 🔄

1. Input image or webcam frame is captured
2. Image preprocessing is applied:
   - Resizing
   - Noise reduction
   - Histogram equalisation
3. Cascade classifiers detect objects
4. Bounding boxes are drawn
5. Results are displayed and saved
6. Evaluation metrics are computed

--------------------------------------------------

# Supported Detection Types 🔍

The system can detect:
- Human face
- Eyes
- Smile
- Full body
- Upper body

Additional cascade XML files can also be added for:
- Cars
- Pedestrians
- Cats
- License plates

--------------------------------------------------

# Folder Structure 📁

project/

├── main.py  
├── preprocess.py  
├── multi_cascade.py  
├── evaluate.py  

├── cascades/  
├── dataset/  
├── output/  

└── README.md

--------------------------------------------------

# Preprocessing Pipeline 🧠

The preprocessing module improves detection quality by:
- Reducing image noise
- Enhancing contrast
- Normalizing illumination
- Improving edge visibility

Techniques used:
- Gaussian Blur
- Histogram Equalization
- Grayscale Conversion
- Image Resizing

--------------------------------------------------

# Haar vs LBP Comparison ⚖️

| Parameter | Haar Cascade | LBP Cascade |
|-----------|--------------|--------------|
| Speed | Moderate | Faster |
| Accuracy | Higher | Moderate |
| Lighting Sensitivity | High | Lower |
| Computational Cost | Higher | Lower |
| Real-time Performance | Good | Very Good |

Observation:
- Haar provides better detection accuracy in controlled lighting.
- LBP performs faster in real-time applications with lower computational power.

--------------------------------------------------

# Evaluation Metrics 📊

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

IoU = Area of Overlap / Area of Union

--------------------------------------------------

# Advantages ✅

- Simple and lightweight implementation
- Works without GPU
- Fast real-time processing
- Easy to understand and modify
- Suitable for academic learning
- Low memory requirements

--------------------------------------------------

# Limitations ⚠️

- Less accurate than deep learning models
- Sensitive to object orientation
- Performance decreases under poor lighting
- High false positives in complex backgrounds
- Limited robustness for large-scale detection

--------------------------------------------------

# Future Enhancements 🚀

Possible improvements:
- Integrate deep learning models like YOLO or SSD
- Add object tracking
- Improve GUI using Tkinter or PyQt
- Deploy as a web application
- Add cloud-based image processing
- Use GPU acceleration
- Mobile app integration

--------------------------------------------------

# Applications 🌍

- Smart surveillance systems
- Attendance monitoring
- Traffic monitoring
- Human activity analysis
- Security systems
- Smart classrooms
- Face-based authentication

--------------------------------------------------

# Sample Commands 💻

python main.py images

python main.py webcam

python main.py experiment

python evaluate.py

--------------------------------------------------

# Expected Output 🖼️

The system generates:
- Annotated detection images
- Bounding boxes around detected objects
- Real-time webcam visualization
- Evaluation statistics
- Experiment comparison results

Output files are automatically saved in:

output/

--------------------------------------------------

# Conclusion 📌

This project successfully demonstrates real-time multi-object detection using traditional computer vision algorithms. Haar and LBP cascade classifiers provide efficient object detection with low computational requirements, making them suitable for educational purposes and lightweight applications.

The project also highlights the importance of preprocessing, evaluation metrics, and performance comparison in practical computer vision systems.

--------------------------------------------------

# References 📚

1. OpenCV Official Documentation
2. OpenCV Cascade Classifier Docs
3. Viola-Jones Object Detection Research Paper
4. LBP Research Papers
5. COCO Dataset Documentation
