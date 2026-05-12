"""
main.py  —  Real-Time Multi-Object Detection using CascadeClassifier
Roll No.: 2116230701020
Topic 1, Algorithm C: cv2.CascadeClassifier (Haar/LBP)
Subject: AD23B31 Image Processing and Computer Vision
"""

import cv2
import os
import sys
import time

from preprocess import preprocess
from multi_cascade import load_cascades, detect, draw


def setup_cascades():
    """Copy cascade XMLs from OpenCV's built-in data directory."""
    import shutil
    src = cv2.__file__.replace("__init__.py", "data/")
    os.makedirs("cascades", exist_ok=True)
    needed = [
        "haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml",
        "haarcascade_fullbody.xml",
    ]
    for f in needed:
        dst = os.path.join("cascades", f)
        if not os.path.exists(dst):
            full = os.path.join(src, f)
            if os.path.exists(full):
                shutil.copy(full, dst)
                print(f"[SETUP] Copied {f}")
            else:
                print(f"[WARN]  Not found in OpenCV data: {f}")


def run_on_images(classifiers, img_dir="dataset/images", out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    images = [f for f in os.listdir(img_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        print(f"[WARN] No images found in {img_dir}")
        return

    print(f"[INFO] Processing {len(images)} images...")
    fps_list = []

    for img_file in sorted(images):
        path  = os.path.join(img_dir, img_file)
        frame = cv2.imread(path)
        if frame is None:
            print(f"[SKIP] {img_file}")
            continue

        t0      = time.time()
        gray    = preprocess(frame)
        results = detect(gray, classifiers)
        elapsed = (time.time() - t0) * 1000
        fps_list.append(1000.0 / max(elapsed, 1.0))

        annotated, total = draw(frame.copy(), results)
        out_path = os.path.join(out_dir, "det_" + img_file)
        cv2.imwrite(out_path, annotated)

        breakdown = {k: len(v) for k, v in results.items() if len(v) > 0}
        print(f"  {img_file:35s}  objects={total:3d}  {breakdown}  "
              f"time={elapsed:.1f}ms")

    if fps_list:
        print(f"\n[INFO] Avg FPS on dataset: {sum(fps_list)/len(fps_list):.2f}")


def run_on_webcam(classifiers, cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("[ERR] Cannot open webcam.")
        return

    print("[INFO] Webcam running. Press Q to quit.")
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.time()
        gray    = preprocess(frame)
        results = detect(gray, classifiers)
        elapsed = time.time() - t0
        fps     = 1.0 / max(elapsed, 1e-9)
        fps_list.append(fps)

        annotated, total = draw(frame, results)
        cv2.putText(annotated,
                    f"FPS: {fps:.1f}  Objects: {total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Multi-Object Detection | 2116230701020", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if fps_list:
        print(f"[INFO] Avg FPS (webcam): {sum(fps_list)/len(fps_list):.2f}")


def haar_vs_lbp_experiment(img_dir="dataset/images"):
    """Compare Haar vs LBP on the same images. Print side-by-side metrics."""
    import shutil

    # Try to copy LBP cascade
    src = cv2.__file__.replace("__init__.py", "data/")
    lbp_src = os.path.join(src, "lbpcascade_frontalface_improved.xml")
    lbp_dst = "cascades/lbpcascade_frontalface_improved.xml"
    if os.path.exists(lbp_src) and not os.path.exists(lbp_dst):
        shutil.copy(lbp_src, lbp_dst)

    for name, path in [
        ("Haar", "cascades/haarcascade_frontalface_default.xml"),
        ("LBP",  "cascades/lbpcascade_frontalface_improved.xml"),
    ]:
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            print(f"[SKIP] {name} cascade not available.")
            continue

        images   = [f for f in os.listdir(img_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        total_det, total_ms = 0, 0.0

        for img_file in images:
            frame = cv2.imread(os.path.join(img_dir, img_file))
            if frame is None:
                continue
            gray = preprocess(frame)
            t0   = time.time()
            dets = clf.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            total_ms  += (time.time() - t0) * 1000
            total_det += len(dets) if len(dets) > 0 else 0

        n = max(len(images), 1)
        print(f"[{name:4s}] Total detections: {total_det:4d}  "
              f"Avg time/image: {total_ms/n:.1f}ms  "
              f"Avg FPS: {1000/(total_ms/n):.1f}")


if __name__ == "__main__":
    setup_cascades()
    classifiers = load_cascades()

    if not classifiers:
        print("[ERR] No classifiers loaded. Check cascade XML files.")
        sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "images"

    if mode == "webcam":
        run_on_webcam(classifiers)
    elif mode == "experiment":
        haar_vs_lbp_experiment()
    else:
        run_on_images(classifiers)
