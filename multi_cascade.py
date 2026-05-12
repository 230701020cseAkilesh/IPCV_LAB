"""
multi_cascade.py
Reusable cascade wrapper for Topic 1 - Algorithm C
Roll: 2116230701020
"""
import cv2

DEFAULT_CASCADES = {
    "face":  "cascades/haarcascade_frontalface_default.xml",
    "eye":   "cascades/haarcascade_eye.xml",
    "body":  "cascades/haarcascade_fullbody.xml",
}

COLORS = {
    "face": (50,  200, 255),
    "eye":  (50,  255, 130),
    "body": (255, 100,  50),
}

def load_cascades(cascade_dict=None):
    cascade_dict = cascade_dict or DEFAULT_CASCADES
    classifiers = {}
    for name, path in cascade_dict.items():
        clf = cv2.CascadeClassifier(path)
        if clf.empty():
            print(f"[WARN] Could not load cascade: {path}")
        else:
            classifiers[name] = clf
            print(f"[OK]   Loaded: {name} <- {path}")
    return classifiers

def detect(gray_image, classifiers, scale=1.1, min_neighbors=5, min_size=(30, 30)):
    results = {}
    for name, clf in classifiers.items():
        dets = clf.detectMultiScale(
            gray_image,
            scaleFactor=scale,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        results[name] = dets if len(dets) > 0 else []
    return results

def draw(frame, results):
    total = 0
    for label, boxes in results.items():
        color = COLORS.get(label, (200, 200, 200))
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            total += 1
    return frame, total
