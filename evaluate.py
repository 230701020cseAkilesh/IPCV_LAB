"""
evaluate.py  —  Precision / Recall / F1 Calculator
Roll No.: 2116230701020
Requires: dataset/annotations/instances_val.json (COCO format)
"""

import cv2
import json
import os

from preprocess import preprocess
from multi_cascade import load_cascades, detect


def iou(a, b):
    """Compute IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter   = inter_w * inter_h
    union   = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def evaluate(ann_file="dataset/annotations/instances_val.json",
             img_dir="dataset/images",
             iou_thresh=0.5,
             scale=1.1,
             min_neighbors=5):

    if not os.path.exists(ann_file):
        print(f"[ERR] Annotation file not found: {ann_file}")
        return

    with open(ann_file) as f:
        data = json.load(f)

    classifiers = load_cascades()
    TP = FP = FN = 0

    for item in data.get("images", []):
        img_path = os.path.join(img_dir, item["file_name"])
        frame    = cv2.imread(img_path)
        if frame is None:
            continue

        gts = [a["bbox"] for a in data.get("annotations", [])
               if a["image_id"] == item["id"]]

        gray    = preprocess(frame)
        results = detect(gray, classifiers, scale=scale, min_neighbors=min_neighbors)

        # Flatten all predictions across all classes
        preds = []
        for boxes in results.values():
            for box in (boxes if len(boxes) > 0 else []):
                preds.append(list(box))

        matched = set()
        for pred in preds:
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if j in matched:
                    continue
                s = iou(pred, gt)
                if s > best_iou:
                    best_iou, best_j = s, j
            if best_iou >= iou_thresh:
                TP += 1
                matched.add(best_j)
            else:
                FP += 1
        FN += len(gts) - len(matched)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\n{'='*45}")
    print(f"  Evaluation Results  (IoU threshold = {iou_thresh})")
    print(f"{'='*45}")
    print(f"  True  Positives (TP) : {TP}")
    print(f"  False Positives (FP) : {FP}")
    print(f"  False Negatives (FN) : {FN}")
    print(f"{'='*45}")
    print(f"  Precision            : {precision:.4f}")
    print(f"  Recall               : {recall:.4f}")
    print(f"  F1-Score             : {f1:.4f}")
    print(f"{'='*45}\n")

    return {"precision": precision, "recall": recall, "f1": f1,
            "TP": TP, "FP": FP, "FN": FN}


if __name__ == "__main__":
    evaluate()
