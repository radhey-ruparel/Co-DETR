#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
import math

import numpy as np
from PIL import Image

from mmdet.apis import init_detector, inference_detector

# --- polygon helpers ---
# Prefer OpenCV (fast), fall back to scikit-image if cv2 isn't available
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from skimage import measure  # find_contours fallback


def _mask_to_polygons(mask_bool, min_area=10.0, epsilon=1.0):
    """
    Convert a HxW boolean mask to a list of COCO polygons.
    - min_area: drop tiny fragments (px^2)
    - epsilon: simplification tolerance (px) for approxPolyDP; 0 = no simplification
    Returns: List[List[float]] of flattened [x1,y1, x2,y2, ..., xn,yn]
    """
    h, w = mask_bool.shape[:2]
    polygons = []

    if _HAS_CV2:
        mask_u8 = (mask_bool.astype('uint8') * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if epsilon > 0:
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
            pts = cnt.reshape(-1, 2).astype(float)
            # clip to image bounds
            pts[:, 0] = pts[:, 0].clip(0, w - 1e-3)
            pts[:, 1] = pts[:, 1].clip(0, h - 1e-3)
            if len(pts) >= 3:
                polygons.append(pts.flatten().tolist())
    else:
        # skimage fallback
        contours = measure.find_contours(mask_bool.astype(float), 0.5)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            # cnt is (row, col) -> (y, x); convert to (x, y)
            pts = cnt[:, ::-1]
            # naive decimation as a simplification fallback
            if epsilon > 0 and len(pts) > 4:
                step = max(1, int(math.ceil(epsilon)))
                pts = pts[::step]
            pts[:, 0] = pts[:, 0].clip(0, w - 1e-3)
            pts[:, 1] = pts[:, 1].clip(0, h - 1e-3)
            if len(pts) >= 3:
                polygons.append(pts.flatten().tolist())

    return polygons


# ---------- MMDetection output adapters ----------

def _result_to_instances_mmdet3(result, score_thr):
    """
    MMDetection >= 3.x:
    result is DetDataSample with .pred_instances {bboxes, scores, labels, (optional) masks}
    """
    pred = result.pred_instances
    # bboxes: Tensor(N,4), scores: Tensor(N,), labels: Tensor(N,)
    to_np = lambda x: x.numpy() if hasattr(x, "numpy") else np.array(x)
    bboxes = to_np(pred.bboxes)
    scores = to_np(pred.scores)
    labels = to_np(pred.labels)

    masks = None
    if hasattr(pred, "masks") and pred.masks is not None:
        # BitmapMasks has to_ndarray(); otherwise try np.array()
        try:
            masks = pred.masks.to_ndarray().astype(bool)
        except Exception:
            masks = np.array(pred.masks, dtype=bool)

    keep = scores >= score_thr
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    if masks is not None:
        masks = masks[keep]
    return bboxes, scores, labels, masks


def _result_to_instances_mmdet2(result, score_thr):
    """
    MMDetection 2.x:
    result may be (bbox_results, segm_results) or bbox_results only.
    segm_results itself can be a tuple (segm_results, cls_agnostic).
    """
    if isinstance(result, tuple):
        bbox_results = result[0]
        segm_results = result[1] if len(result) > 1 else None
        if isinstance(segm_results, tuple):
            segm_results = segm_results[0]
    else:
        bbox_results, segm_results = result, None

    all_bboxes, all_scores, all_labels, all_masks = [], [], [], []
    num_classes = len(bbox_results) if isinstance(bbox_results, (list, tuple)) else 0

    for cls_id in range(num_classes):
        b = bbox_results[cls_id]
        if b is None or len(b) == 0:
            continue
        b = np.asarray(b)  # (N,5): x1,y1,x2,y2,score
        keep = b[:, 4] >= score_thr
        b = b[keep]
        if b.size == 0:
            continue

        all_bboxes.append(b[:, :4])
        all_scores.append(b[:, 4])
        all_labels.append(np.full((b.shape[0],), cls_id, dtype=np.int32))

        # masks (optional per class)
        class_masks = None
        if segm_results is not None and isinstance(segm_results, (list, tuple)):
            if cls_id < len(segm_results) and segm_results[cls_id] is not None:
                seg_list_full = segm_results[cls_id]
                # align with 'keep'
                seg_list = [seg_list_full[i] for i, k in enumerate(keep) if k]
                masks_np = []
                for m in seg_list:
                    # various MMDet 2.x mask shapes
                    if hasattr(m, "to_ndarray"):
                        masks_np.append(m.to_ndarray().astype(bool))
                    elif isinstance(m, dict) and "counts" in m:
                        # RLE provided; decode via pycocotools if available, else skip
                        try:
                            from pycocotools import mask as mask_utils
                            masks_np.append(mask_utils.decode(m).astype(bool))
                        except Exception:
                            # no decoder available; skip this mask
                            continue
                    else:
                        masks_np.append(np.array(m, dtype=bool))
                if len(masks_np) > 0:
                    class_masks = np.stack(masks_np, axis=0)
        all_masks.append(class_masks)

    if len(all_bboxes) == 0:
        return (np.zeros((0, 4)),
                np.zeros((0,)),
                np.zeros((0,), dtype=np.int32),
                None)

    bboxes = np.concatenate(all_bboxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    non_empty = [m for m in all_masks if m is not None]
    masks = np.concatenate(non_empty, axis=0) if len(non_empty) > 0 else None
    return bboxes, scores, labels, masks


# -------------------- Main --------------------

def main():
    parser = argparse.ArgumentParser(description="Batch export detections (bboxes + polygon segmentations) to JSON")
    parser.add_argument("--img-dir", required=True, help="Folder containing images")
    parser.add_argument("--config", required=True, help="MMDetection config (.py)")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pth)")
    parser.add_argument("--out-json", required=True, help="Output JSON path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Detection score threshold")
    parser.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.JPG,.PNG", help="Comma-separated extensions")
    parser.add_argument("--min-area", type=float, default=10.0, help="Min polygon area (px^2)")
    parser.add_argument("--epsilon", type=float, default=1.5, help="Polygon simplification tolerance (px)")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    # init model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # Try to get class names
    class_names = None
    # MMDet 3.x usually has dataset_meta
    if hasattr(model, "dataset_meta") and isinstance(model.dataset_meta, dict):
        class_names = model.dataset_meta.get("classes", None)
    # MMDet 2.x uses model.CLASSES
    if class_names is None and hasattr(model, "CLASSES"):
        class_names = getattr(model, "CLASSES")

    # collect images
    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])
    img_paths = sorted([p for p in img_dir.iterdir() if str(p).endswith(exts)])

    os.makedirs(Path(args.out_json).parent, exist_ok=True)

    results_all = []
    for img_path in img_paths:
        with Image.open(img_path) as im:
            w, h = im.size

        out = inference_detector(model, str(img_path))

        # MMDet 3.x vs 2.x dispatch
        if hasattr(out, "pred_instances"):
            bboxes, scores, labels, masks = _result_to_instances_mmdet3(out, args.score_thr)
        else:
            bboxes, scores, labels, masks = _result_to_instances_mmdet2(out, args.score_thr)

        image_entry = {
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "detections": [],
        }

        N = bboxes.shape[0]
        for i in range(N):
            det = {
                "bbox": [float(x) for x in bboxes[i].tolist()],  # [x1,y1,x2,y2]
                "score": float(scores[i]),
                "category_id": int(labels[i]),
            }
            if class_names is not None:
                lid = int(labels[i])
                if 0 <= lid < len(class_names):
                    det["category_name"] = class_names[lid]

            # polygon segmentation (if available)
            if masks is not None and masks.shape[0] > i:
                m = masks[i]
                if m.ndim == 3:  # (1,H,W) -> (H,W)
                    m = m[0]
                polys = _mask_to_polygons(m.astype(bool),
                                          min_area=args.min_area,
                                          epsilon=args.epsilon)
                det["segmentation"] = polys
                det["iscrowd"] = 0

            image_entry["detections"].append(det)

        results_all.append(image_entry)
        print(f"Processed {img_path}: {len(image_entry['detections'])} detections")

    with open(args.out_json, "w") as f:
        json.dump({"images": results_all}, f, indent=2)
    print(f"\nSaved JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
