#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

from mmdet.apis import init_detector, inference_detector

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    from PIL import ImageDraw

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from skimage import measure


def _mask_to_polygons(mask, min_area=10.0, epsilon=1.0):
    h, w = mask.shape[:2]
    polygons = []
    if _HAS_CV2:
        mask_u8 = (mask.astype('uint8') * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            if epsilon > 0:
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
            pts = cnt.reshape(-1, 2).astype(float)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1e-3)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1e-3)
            if len(pts) >= 3:
                polygons.append(pts.flatten().tolist())
    else:
        contours = measure.find_contours(mask.astype(float), 0.5)
        for cnt in contours:
            if len(cnt) < 3:
                continue
            pts = cnt[:, ::-1]
            if epsilon > 0 and len(pts) > 4:
                step = max(1, int(np.ceil(epsilon)))
                pts = pts[::step]
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1e-3)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1e-3)
            if len(pts) >= 3:
                polygons.append(pts.flatten().tolist())
    return polygons


def _result_to_instances(result, score_thr):
    """
    Handles MMDetection 2.x tuple output and 3.x DetDataSample output.
    Returns: bboxes, scores, labels, masks
    """
    # --- MMDet 3.x ---
    if hasattr(result, "pred_instances"):
        pred = result.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        masks = None
        if hasattr(pred, "masks") and pred.masks is not None:
            try:
                masks = pred.masks.to_ndarray().astype(bool)
            except Exception:
                masks = np.array(pred.masks, dtype=bool)
        keep = scores >= score_thr
        return (
            bboxes[keep],
            scores[keep],
            labels[keep],
            masks[keep] if masks is not None else None,
        )

    # --- MMDet 2.x style tuple/list ---
    if isinstance(result, (list, tuple)):
        if len(result) == 2:
            bbox_results, segm_results = result
            if isinstance(segm_results, tuple):
                segm_results = segm_results[0]
        else:
            bbox_results, segm_results = result, None

        all_bboxes, all_scores, all_labels, all_masks = [], [], [], []
        for cls_id, cls_dets in enumerate(bbox_results):
            if cls_dets is None or len(cls_dets) == 0:
                continue
            cls_dets = np.atleast_2d(cls_dets)
            if cls_dets.shape[1] < 5:
                continue
            keep = cls_dets[:, 4] >= score_thr
            cls_dets = cls_dets[keep]
            if cls_dets.shape[0] == 0:
                continue

            all_bboxes.append(cls_dets[:, :4])
            all_scores.append(cls_dets[:, 4])
            all_labels.append(np.full(cls_dets.shape[0], cls_id))

            if segm_results is not None and cls_id < len(segm_results) and segm_results[cls_id] is not None:
                seg_list_full = segm_results[cls_id]
                seg_list = [seg_list_full[i] for i, k in enumerate(keep) if k]
                masks_np = []
                for m in seg_list:
                    if hasattr(m, "to_ndarray"):
                        masks_np.append(m.to_ndarray().astype(bool))
                    elif isinstance(m, dict) and "counts" in m:
                        try:
                            from pycocotools import mask as mask_utils
                            masks_np.append(mask_utils.decode(m).astype(bool))
                        except Exception:
                            continue
                    else:
                        masks_np.append(np.array(m, dtype=bool))
                if masks_np:
                    all_masks.extend(masks_np)

        if not all_bboxes:
            return (
                np.zeros((0, 4)),
                np.zeros((0,)),
                np.zeros((0,), dtype=np.int32),
                None,
            )

        bboxes = np.concatenate(all_bboxes)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        masks = np.stack(all_masks) if all_masks else None
        return bboxes, scores, labels, masks

    raise ValueError("Unsupported result format")


  
def visualize_and_export(image_path, bboxes, scores, labels, masks, class_names, output_path):
    img = cv2.imread(str(image_path))
    overlay = img.copy()
    out_entry = {
        "file_name": image_path.name,
        "width": img.shape[1],
        "height": img.shape[0],
        "detections": []
    }

    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i].astype(int)
        label_id = int(labels[i])
        if class_names and 0 <= label_id < len(class_names):
            label = class_names[label_id]
        else:
            label = f"class_{label_id}"
        score = scores[i]
        color = tuple(int(c) for c in np.random.default_rng(label_id + 42).integers(50, 255, size=3))

        if masks is not None and masks.shape[0] > i:
            poly = _mask_to_polygons(masks[i])
            if poly:
                mask_img = np.zeros_like(img, dtype=np.uint8)
                for p in poly:
                    pts = np.array(p).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask_img, [pts], color)
                overlay = cv2.addWeighted(overlay, 1.0, mask_img, 0.4, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{label} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        out_entry["detections"].append({
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(score),
            "category_id": int(labels[i]),
            "category_name": label,
            "segmentation": _mask_to_polygons(masks[i]) if masks is not None and masks.shape[0] > i else []
        })

    cv2.imwrite(str(output_path / image_path.name), overlay)
    return out_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", required=True, help="Single image or folder path")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file")
    parser.add_argument("--out-dir", default="output", help="Output folder")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    args = parser.parse_args()

    input_path = Path(args.img_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = init_detector(args.config, args.checkpoint, device=args.device)

    class_names = None
    if hasattr(model, "dataset_meta"):
        class_names = model.dataset_meta.get("classes", None)
    elif hasattr(model, "CLASSES"):
        class_names = model.CLASSES

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = [p for p in input_path.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    json_results = []
    for img_path in image_paths:
        result = inference_detector(model, str(img_path))
        bboxes, scores, labels, masks = _result_to_instances(result, args.score_thr)

        print(f"Image: {img_path.name}, Num BBoxes: {len(bboxes)}, Masks: {masks is not None}, Mask Shape: {masks.shape if masks is not None else 'N/A'}")

        entry = visualize_and_export(img_path, bboxes, scores, labels, masks, class_names, output_dir)
        json_results.append(entry)

    with open(output_dir / "detections_polygons.json", "w") as f:
        json.dump({"images": json_results}, f, indent=2)

    print(f"\n✅ Saved visualized images and detections_polygons.json to: {output_dir}")


if __name__ == "__main__":
    main()
