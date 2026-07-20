#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

import numpy as np

# Prefer OpenCV for speed/blending; fall back to PIL if not available
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    from PIL import Image, ImageDraw, ImageFont


def _color_from_id(k: int):
    """Deterministic distinct-ish color per class/id."""
    rng = np.random.default_rng(k + 12345)
    return tuple(int(x) for x in rng.integers(50, 255, size=3))  # avoid too-dark


def _draw_cv2(img_bgr, dets, alpha=0.4, draw_bbox=False, thickness=2):
    overlay = img_bgr.copy()
    h, w = img_bgr.shape[:2]

    for det in dets:
        score = det.get("score", 0.0)
        cls_id = int(det.get("category_id", -1))
        cls_name = det.get("category_name", str(cls_id))
        color = _color_from_id(cls_id)

        # Draw polygons
        seg = det.get("segmentation", [])
        for poly in seg:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            # clip to image just in case
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            pts = pts.astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)

        # Blend overlay for filled masks once per det loop
        # (All polys of this object already painted on overlay)
        # We'll blend after all objects to keep consistent alpha.

        # Draw bbox (optional)
        if draw_bbox and "bbox" in det:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)

        # Put label
        if draw_bbox and "bbox" in det:
            x1, y1, _, _ = [int(v) for v in det["bbox"]]
            label = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # One-time alpha blend of all filled polygons
    img_bgr[:] = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    return img_bgr


def _draw_pil(img_pil, dets, alpha=0.4, draw_bbox=False, thickness=2):
    # Fallback if OpenCV isn't present
    base = img_pil.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size

    for det in dets:
        score = det.get("score", 0.0)
        cls_id = int(det.get("category_id", -1))
        cls_name = det.get("category_name", str(cls_id))
        color = _color_from_id(cls_id)
        fill = (*color, int(255 * alpha))

        seg = det.get("segmentation", [])
        for poly in seg:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            draw.polygon(pts.flatten().tolist(), fill=fill)

        if draw_bbox and "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=tuple(color) + (255,), width=thickness)
            # simple label box
            label = f"{cls_name} {score:.2f}"
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=(0, 0, 0, 255))

    return Image.alpha_composite(base, overlay).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description="Overlay polygon segmentations (and bboxes) from JSON onto images")
    ap.add_argument("--json", required=True, help="detections_polygons.json path")
    ap.add_argument("--img-root", required=True, help="folder where the images live")
    ap.add_argument("--out-dir", required=True, help="where to save visualizations")
    ap.add_argument("--alpha", type=float, default=0.4, help="mask transparency 0..1")
    ap.add_argument("--draw-bbox", action="store_true", help="also draw bounding boxes and labels")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.json, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        print("No images in JSON.")
        return

    for item in images:
        fname = item["file_name"]
        dets = item.get("detections", [])
        img_path = img_root / fname
        if not img_path.exists():
            print(f"[WARN] Missing image file: {img_path}")
            continue

        if _HAS_CV2:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[WARN] Cannot read image: {img_path}")
                continue
            vis = _draw_cv2(img_bgr, dets, alpha=args.alpha, draw_bbox=args.draw_bbox)
            cv2.imwrite(str(out_dir / fname), vis)
        else:
            img_pil = Image.open(img_path).convert("RGB")
            vis = _draw_pil(img_pil, dets, alpha=args.alpha, draw_bbox=args.draw_bbox)
            vis.save(out_dir / fname)

        print(f"Saved -> {out_dir / fname}")


if __name__ == "__main__":
    main()
