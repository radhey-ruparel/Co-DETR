#!/usr/bin/env python3
import json, argparse, os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_images(ann_path, img_prefix):
    coco = COCO(ann_path)
    imgs = coco.loadImgs(coco.getImgIds())
    return coco, [Path(img_prefix) / im['file_name'] for im in imgs]

def to_coco_dets(results, filename, img_id, bird_label_index):
    """results can be MMDet 2.x (list of per-class bboxes) or 3.x DetDataSample."""
    dets = []
    # --- MMDet 3.x ---
    if hasattr(results, "pred_instances"):
        pred = results.pred_instances
        bboxes = pred.bboxes.cpu().numpy()
        scores = pred.scores.cpu().numpy()
        labels = pred.labels.cpu().numpy()
        for b, s, l in zip(bboxes, scores, labels):
            if int(l) != bird_label_index:  # keep only bird
                continue
            x1,y1,x2,y2 = b
            dets.append({
                "image_id": int(img_id),
                "category_id": 1,          # single-class target JSON uses id=1
                "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                "score": float(s),
            })
        return dets

    # --- MMDet 2.x: list of per-class arrays ---
    if isinstance(results, (list, tuple)):
        # results is a list where results[c] is Nx5 for class c: [x1,y1,x2,y2,score]
        if bird_label_index < len(results):
            arr = np.asarray(results[bird_label_index])
            for b in arr:
                x1,y1,x2,y2,score = b.tolist()
                dets.append({
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "score": float(score),
                })
        return dets

    raise ValueError("Unsupported result format")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ann-file", required=True)   # single-class GT (id=1,name='bird') or your fixed one
    ap.add_argument("--img-prefix", required=True)
    ap.add_argument("--out", default="bird_dets.json")
    ap.add_argument("--device", default="cuda")
    # For the COCO model, 'bird' class index is 14 (0-based) in the 80-class list
    ap.add_argument("--bird-index", type=int, default=14)
    args = ap.parse_args()

    # build model
    model = init_detector(args.config, args.checkpoint, device=args.device)

    coco, img_paths = load_images(args.ann_file, args.img_prefix)
    imgid_by_name = {img['file_name']: img['id'] for img in coco.dataset['images']}

    all_dets = []
    for p in tqdm(img_paths):
        res = inference_detector(model, str(p))
        img_id = imgid_by_name[p.name]
        all_dets.extend(to_coco_dets(res, p.name, img_id, args.bird_index))

    # write detections
    with open(args.out, "w") as f:
        json.dump(all_dets, f)
    print("Wrote:", args.out, " (#dets:", len(all_dets), ")")

    # evaluate with COCOeval
    coco_dt = coco.loadRes(args.out) if len(all_dets) else coco.loadRes([])
    E = COCOeval(coco, coco_dt, iouType='bbox')
    E.params.catIds = [1]   # single class id in your GT json
    E.evaluate(); E.accumulate(); E.summarize()

if __name__ == "__main__":
    main()
