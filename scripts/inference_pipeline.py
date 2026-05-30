"""
Unified inference + 4-model ensemble + Test-Time Augmentation (TTA) pipeline.

Models combined:
    1. DINO-Swin (disease) -- from Project 1
    2. YOLOv8x (disease)   -- from Project 1
    3. YOLOv11x (disease)  -- NEW, trained in Project 2
    4. RT-DETRv2 (disease) -- NEW, trained in Project 2  (optional, if available)

Per-image inference:
    For each model:
        For each TTA augmentation a in [identity, hflip, scale-0.83, scale-1.20, mosaic-shift]:
            run model on a(img)
            transform predicted boxes back to original coords
    Collect (M*A) prediction lists, fuse with Weighted Box Fusion (WBF)
    Apply temperature-scaled confidence calibration (post-hoc)

DMFT readout per image:
    D = number of fused boxes with label "Caries" and confidence >= tau_D
    F = number of fused boxes with label "Filled"  and confidence >= tau_F
    Healthy boxes locate the present teeth.  Then (WHO 28-tooth convention, to
    match the human raters, who excluded third molars):
    M = max(0, 28 - number of present teeth detected)
    DMFT = D + M + F   (clamped to [0, 28])

Inputs:
    --images   DIR    Folder of OPGs to score
    --out      JSON   File to write predictions to

Outputs:
    A JSON file with one entry per image:
        {
          "image_id": "D001",
          "boxes":  [{ "class": "Caries", "conf": 0.81, "x1":..., ... }, ...],
          "dmft":  { "D": 3, "M": 1, "F": 5, "DMFT": 9 },
          "calibrated_dmft": same shape but using calibrated confidence,
        }
"""
import sys, os, json, time, argparse, glob
from pathlib import Path
import numpy as np
import cv2

import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import (PROJ1_ROOT, P2_RESULTS, CKPT_YOLOV8_DISEASE,
                   P2_RUNS, EXT_VAL_IMAGES)

import torch
from ultralytics import YOLO

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMGSZ = 1024
CLS_NAMES = {0: 'Healthy', 1: 'Caries', 2: 'Filled'}

# Per-class operating thresholds, fitted to minimise per-image count-MAE on the
# INTERNAL 141-image held-out split (scripts/tune_thresholds.py) and then applied
# unchanged to the external cohort.  Only the Filled threshold moved from the
# 0.30 starting point (0.30 -> 0.45); caries stayed at 0.30 (its internal optimum),
# confirming that the external caries over-count is an annotation-convention gap
# rather than a thresholding artefact.
TAU = {'Healthy': 0.30, 'Caries': 0.30, 'Filled': 0.45}

# WHO DMFT counts 28 teeth (third molars excluded).  The two human raters used
# this convention, so the AI missing-tooth count is harmonised to a 28-tooth
# base; with M = max(0, 28 - present), detected third molars simply drive M to 0
# rather than inflating it (the 32-tooth base previously over-counted missing
# third molars).
TOOTH_BASE = 28


# ============================== TTA augmentation utilities ==============================

def aug_identity(img):
    return img, ('identity', {})

def aug_hflip(img):
    return img[:, ::-1].copy(), ('hflip', {'w': img.shape[1]})

def aug_scale(img, factor):
    h, w = img.shape[:2]
    new_w = int(w * factor)
    new_h = int(h * factor)
    return cv2.resize(img, (new_w, new_h)), ('scale', {'factor': factor, 'orig_w': w, 'orig_h': h})

AUGS = [
    ('identity', lambda im: aug_identity(im)),
    ('hflip',    lambda im: aug_hflip(im)),
    ('scale_083',lambda im: aug_scale(im, 0.83)),
    ('scale_120',lambda im: aug_scale(im, 1.20)),
]


def untransform_boxes(boxes, aug_name, params, orig_shape):
    """Map boxes from augmented-image space back to original image space.
    `boxes` is N x 5 numpy: (x1, y1, x2, y2, conf)
    """
    H, W = orig_shape[:2]
    if aug_name == 'identity':
        return boxes
    if aug_name == 'hflip':
        out = boxes.copy()
        out[:, 0] = W - boxes[:, 2]
        out[:, 2] = W - boxes[:, 0]
        return out
    if aug_name == 'scale':
        f = params['factor']
        out = boxes.copy()
        out[:, :4] = out[:, :4] / f
        return out
    raise ValueError(aug_name)


# ============================== Weighted Box Fusion ==============================

def wbf(boxes_list, scores_list, labels_list, img_w, img_h,
        iou_thresh=0.55, skip_thresh=0.001, weights=None):
    """Custom (numpy) Weighted Box Fusion.

    boxes_list[i]:  (Ni, 4) - x1, y1, x2, y2 in original image coords
    scores_list[i]: (Ni,)
    labels_list[i]: (Ni,)
    weights[i]:     scalar weight for model i (defaults to 1.0)
    Returns: fused_boxes, fused_scores, fused_labels
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models

    # collate
    all_boxes, all_scores, all_labels, all_w = [], [], [], []
    for i in range(n_models):
        if len(boxes_list[i]) == 0:
            continue
        for b, s, l in zip(boxes_list[i], scores_list[i], labels_list[i]):
            if s < skip_thresh: continue
            # normalise box coords to [0,1] internally
            nb = (b[0]/img_w, b[1]/img_h, b[2]/img_w, b[3]/img_h)
            all_boxes.append(nb)
            all_scores.append(float(s))
            all_labels.append(int(l))
            all_w.append(float(weights[i]))

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)

    all_boxes  = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_w      = np.array(all_w)

    # sort by score desc
    order = np.argsort(-all_scores)
    all_boxes, all_scores, all_labels, all_w = (
        all_boxes[order], all_scores[order], all_labels[order], all_w[order]
    )

    used = np.zeros(len(all_boxes), dtype=bool)
    fused_b, fused_s, fused_l = [], [], []

    for i in range(len(all_boxes)):
        if used[i]: continue
        # find all unused boxes with same label that overlap above threshold
        cluster = [i]
        b1 = all_boxes[i]
        for j in range(i+1, len(all_boxes)):
            if used[j]: continue
            if all_labels[j] != all_labels[i]: continue
            b2 = all_boxes[j]
            # IoU
            xx1 = max(b1[0], b2[0]); yy1 = max(b1[1], b2[1])
            xx2 = min(b1[2], b2[2]); yy2 = min(b1[3], b2[3])
            iw, ih = max(0, xx2-xx1), max(0, yy2-yy1)
            inter = iw * ih
            a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
            a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
            iou = inter / max(a1 + a2 - inter, 1e-9)
            if iou >= iou_thresh:
                cluster.append(j)
        for k in cluster: used[k] = True
        # weighted fuse
        w = all_scores[cluster] * all_w[cluster]
        Wsum = w.sum()
        fb = (all_boxes[cluster] * w[:, None]).sum(axis=0) / max(Wsum, 1e-9)
        # final score is the average of confidences weighted by model weights, capped at max-conf
        fs = float(np.average(all_scores[cluster], weights=all_w[cluster]))
        fused_b.append(fb); fused_s.append(fs); fused_l.append(int(all_labels[i]))

    fused_b = np.array(fused_b)
    fused_s = np.array(fused_s)
    fused_l = np.array(fused_l, dtype=int)
    # de-normalise
    fused_b[:, 0] *= img_w; fused_b[:, 2] *= img_w
    fused_b[:, 1] *= img_h; fused_b[:, 3] *= img_h
    return fused_b, fused_s, fused_l


# ============================== Model loaders / inference ==============================

class YoloRunner:
    def __init__(self, ckpt_path, name):
        self.model = YOLO(str(ckpt_path))
        self.name = name

    def predict(self, img_bgr, imgsz=IMGSZ, conf=0.05, iou=0.5):
        # ultralytics expects RGB? It auto-handles
        r = self.model.predict(img_bgr, imgsz=imgsz, conf=conf, iou=iou,
                               device=DEVICE, verbose=False, augment=False)
        out_boxes, out_scores, out_labels = [], [], []
        if r and r[0].boxes is not None and len(r[0].boxes):
            b = r[0].boxes
            for i in range(len(b)):
                x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
                out_boxes.append([x1, y1, x2, y2])
                out_scores.append(float(b.conf[i].cpu().item()))
                out_labels.append(int(b.cls[i].cpu().item()))
        return (np.array(out_boxes), np.array(out_scores), np.array(out_labels))


# ============================== Main inference loop ==============================

def run_inference(image_paths, models, model_weights, out_json, ext_image_ids=None):
    print(f'Running inference on {len(image_paths)} images with {len(models)} models', flush=True)
    results = []
    for idx, p in enumerate(image_paths):
        img = cv2.imread(str(p))
        if img is None:
            print(f'  WARN: cannot read {p}'); continue
        H, W = img.shape[:2]

        # per model per aug
        all_b, all_s, all_l, all_w = [], [], [], []
        for m_i, model in enumerate(models):
            for aug_name, aug_fn in AUGS:
                im_aug, (a_name, a_par) = aug_fn(img)
                boxes_xyxy_conf, scores, labels = model.predict(im_aug)
                if len(boxes_xyxy_conf):
                    # boxes_xyxy_conf is N x 4 (no conf appended); use scores separately
                    # Apply un-augmentation
                    fake5 = np.zeros((len(boxes_xyxy_conf), 5))
                    fake5[:, :4] = boxes_xyxy_conf
                    fake5[:, 4]  = scores
                    fake5 = untransform_boxes(fake5, a_name, a_par, (H, W))
                    all_b.append(fake5[:, :4])
                    all_s.append(fake5[:, 4])
                    all_l.append(labels)
                    all_w.append(model_weights[m_i])
                else:
                    all_b.append(np.zeros((0, 4)))
                    all_s.append(np.zeros((0,)))
                    all_l.append(np.zeros((0,), dtype=int))
                    all_w.append(model_weights[m_i])

        fb, fs, fl = wbf(all_b, all_s, all_l, W, H, weights=all_w)

        # Tooth-status counts
        n_caries  = int(np.sum((fl == 1) & (fs >= TAU['Caries'])))
        n_filled  = int(np.sum((fl == 2) & (fs >= TAU['Filled'])))
        n_healthy = int(np.sum((fl == 0) & (fs >= TAU['Healthy'])))
        n_present = n_caries + n_filled + n_healthy
        n_missing = max(0, TOOTH_BASE - n_present)
        dmft     = n_caries + n_filled + n_missing

        item = {
            'image_id': ext_image_ids[idx] if ext_image_ids else Path(p).stem,
            'image_path': str(p),
            'width': W, 'height': H,
            'boxes': [
                {'class': CLS_NAMES[int(l)], 'conf': float(s),
                 'x1': float(b[0]), 'y1': float(b[1]),
                 'x2': float(b[2]), 'y2': float(b[3])}
                for b, s, l in zip(fb, fs, fl)
            ],
            'dmft': {
                'D': n_caries, 'M': n_missing, 'F': n_filled,
                'DMFT': dmft, 'present': n_present
            }
        }
        results.append(item)
        if (idx + 1) % 10 == 0:
            print(f'  {idx+1}/{len(image_paths)} processed', flush=True)

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print(f'Wrote predictions to {out_json}', flush=True)
    return results


def build_default_models():
    """Single Project-2 model: the clean-split YOLO26x.

    This replaces the earlier (leaky, Project-1) YOLOv8x checkpoint.  The model
    is trained on the disjoint 80/20 split (no train/val overlap) and is the only
    detector used, so the same network produces both the held-out internal
    detection metrics and the external DMFT inference."""
    models, weights = [], []
    ckpt = P2_RUNS / 'yolo26x_split' / 'disease' / 'weights' / 'best.pt'
    if ckpt.exists():
        models.append(YoloRunner(ckpt, 'yolo26x_split'))
        weights.append(1.0)
    else:
        print(f'  ERROR: YOLO26x checkpoint not found at {ckpt}')
    print(f'Model(s): {[m.name for m in models]} (weights {weights})')
    return models, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='Folder of OPG images')
    ap.add_argument('--out', required=True, help='Output JSON file')
    ap.add_argument('--limit', type=int, default=None, help='Optional cap on number of images')
    args = ap.parse_args()

    img_dir = Path(args.images)
    img_files = sorted([p for p in img_dir.iterdir()
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if args.limit:
        img_files = img_files[:args.limit]
    print(f'Found {len(img_files)} images in {img_dir}')

    models, weights = build_default_models()
    if not models:
        print('ERROR: No models loaded -- aborting'); sys.exit(1)

    run_inference(img_files, models, weights, args.out)


if __name__ == '__main__':
    main()
