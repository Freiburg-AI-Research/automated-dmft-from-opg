"""
Per-class confidence-threshold calibration on the INTERNAL held-out split.

The DMFT readout previously used a single arbitrary operating point (tau = 0.30
for every class), which over-detects caries and inflates the external DMFT.
Here we choose, for each class independently, the confidence threshold that
minimises the per-image |predicted count - ground-truth count| (count-MAE) on
the disjoint 141-image internal validation split.  Thresholds are therefore
fitted ONLY on internal data and then applied unchanged to the external cohort
(no external-test tuning).

Outputs:
  results/tuned_thresholds.json
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import P2_RESULTS, DENTEX_YOLO_DISEASE_ALL
sys.path.insert(0, r'.\scripts')
import inference_pipeline as IP

VAL_LIST = Path(DENTEX_YOLO_DISEASE_ALL) / 'val_split.txt'
LBL_DIR  = Path(DENTEX_YOLO_DISEASE_ALL) / 'labels' / 'train2017'
CLASSES  = {0: 'Healthy', 1: 'Caries', 2: 'Filled'}
GRID     = np.round(np.arange(0.05, 0.96, 0.05), 2)


def fused_scored_boxes(models, weights, img):
    H, W = img.shape[:2]
    all_b, all_s, all_l, all_w = [], [], [], []
    for m_i, model in enumerate(models):
        for a_name, aug_fn in IP.AUGS:
            im_aug, (an, ap) = aug_fn(img)
            boxes, scores, labels = model.predict(im_aug)
            if len(boxes):
                f5 = np.zeros((len(boxes), 5)); f5[:, :4] = boxes; f5[:, 4] = scores
                f5 = IP.untransform_boxes(f5, an, ap, (H, W))
                all_b.append(f5[:, :4]); all_s.append(f5[:, 4]); all_l.append(labels)
            else:
                all_b.append(np.zeros((0, 4))); all_s.append(np.zeros((0,)))
                all_l.append(np.zeros((0,), dtype=int))
            all_w.append(weights[m_i])
    fb, fs, fl = IP.wbf(all_b, all_s, all_l, W, H, weights=all_w)
    return fl, fs  # labels, scores of fused boxes


def gt_counts(stem):
    f = LBL_DIR / (stem + '.txt')
    c = {0: 0, 1: 0, 2: 0}
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(float(line.split()[0]))
            if cls in c:
                c[cls] += 1
    return c


def main():
    models, weights = IP.build_default_models()
    assert models, 'no model'
    paths = [Path(p.strip()) for p in VAL_LIST.read_text().splitlines() if p.strip()]
    print(f'Collecting fused boxes on {len(paths)} internal-val images...')

    # per image: dict class-> list of scores ; and GT counts
    pred_scores = []  # list over images: {cls: np.array(scores)}
    gts = []          # list over images: {cls: count}
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        fl, fs = fused_scored_boxes(models, weights, img)
        pred_scores.append({c: fs[fl == c] for c in CLASSES})
        gts.append(gt_counts(p.stem))
        if (i + 1) % 30 == 0:
            print(f'  {i+1}/{len(paths)}')

    tuned = {}
    curves = {}
    for c, name in CLASSES.items():
        gt_c = np.array([g[c] for g in gts], dtype=float)
        best_tau, best_mae = 0.30, 1e9
        curve = {}
        for tau in GRID:
            pred_c = np.array([int(np.sum(ps[c] >= tau)) for ps in pred_scores], dtype=float)
            mae = float(np.mean(np.abs(pred_c - gt_c)))
            curve[float(tau)] = round(mae, 4)
            if mae < best_mae:
                best_mae, best_tau = mae, float(tau)
        tuned[name] = best_tau
        curves[name] = curve
        # also report MAE at the old default 0.30 for transparency
        pred_30 = np.array([int(np.sum(ps[c] >= 0.30)) for ps in pred_scores], dtype=float)
        mae_30 = float(np.mean(np.abs(pred_30 - gt_c)))
        print(f'{name:8s}: tau*={best_tau:.2f} count-MAE={best_mae:.3f}  (default 0.30 -> {mae_30:.3f})')

    out = {'tuned_thresholds': tuned, 'grid_mae': curves,
           'note': 'thresholds minimise per-image count-MAE on the internal 141-image held-out split'}
    with open(P2_RESULTS / 'tuned_thresholds.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('\nTuned thresholds:', tuned)
    print(f'Wrote {P2_RESULTS / "tuned_thresholds.json"}')


if __name__ == '__main__':
    main()
