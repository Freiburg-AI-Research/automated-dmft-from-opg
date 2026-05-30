"""
Decide TTA keep/drop on INTERNAL data only (no external-test leakage).

For each internal held-out image, compute per-class predicted counts WITH the
4-augmentation TTA + WBF and WITHOUT TTA (identity only), at the tuned operating
thresholds, and compare per-class count-MAE against the DENTEX ground-truth
counts.  If TTA lowers internal count error, the a-priori TTA design is justified.

Outputs: results/internal_tta_check.json
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
TAU_BY_ID = {0: IP.TAU['Healthy'], 1: IP.TAU['Caries'], 2: IP.TAU['Filled']}
IDENTITY = [('identity', lambda im: IP.aug_identity(im))]


def fused(models, weights, img, augs):
    H, W = img.shape[:2]
    ab, as_, al, aw = [], [], [], []
    for mi, model in enumerate(models):
        for an, fn in augs:
            im2, (nm, pr) = fn(img)
            b, s, l = model.predict(im2)
            if len(b):
                f5 = np.zeros((len(b), 5)); f5[:, :4] = b; f5[:, 4] = s
                f5 = IP.untransform_boxes(f5, nm, pr, (H, W))
                ab.append(f5[:, :4]); as_.append(f5[:, 4]); al.append(l)
            else:
                ab.append(np.zeros((0, 4))); as_.append(np.zeros((0,))); al.append(np.zeros((0,), int))
            aw.append(weights[mi])
    fb, fs, fl = IP.wbf(ab, as_, al, W, H, weights=aw)
    return fl, fs


def gt_counts(stem):
    f = LBL_DIR / (stem + '.txt'); c = {0: 0, 1: 0, 2: 0}
    if f.exists():
        for ln in f.read_text().splitlines():
            ln = ln.strip()
            if ln:
                k = int(float(ln.split()[0]))
                if k in c: c[k] += 1
    return c


def counts(fl, fs):
    return {c: int(np.sum((fl == c) & (fs >= TAU_BY_ID[c]))) for c in CLASSES}


def main():
    models, weights = IP.build_default_models()
    paths = [Path(p.strip()) for p in VAL_LIST.read_text().splitlines() if p.strip()]
    err_tta = {c: [] for c in CLASSES}; err_no = {c: [] for c in CLASSES}
    dmft_tta, dmft_no = [], []  # internal "DMFT-like" = D + F + (28-present) using box counts
    for i, p in enumerate(paths):
        img = cv2.imread(str(p))
        if img is None: continue
        g = gt_counts(p.stem)
        for tag, augs, store in (('tta', IP.AUGS, err_tta), ('no', IDENTITY, err_no)):
            fl, fs = fused(models, weights, img, augs)
            c = counts(fl, fs)
            for k in CLASSES: store[k].append(abs(c[k] - g[k]))
        if (i + 1) % 30 == 0: print(f'  {i+1}/{len(paths)}')
    out = {'with_tta_count_MAE': {CLASSES[k]: round(float(np.mean(err_tta[k])), 3) for k in CLASSES},
           'without_tta_count_MAE': {CLASSES[k]: round(float(np.mean(err_no[k])), 3) for k in CLASSES}}
    out['with_tta_total'] = round(sum(out['with_tta_count_MAE'].values()), 3)
    out['without_tta_total'] = round(sum(out['without_tta_count_MAE'].values()), 3)
    with open(P2_RESULTS / 'internal_tta_check.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
