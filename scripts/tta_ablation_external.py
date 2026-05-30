"""
TTA ablation on the CLINICAL endpoint (external DMFT agreement).

Runs the single YOLO26x model over the 100 Düsseldorf OPGs twice:
  - full pipeline : 5 geometric augmentations + Weighted Box Fusion (WBF)
  - no TTA        : identity augmentation only
and reports the effect on DMFT agreement with the adjudicated rater score
(MAE, ICC, Bland-Altman bias).  This replaces the previous ablation, which
measured COCO AP on leaked training images.

Outputs:
  results/tta_ablation_external.json
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import P2_RESULTS, EXT_VAL_IMAGES
sys.path.insert(0, r'.\scripts')
import inference_pipeline as IP
from statistics import icc_2_1, mae, bland_altman

IDENTITY_ONLY = [('identity', lambda im: IP.aug_identity(im))]


def score(models, weights, img, augs):
    H, W = img.shape[:2]
    all_b, all_s, all_l, all_w = [], [], [], []
    for m_i, model in enumerate(models):
        for a_name, aug_fn in augs:
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
    nD = int(np.sum((fl == 1) & (fs >= IP.TAU['Caries'])))
    nF = int(np.sum((fl == 2) & (fs >= IP.TAU['Filled'])))
    nH = int(np.sum((fl == 0) & (fs >= IP.TAU['Healthy'])))
    present = nD + nF + nH
    M = max(0, IP.TOOTH_BASE - present)
    return nD + nF + M  # DMFT


def main():
    models, weights = IP.build_default_models()
    assert models, 'no model'
    gt = pd.read_csv(P2_RESULTS / 'external_dmft_combined.csv')[
        ['image_id', 'A_DMFT', 'B_DMFT', 'GT_DMFT']].set_index('image_id')

    imgs = sorted(p for p in Path(EXT_VAL_IMAGES).iterdir()
                  if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    rows = []
    for p in imgs:
        iid = p.stem
        if iid not in gt.index:
            continue
        img = cv2.imread(str(p))
        rows.append({
            'image_id': iid,
            'tta':   score(models, weights, img, IP.AUGS),
            'notta': score(models, weights, img, IDENTITY_ONLY),
            'GT':    int(gt.loc[iid, 'GT_DMFT']),
        })
    df = pd.DataFrame(rows)
    out = {
        'n': int(len(df)),
        'with_tta':    {'mae': mae(df['GT'], df['tta']),
                        'icc_2_1': icc_2_1(df['GT'], df['tta']),
                        'bias': bland_altman(df['GT'], df['tta'])['bias']},
        'without_tta': {'mae': mae(df['GT'], df['notta']),
                        'icc_2_1': icc_2_1(df['GT'], df['notta']),
                        'bias': bland_altman(df['GT'], df['notta'])['bias']},
    }
    out['delta_mae_tta_minus_notta'] = out['with_tta']['mae'] - out['without_tta']['mae']
    out['delta_icc_tta_minus_notta'] = out['with_tta']['icc_2_1'] - out['without_tta']['icc_2_1']
    with open(P2_RESULTS / 'tta_ablation_external.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
