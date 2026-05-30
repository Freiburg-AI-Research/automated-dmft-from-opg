"""
Held-out internal detection metrics for the CLEAN-split YOLO26x model.

Unlike the previous internal_metrics.py (which evaluated on val2017 == a copy of
train2017 -> data leakage), this evaluates on the disjoint 141-image validation
split defined in dentex_disease_split.yaml.  Reported per-class AP/AP50/AP75 and
precision/recall are therefore genuine held-out estimates.

TTA is deliberately NOT applied here: detection AP reflects the bare model, and
the effect of test-time augmentation is reported where it matters clinically -
on the external DMFT agreement (see tta_ablation_external.py).

Outputs:
  results/internal_metrics_clean.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import PROJ2_ROOT, P2_RESULTS, P2_RUNS
from ultralytics import YOLO

IMGSZ = 1024
DATA_YAML = str(PROJ2_ROOT / 'configs' / 'dentex_disease_split.yaml')
CKPT = P2_RUNS / 'yolo26x_split' / 'disease' / 'weights' / 'best.pt'


def main():
    assert CKPT.exists(), f'checkpoint not found: {CKPT}'
    print(f'Loading {CKPT}')
    model = YOLO(str(CKPT))

    print('=== Held-out validation (clean split, no TTA) ===')
    res = model.val(data=DATA_YAML, imgsz=IMGSZ, batch=2, device=0,
                    save_json=False, plots=False, verbose=False, augment=False)
    names = ['Healthy', 'Caries', 'Filled']
    metrics = {
        'split': 'clean_80_20_val (n=141, disjoint from train)',
        'AP':   float(res.box.map),
        'AP50': float(res.box.map50),
        'AP75': float(res.box.map75),
        'precision': float(res.box.mp),
        'recall':    float(res.box.mr),
        'per_class_AP':  {n: float(res.box.maps[i]) for i, n in enumerate(names)},
    }
    # per-class precision/recall if available
    try:
        metrics['per_class_precision'] = {n: float(res.box.p[i]) for i, n in enumerate(names)}
        metrics['per_class_recall']    = {n: float(res.box.r[i]) for i, n in enumerate(names)}
    except Exception as e:
        print('  (per-class P/R unavailable:', e, ')')

    # per-image timing on the held-out split
    val_imgs = (PROJ2_ROOT.parent / 'dentex_dataset' / 'yolo' / 'disease_all' / 'images' / 'train2017')
    import random
    files = sorted(val_imgs.iterdir())[:20]
    _ = model.predict(str(files[0]), imgsz=IMGSZ, device=0, verbose=False)  # warmup
    t = []
    for p in files:
        t0 = time.time()
        _ = model.predict(str(p), imgsz=IMGSZ, device=0, verbose=False, augment=False)
        t.append(time.time() - t0)
    metrics['timing'] = {
        'per_image_mean_ms': round(1000 * float(np.mean(t)), 1),
        'per_image_p95_ms':  round(1000 * float(np.percentile(t, 95)), 1),
        'n_measured': len(files),
    }

    P2_RESULTS.mkdir(parents=True, exist_ok=True)
    out = P2_RESULTS / 'internal_metrics_clean.json'
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
