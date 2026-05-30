"""
Train YOLO26x on the CLEAN 80/20 DENTEX disease_all split (no leakage).

YOLO26x (Ultralytics, 2025) is the newest-generation, end-to-end (NMS-free)
detector.  Chosen because (a) we retrain from scratch anyway, (b) it is fully
independent of the Project-1 YOLOv8x checkpoint, and (c) it is the strongest
"better than the first paper" backbone.  COCO-pretrained weights are used for
transfer learning (essential for a 705-image fine-tune).

Single model -> produces BOTH the held-out internal detection metrics and the
external (Düsseldorf) DMFT inference.  Hyperparameters match the manuscript:
AdamW, lr0 1e-3, cosine schedule, imgsz 1024, mosaic, early-stopping patience 20.

GPU note: a foreign process occupies ~13.5 GB of the 24 GB RTX 3090, so we use a
FIXED batch=4 (auto-batch would assume 60% of TOTAL memory and OOM).

Outputs:
  runs/yolo26x_split/disease/weights/best.pt
  runs/yolo26x_split/disease/results.csv
"""
import sys, time
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import PROJ2_ROOT, P2_RUNS

from ultralytics import YOLO

DATA_YAML   = str(PROJ2_ROOT / 'configs' / 'dentex_disease_split.yaml')
STARTER     = 'yolo26x.pt'                              # COCO-pretrained YOLO26x (auto-cached)
PROJECT_DIR = str(P2_RUNS / 'yolo26x_split')
NAME        = 'disease'


def main():
    print('Starting CLEAN YOLO26x training on DENTEX disease_all (80/20 split)', flush=True)
    print(f'  Starter: {STARTER}', flush=True)
    print(f'  Data:    {DATA_YAML}', flush=True)
    print(f'  Out:     {PROJECT_DIR}/{NAME}', flush=True)

    model = YOLO(STARTER)

    t0 = time.time()
    model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=1024,
        batch=2,            # FIXED — shared GPU; batch=4 left only 433 MiB free, batch=2 leaves ~4.3 GB headroom
        device=0,
        workers=4,          # be polite to the co-resident process
        project=PROJECT_DIR,
        name=NAME,
        exist_ok=True,
        seed=42,
        amp=True,
        # augmentation (same recipe as the original training)
        hsv_h=0.0, hsv_s=0.4, hsv_v=0.3,
        translate=0.05, scale=0.4, fliplr=0.5, flipud=0.0,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        close_mosaic=10,
        # optimisation
        optimizer='AdamW',
        lr0=1e-3, lrf=0.01, weight_decay=5e-4,
        cos_lr=True, warmup_epochs=3.0,
        # validation / checkpoints
        val=True, patience=20, save_period=25,
        plots=True, verbose=True,
    )
    dt = time.time() - t0
    print(f'Training finished in {dt/3600:.2f} h', flush=True)

    best = f'{PROJECT_DIR}/{NAME}/weights/best.pt'
    print(f'Best.pt at: {best}', flush=True)
    m = YOLO(best)
    metrics = m.val(data=DATA_YAML, imgsz=1024, batch=4, device=0, verbose=False)
    print('Held-out val mAP50-95:', float(metrics.box.map), flush=True)
    print('DONE_TRAINING', flush=True)


if __name__ == '__main__':
    main()
