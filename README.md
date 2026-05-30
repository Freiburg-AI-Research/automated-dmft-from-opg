# Automated DMFT determination from panoramic radiographs

End-to-end research pipeline for the fully automated determination of the
Decayed-Missing-Filled-Teeth (DMFT) index from a single panoramic radiograph
(orthopantomogram, OPG), using a single-stage deep-learning detector with
test-time augmentation (TTA) and Weighted-Box-Fusion (WBF) aggregation, and an
external validation against two independent dentists.

This repository contains the **complete training, inference, evaluation and
figure-generation code** that backs our manuscript. **No patient data, no
external annotations and no Düsseldorf cohort imagery are included** — the
repository is code-only and reproducible against the public
[DENTEX 2023](https://dentex.grand-challenge.org/) dataset.

---

## Pipeline at a glance

```
                       ┌──────────────────────────┐
   OPG (1024×1024) ─►  │   YOLO26x  +  4-fold TTA │
                       │   (identity / h-flip /   │
                       │    scale 0.83× / 1.20×)  │
                       └────────────┬─────────────┘
                                    ▼
                       ┌──────────────────────────┐
                       │   Weighted Box Fusion    │
                       │   (per-class, IoU 0.55)  │
                       └────────────┬─────────────┘
                                    ▼
        D  =  count of "Caries"      (τ = 0.30)
        F  =  count of "Filled"      (τ = 0.45, tuned on internal val)
        M  =  max(0, 28 − present)   (WHO 28-tooth convention)
        DMFT  =  D + M + F           ∈ [0, 28]
```

**Detector.** YOLO26x (Ultralytics, COCO-pretrained), fine-tuned on the 705
disease-annotated DENTEX radiographs re-coded into three tooth-status classes
(Healthy, Caries, Filled).

**No data leakage.** The 705 annotated radiographs are split image-wise 80/20
(564 train / 141 held-out validation, seed 42) by `make_clean_split.py`; the
held-out split is never seen during training. Per-class count thresholds are
tuned **only** on this internal split and then applied unchanged externally.

## Key results (honest, leakage-free)

| Endpoint | Value |
|---|---|
| Internal held-out detection (n=141) | mAP₅₀–₉₅ 0.59, AP₅₀ 0.81 (Healthy 0.71 / Caries 0.43 / Filled 0.63) |
| External DMFT vs adjudicated (n=100) | ICC 0.87, Pearson 0.92, MAE 2.70 |
| Human inter-rater (A vs B) | ICC 0.89, MAE 1.46 |
| AI vs inter-rater | ICC difference −0.01 (95% CI −0.09 to 0.08) → **comparable, not superior** |
| Component agreement | Missing r = 0.91, Filled r = 0.91, **Decayed r = 0.23** |
| Systematic bias | +1.96 DMFT (AI over-estimates, caries-driven) |

The pipeline agrees with dentists at a level comparable to their agreement with
each other, is strongest for missing/filled teeth, and over-estimates the
decayed component — reflecting the known limits of panoramic radiography for
caries and the more liberal caries labelling of the training data relative to
the WHO convention.

## Repository layout

```
.
├── configs/
│   ├── paths.py                    portable path config
│   ├── dentex_disease.yaml         original DENTEX YOLO data config
│   └── dentex_disease_split.yaml   clean 80/20 split config (no leakage)
└── scripts/
    ├── make_clean_split.py         disjoint 80/20 image-wise split
    ├── train_yolo26x_split.py      YOLO26x training driver
    ├── internal_metrics_clean.py   held-out per-class AP / precision / recall
    ├── tune_thresholds.py          per-class count thresholds on internal val
    ├── inference_pipeline.py       single-model 4-fold TTA + WBF + WHO-28 DMFT
    ├── tta_ablation_external.py    TTA effect on the external DMFT endpoint
    ├── internal_tta_check.py       TTA effect on internal count error
    ├── statistics.py               bootstrap CIs, Bland-Altman, ICC, κ, symmetric analysis
    └── make_figures_final.py       regenerate the result figures from the metrics
```

## Reproducing the results on DENTEX

```bash
conda create -n dmft python=3.10 && conda activate dmft
pip install -r requirements.txt
export DENTEX_ROOT=/path/to/dentex_dataset      # public DENTEX 2023

python scripts/make_clean_split.py              # 564/141 disjoint split
python scripts/train_yolo26x_split.py           # train YOLO26x (~1.5 h on RTX 3090)
python scripts/internal_metrics_clean.py        # held-out detection metrics
python scripts/tune_thresholds.py               # per-class thresholds (internal only)
```

External-cohort evaluation (your own anonymised OPGs + two-rater DMFT files):

```bash
python scripts/inference_pipeline.py --images "$EXTERNAL/images" --out results/external_predictions.json
python scripts/statistics.py --ai_external results/external_predictions.json \
       --rater_a "$EXTERNAL/annotation_RaterA.xlsx" --rater_b "$EXTERNAL/annotation_RaterB.xlsx"
python scripts/make_figures_final.py
```

Tested with Python 3.10, PyTorch 2.10 + CUDA 13, Ultralytics 8.4, matplotlib
3.10 on a single NVIDIA RTX 3090 (24 GB).

## Citation

If you use this code, please cite our manuscript (citation added upon
acceptance) and the underlying methodological references:

* Jocher G, Qiu J. *Ultralytics YOLO26.* Ultralytics (2026).
* Solovyev R, Wang W, Gabruseva T. Weighted boxes fusion. *Image Vis Comput*
  2021;107:104117. <https://doi.org/10.1016/j.imavis.2021.104117>
* Hamamci IE, Er S, Simsar E, *et al.* DENTEX. *arXiv:2305.19112* (2023).
  <https://doi.org/10.48550/arXiv.2305.19112>

## License

Released under the MIT License (see `LICENSE`).

## Contact

For methodological questions please open a GitHub issue. Requests for
de-identified derived statistics may be directed to the corresponding author.
