"""
Portable path configuration.

Set the dataset locations via environment variables so the pipeline runs on any
machine without code edits:

    export DENTEX_ROOT=/path/to/dentex_dataset           # public DENTEX 2023
    export EXTERNAL_OPGS=/path/to/external_validation     # your own anonymised cohort
    export PROJ_ROOT=/path/to/this/checkout               # optional; defaults to repo root

No patient data or user-specific absolute paths are stored in this repository.
"""
import os
from pathlib import Path

# Repo root (this file lives in <root>/configs/paths.py)
PROJ2_ROOT = Path(os.environ.get('PROJ_ROOT', Path(__file__).resolve().parents[1]))
PROJ1_ROOT = PROJ2_ROOT

# --- public DENTEX 2023 dataset (set DENTEX_ROOT) ---
DENTEX_ROOT             = Path(os.environ.get('DENTEX_ROOT', PROJ2_ROOT / 'dentex_dataset'))
DENTEX_COCO_DISEASE_ALL = DENTEX_ROOT / 'coco' / 'disease_all'
DENTEX_YOLO_DISEASE_ALL = DENTEX_ROOT / 'yolo' / 'disease_all'

# --- checkpoints / starters ---
CHECKPOINTS_SHARED  = Path(os.environ.get('CHECKPOINTS', PROJ2_ROOT / 'checkpoints'))
CKPT_YOLOV8_DISEASE = CHECKPOINTS_SHARED / 'yolo_disease_all.pt'

# --- this project's outputs ---
P2_RESULTS    = PROJ2_ROOT / 'results'
P2_FIGURES    = PROJ2_ROOT / 'figures'
P2_RUNS       = PROJ2_ROOT / 'runs'
P2_LOGS       = PROJ2_ROOT / 'logs'
P2_MANUSCRIPT = PROJ2_ROOT / 'manuscript'

# --- external validation cohort (set EXTERNAL_OPGS); never committed ---
EXT_VAL_ROOT    = Path(os.environ.get('EXTERNAL_OPGS', PROJ2_ROOT / 'external_validation'))
EXT_VAL_IMAGES  = EXT_VAL_ROOT / 'images'
EXT_VAL_RESULTS = EXT_VAL_ROOT / 'results'
DUSSELDORF_SRC  = EXT_VAL_ROOT  # source imagery is private and not distributed


if __name__ == '__main__':
    print('PROJ2_ROOT  :', PROJ2_ROOT)
    print('DENTEX_ROOT :', DENTEX_ROOT, '(exists:', DENTEX_ROOT.exists(), ')')
    print('EXT_VAL_ROOT:', EXT_VAL_ROOT)
