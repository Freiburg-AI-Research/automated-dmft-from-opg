"""
Create a clean, disjoint 80/20 image-wise split of the 705 DENTEX disease_all
images, fixing the train==val leakage (val2017 was a byte-identical copy of
train2017).

All 705 source images live in images/train2017 with labels in labels/train2017.
We list them, shuffle with a fixed seed, and write two disjoint path lists plus a
new Ultralytics data yaml.  Ultralytics resolves each label by swapping
'/images/' -> '/labels/' and the extension to .txt, so both splits read labels
from labels/train2017 with no copying required.

Outputs (in the dataset root):
  train_split.txt   564 absolute image paths
  val_split.txt     141 absolute image paths
  dentex_disease_split.yaml
"""
import sys, io, random
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT      = Path(r'.\dentex_dataset\yolo\disease_all')
IMG_DIR   = ROOT / 'images' / 'train2017'
LBL_DIR   = ROOT / 'labels' / 'train2017'
CFG_DIR   = Path(r'.\configs')
SEED      = 42
VAL_FRAC  = 0.20

imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() == '.png')
print(f'Found {len(imgs)} images in {IMG_DIR}')

# keep only images that actually have a (non-empty path) label file
paired = [p for p in imgs if (LBL_DIR / (p.stem + '.txt')).exists()]
missing = len(imgs) - len(paired)
print(f'  with label file: {len(paired)}   (missing labels: {missing})')

rng = random.Random(SEED)
rng.shuffle(paired)
n_val = round(len(paired) * VAL_FRAC)
val   = sorted(paired[:n_val], key=lambda p: p.name)
train = sorted(paired[n_val:], key=lambda p: p.name)
print(f'Split: {len(train)} train / {len(val)} val (seed {SEED}, val_frac {VAL_FRAC})')

# sanity: disjoint
assert set(train).isdisjoint(set(val)), 'train/val overlap!'

train_txt = ROOT / 'train_split.txt'
val_txt   = ROOT / 'val_split.txt'
train_txt.write_text('\n'.join(str(p) for p in train) + '\n', encoding='utf-8')
val_txt.write_text('\n'.join(str(p) for p in val) + '\n', encoding='utf-8')
print(f'Wrote {train_txt}\nWrote {val_txt}')

yaml = f"""# Clean 80/20 image-wise split of the DENTEX disease_all set (seed {SEED}).
# Replaces dentex_disease.yaml, whose val2017 was a copy of train2017 (leakage).
path: {ROOT.as_posix()}
train: train_split.txt
val:   val_split.txt

nc: 3
names:
  0: Healthy
  1: Caries
  2: Filled
"""
yaml_path = CFG_DIR / 'dentex_disease_split.yaml'
yaml_path.write_text(yaml, encoding='utf-8')
print(f'Wrote {yaml_path}')
print('\n--- yaml ---')
print(yaml)
