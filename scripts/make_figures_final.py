"""
Final, title-free publication figures for the honest (clean-split, WHO-28,
tuned-threshold) results.  In-plot titles are removed (they belong in the
captions); panel identifiers and per-example data labels are kept.

DPI is capped so the longest side stays <= 1900 px (global image-size rule).

Outputs (figures/):
  fig1_studyflow.png      study flow (clean 80/20 split; 269 -> 100 disclosed)
  fig2_perclass.png       per-class held-out detection AP (n=141)
  fig3_bland_altman.png   AI vs adjudicated scatter + Bland-Altman (n=100)
  fig4_cumulative.png     cumulative agreement: AI-vs-adj + inter-rater
  figS1_component.png     per-component D/M/F scatter (M & F strong, D weak)
  figS2_subgroups.png     subgroup MAE forest
  figS3_examples.png      qualitative example OPGs
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
import cv2
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import P2_FIGURES, P2_RESULTS, EXT_VAL_IMAGES, EXT_VAL_ROOT

P2_FIGURES.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 1.0, 'legend.frameon': False, 'legend.fontsize': 9,
})
C_AI, C_RATER = '#2E75B6', '#E07B00'
C_HEALTHY, C_CARIES, C_FILLED = '#54A14F', '#C84B30', '#9D5DB0'


def save(fig, name, max_px=1850):
    w_in = fig.get_size_inches()[0]
    dpi = min(200, int(max_px / w_in))
    out = P2_FIGURES / name
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    # hard cap: tight bbox can overshoot the nominal width -> enforce <= 1850 px
    from PIL import Image
    im = Image.open(out)
    if max(im.size) > 1850:
        sc = 1850 / max(im.size)
        im.resize((round(im.size[0] * sc), round(im.size[1] * sc)), Image.LANCZOS).save(out)
        im = Image.open(out)
    print(f'Wrote {out}  (dpi={dpi}, {im.size[0]}x{im.size[1]})')


def combined():
    return pd.read_csv(P2_RESULTS / 'external_dmft_combined.csv')

def summary():
    return json.load(open(P2_RESULTS / 'external_dmft_summary.json', encoding='utf-8'))

def internal():
    return json.load(open(P2_RESULTS / 'internal_metrics_clean.json', encoding='utf-8'))


# --------------------------------------------------------------------------
def fig1_studyflow():
    c = {'dentex_total': 1005, 'disease': 705, 'train': 564, 'val': 141,
         'ext_source': 269, 'ext_sel': 100}
    fig, ax = plt.subplots(figsize=(7.2, 9.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 124); ax.axis('off')

    def box(x, y, w, h, t, fill='#DDEBF7', border='#2E75B6', bold=False, fs=9.5, fg='black'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=0.9',
                    linewidth=1.0, facecolor=fill, edgecolor=border))
        ax.text(x + w/2, y + h/2, t, ha='center', va='center', color=fg,
                fontsize=fs, fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                    mutation_scale=12, color='#404040', linewidth=1.3))

    # Internal (DENTEX)
    box(13, 113, 74, 8, f'DENTEX 2023 (MICCAI) panoramic radiographs\nn = {c["dentex_total"]:,}',
        bold=True)
    arrow(50, 113, 50, 107)
    box(13, 99, 74, 7, f'Disease-annotated subset used for detector training\nn = {c["disease"]}',
        fill='#E2EFDA', border='#548235')
    arrow(50, 99, 50, 93)
    box(8, 84, 38, 8, f'Training split\nn = {c["train"]} (80%)', fill='#E2EFDA', border='#548235')
    box(54, 84, 38, 8, f'Held-out internal validation\nn = {c["val"]} (20%, disjoint)',
        fill='#E2EFDA', border='#548235')

    # External
    box(13, 66, 74, 7, 'External cohort: University Hospital Düsseldorf\n'
        f'consecutive series n = {c["ext_source"]}', fill='#FCE4D6', border='#C65911', bold=True)
    arrow(50, 66, 50, 60)
    box(13, 52, 74, 7, f'Selected for image quality\nn = {c["ext_sel"]} anonymised OPGs (D001–D100)',
        fill='#FCE4D6', border='#C65911')
    arrow(50, 52, 50, 46)
    box(6, 35, 40, 9, f'Rater A\nindependent DMFT counts\nn = {c["ext_sel"]} (excluded 0)',
        fill='#FFF2CC', border='#BF8F00', fs=9)
    box(54, 35, 40, 9, f'Rater B\nindependent DMFT counts\nn = {c["ext_sel"]} (excluded 0)',
        fill='#FFF2CC', border='#BF8F00', fs=9)
    arrow(26, 35, 50, 28); arrow(74, 35, 50, 28)
    box(15, 18, 70, 8, f'Adjudicated reference DMFT (component-wise mean)\nn = {c["ext_sel"]}',
        fill='#EFEFEF', border='#404040', bold=True)
    arrow(50, 18, 50, 12)
    box(15, 2, 70, 8, 'External DMFT validation\n(AI predictions vs adjudicated reference)',
        fill='#1F4E79', border='#1F4E79', bold=True, fs=10.5, fg='white')
    save(fig, 'fig1_studyflow.png')


# --------------------------------------------------------------------------
def fig2_perclass():
    d = internal()
    classes = ['Healthy', 'Caries', 'Filled']
    ap = [d['per_class_AP'][c] for c in classes]
    prec = [d.get('per_class_precision', {}).get(c, np.nan) for c in classes]
    rec = [d.get('per_class_recall', {}).get(c, np.nan) for c in classes]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = np.arange(3); w = 0.26
    cols = [C_HEALTHY, C_CARIES, C_FILLED]
    ax.bar(x - w, ap, w, label='AP (IoU 0.50:0.95)', color=cols, edgecolor='#333', linewidth=0.8)
    ax.bar(x, prec, w, label='Precision', color=cols, alpha=0.6, edgecolor='#333', linewidth=0.8)
    ax.bar(x + w, rec, w, label='Recall', color=cols, alpha=0.32, edgecolor='#333', linewidth=0.8)
    for i in range(3):
        ax.text(i - w, ap[i] + 0.015, f'{ap[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, axis='y'); ax.set_axisbelow(True)
    # overall annotation (no title)
    ax.text(0.02, 0.97, f"Overall mAP$_{{50-95}}$ = {d['AP']:.3f}   AP$_{{50}}$ = {d['AP50']:.3f}   "
            f"AP$_{{75}}$ = {d['AP75']:.3f}", transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc'))
    ax.legend(loc='upper right', ncol=1)
    save(fig, 'fig2_perclass.png')


# --------------------------------------------------------------------------
def fig3_bland_altman():
    df = combined()
    s = summary()['ai_vs_gt']['bland_altman']
    gt, ai = df['GT_DMFT'].values, df['AI_DMFT'].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    # scatter
    m = max(gt.max(), ai.max()) + 1
    ax1.scatter(gt, ai, color=C_AI, alpha=0.6, s=26)
    ax1.plot([0, m], [0, m], '--', color='#999', lw=1, label='Line of identity')
    sl, ic = np.polyfit(gt, ai, 1)
    xs = np.linspace(0, m, 50); ax1.plot(xs, sl*xs+ic, color=C_RATER, lw=1.6,
        label=f'Least-squares fit (y={sl:.2f}x+{ic:.2f})')
    r = np.corrcoef(gt, ai)[0, 1]
    ax1.text(0.05, 0.93, f'Pearson r = {r:.2f}\nICC = {summary()["ai_vs_gt"]["icc_2_1"]:.2f}\nn = {len(df)}',
             transform=ax1.transAxes, va='top', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc'))
    ax1.set_xlabel('Adjudicated reference DMFT'); ax1.set_ylabel('AI-predicted DMFT')
    ax1.set_xlim(-1, m); ax1.set_ylim(-1, m); ax1.legend(loc='lower right', fontsize=8.5)
    ax1.grid(True, alpha=0.25, ls='--', lw=0.6); ax1.set_axisbelow(True)
    # bland-altman
    mean_v = (gt + ai) / 2; diff = ai - gt
    ax2.scatter(mean_v, diff, color=C_AI, alpha=0.6, s=26)
    ax2.axhline(s['bias'], color=C_RATER, lw=1.6, label=f"Bias = {s['bias']:.2f}")
    ax2.axhline(s['loa_upper'], color='#999', ls='--', lw=1.2,
                label=f"95% LoA [{s['loa_lower']:.2f}, {s['loa_upper']:.2f}]")
    ax2.axhline(s['loa_lower'], color='#999', ls='--', lw=1.2)
    ax2.axhline(0, color='#333', lw=0.7)
    ax2.set_xlabel('Mean of AI and reference DMFT'); ax2.set_ylabel('AI − reference DMFT')
    ax2.legend(loc='upper right', fontsize=8.5)
    ax2.grid(True, alpha=0.25, ls='--', lw=0.6); ax2.set_axisbelow(True)
    save(fig, 'fig3_bland_altman.png')


# --------------------------------------------------------------------------
def fig4_cumulative():
    ca = summary()['cumulative_agreement']
    ns = list(range(0, 9))
    ai = [ca['AI_vs_adjudicated'][str(n)] * 100 for n in ns]
    ab = [ca['inter_rater_A_vs_B'][str(n)] * 100 for n in ns]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(ns, ai, marker='o', ms=7, lw=1.8, color=C_AI, label='AI vs adjudicated reference')
    ax.plot(ns, ab, marker='s', ms=6, lw=1.6, color=C_RATER, ls='--',
            label='Rater A vs Rater B (between-dentist)')
    for x, y in zip(ns, ai):
        ax.text(x, y + 2, f'{y:.0f}', ha='center', fontsize=8, color=C_AI)
    ax.set_xticks(ns); ax.set_xlabel('Absolute DMFT tolerance N (units)')
    ax.set_ylabel('OPGs within tolerance (%)'); ax.set_ylim(0, 108)
    ax.grid(True, alpha=0.25, ls='--', lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc='lower right')
    save(fig, 'fig4_cumulative.png')


# --------------------------------------------------------------------------
def figS1_component():
    df = combined()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, comp, color, lab in [(axes[0], 'D', C_CARIES, 'Decayed (D)'),
                                 (axes[1], 'M', '#5D7B8C', 'Missing (M)'),
                                 (axes[2], 'F', C_FILLED, 'Filled (F)')]:
        gt, ai = df[f'GT_{comp}'], df[f'AI_{comp}']
        ax.scatter(gt, ai, color=color, alpha=0.65, s=20)
        m = max(gt.max(), ai.max()) + 1
        ax.plot([0, m], [0, m], '--', color='#999', lw=1)
        sl, ic = np.polyfit(gt, ai, 1); xs = np.linspace(0, m, 50)
        ax.plot(xs, sl*xs+ic, color='#1F1F1F', lw=1.3)
        r = np.corrcoef(gt, ai)[0, 1]
        ax.text(0.05, 0.93, f'{lab}\nr = {r:.2f}', transform=ax.transAxes, va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc'))
        ax.set_xlim(-0.5, m); ax.set_ylim(-0.5, m)
        ax.set_xlabel('Adjudicated count'); ax.set_ylabel('AI-predicted count')
        ax.grid(True, alpha=0.25, ls='--', lw=0.6); ax.set_axisbelow(True)
    save(fig, 'figS1_component.png')


# --------------------------------------------------------------------------
def figS2_subgroups():
    df = combined(); df['abs_err'] = np.abs(df['AI_DMFT'] - df['GT_DMFT'])
    rng = np.random.default_rng(42)
    def ci(v):
        v = np.asarray(v, float)
        b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(2000)])
        return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
    rows = []
    mean, lo, hi = ci(df['abs_err']); rows.append(('Overall', mean, lo, hi, len(df)))
    for lab, a, b in [('DMFT = 0', 0, 0), ('DMFT 1–5', 1, 5), ('DMFT 6–12', 6, 12), ('DMFT ≥ 13', 13, 99)]:
        sub = df[(df['GT_DMFT'] >= a) & (df['GT_DMFT'] <= b)]
        if len(sub) >= 3:
            m, l, h = ci(sub['abs_err']); rows.append((lab, m, l, h, len(sub)))
    try:
        a_full = pd.read_excel(EXT_VAL_ROOT / 'annotation_RaterA_Duesseldorf.xlsx', sheet_name='annotations')
        dc = df.merge(a_full[['image_id', 'confidence']], on='image_id', how='left')
        for lvl, lab in [(3, 'Rater conf: high'), (2, 'Rater conf: medium'), (1, 'Rater conf: low')]:
            sub = dc[dc['confidence'] == lvl]
            if len(sub) >= 3:
                m, l, h = ci(sub['abs_err']); rows.append((lab, m, l, h, len(sub)))
    except Exception as e:
        print('  (confidence subgroup skipped:', e, ')')
    try:
        smap = pd.read_csv(EXT_VAL_ROOT / 'source_mapping.csv')
        col = 'blinded_id' if 'blinded_id' in smap else 'image_id'
        lapcol = [c for c in smap.columns if 'lap' in c.lower()][0]
        dq = df.merge(smap[[col, lapcol]], left_on='image_id', right_on=col, how='left')
        if dq[lapcol].notna().all():
            terc = pd.qcut(dq[lapcol], 3, labels=['lower', 'middle', 'upper'])
            for lab in terc.cat.categories:
                sub = dq[terc == lab]
                if len(sub) >= 3:
                    m, l, h = ci(sub['abs_err']); rows.append((f'Sharpness: {lab}', m, l, h, len(sub)))
    except Exception as e:
        print('  (sharpness subgroup skipped:', e, ')')

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 1.2))
    ys = np.arange(len(rows))[::-1]; overall = rows[0][1]
    for y, (lab, m, l, h, n) in zip(ys, rows):
        ov = lab == 'Overall'
        ax.errorbar([m], [y], xerr=[[m - l], [h - m]], fmt='s' if ov else 'o',
                    color=C_AI if ov else '#404040', ms=9 if ov else 6, capsize=4, lw=1.4)
        ax.text(1.02, y, f'{m:.2f} ({l:.2f}, {h:.2f})  n={n}', transform=ax.get_yaxis_transform(),
                va='center', fontsize=8.5, family='monospace')
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlabel('Mean absolute DMFT error (95% CI)')
    ax.axvline(overall, color='#999', ls=':', lw=0.9)
    ax.set_xlim(0, max(max(r[3] for r in rows) * 1.05, 6))
    ax.grid(True, alpha=0.25, ls='--', lw=0.6, axis='x'); ax.set_axisbelow(True)
    save(fig, 'figS2_subgroups.png')


# --------------------------------------------------------------------------
CLASS_COLORS = {0: (84, 161, 79), 1: (200, 75, 48), 2: (157, 93, 176)}
CLASS_NAMES = {0: 'Healthy', 1: 'Caries', 2: 'Filled'}

def figS3_examples():
    df = combined()
    preds = {p['image_id']: p for p in json.load(open(P2_RESULTS / 'external_predictions.json', encoding='utf-8'))}
    df['err'] = df['AI_DMFT'] - df['GT_DMFT']; df['abs_err'] = df['err'].abs()
    sel, titles = [], []
    def add(c, tag):
        if c['image_id'] not in sel:
            sel.append(c['image_id'])
            titles.append(f"{c['image_id']}  {tag}: ref DMFT={int(c['GT_DMFT'])}, AI={int(c['AI_DMFT'])}")
    for q in [df[(df['GT_DMFT'] == 0) & (df['abs_err'] <= 1)],
              df[(df['GT_DMFT'].between(3, 8)) & (df['abs_err'] <= 1)].sort_values('abs_err'),
              df[df['GT_DMFT'] >= 13].sort_values('abs_err'),
              df[df['err'] < -3].sort_values('err'),
              df[df['err'] > 3].sort_values('err', ascending=False)]:
        if len(q): add(q.iloc[0], {0:'caries-free',1:'mid-range',2:'high-burden',3:'AI under',4:'AI over'}[len(sel)] if len(sel)<5 else 'case')
    c = df.sort_values('abs_err', ascending=False).iloc[0]; add(c, 'largest disagreement')
    sel, titles = sel[:6], titles[:6]
    rows = (len(sel) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12.5, 3.3 * rows))
    axes = np.atleast_2d(axes)
    for i, (sid, ttl) in enumerate(zip(sel, titles)):
        ax = axes[i // 2, i % 2]
        img = cv2.imread(str(EXT_VAL_IMAGES / f'{sid}.jpg'))
        for p in preds[sid]['boxes']:
            if p['conf'] < 0.4: continue
            cls = {'Healthy': 0, 'Caries': 1, 'Filled': 2}[p['class']]
            col = (CLASS_COLORS[cls][2], CLASS_COLORS[cls][1], CLASS_COLORS[cls][0])
            cv2.rectangle(img, (int(p['x1']), int(p['y1'])), (int(p['x2']), int(p['y2'])),
                          col, max(2, int(img.shape[1] / 600)))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax.set_title(ttl, fontsize=9, pad=4); ax.axis('off')
    for j in range(len(sel), rows * 2):
        axes[j // 2, j % 2].axis('off')
    fig.legend(handles=[Patch(facecolor=tuple(v/255 for v in CLASS_COLORS[i]), edgecolor='k',
               label=CLASS_NAMES[i]) for i in [0, 1, 2]], loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=10, frameon=False)
    save(fig, 'figS3_examples.png')


if __name__ == '__main__':
    fig1_studyflow(); fig2_perclass(); fig3_bland_altman(); fig4_cumulative()
    figS1_component(); figS2_subgroups(); figS3_examples()
    print('All figures regenerated.')
