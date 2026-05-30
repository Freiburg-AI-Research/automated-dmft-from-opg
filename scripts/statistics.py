"""
Statistical analysis pipeline for Project 2.

Inputs (JSON files produced by inference_pipeline.py + the rater XLSX files):
    --ai_internal      predictions on internal DENTEX test set
    --gt_internal      ground-truth COCO JSON for DENTEX test
    --ai_external      predictions on the 100 Düsseldorf OPGs
    --rater_a, --rater_b   per-rater DMFT counts (XLSX from build_annotation_pack)

Outputs:
    results/tables/   CSV files for every paper table
    results/stats.json    machine-readable summary
    results/raters_adjudicated.csv  unified ground truth for external set

Functions:
    bootstrap_ci()    Bootstrap 95% CI of any callable metric
    bland_altman()    Means, biases, limits of agreement
    icc_2_1()         Intraclass correlation, two-way random, single rater
    cohens_kappa()    For DMFT bucketed into clinical ranges
    delong_test()     For ROC AUC comparisons
    expected_calibration_error()  ECE with N bins
    temperature_scaling_fit()    Post-hoc calibration
    mae_dmft()        MAE between rater DMFT and AI DMFT
    per_class_metrics() Precision, Recall, F1 with bootstrap CIs

This module is purely numerical - it does NOT touch the figures or the
manuscript.  Tables in this file map 1:1 onto tables in the manuscript.
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'configs'))
from paths import P2_RESULTS, EXT_VAL_ROOT

N_BOOT = 10000
RNG = np.random.default_rng(42)


# ============================================================================
# Bootstrap CI
# ============================================================================

def bootstrap_ci(values, fn=np.mean, n_boot=N_BOOT, alpha=0.05, paired_with=None):
    """Percentile bootstrap 95% CI of `fn(values)`.
    If `paired_with` is provided, both arrays are resampled with the same indices
    (paired bootstrap)."""
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        if paired_with is None:
            samples[i] = fn(values[idx])
        else:
            samples[i] = fn(values[idx], np.asarray(paired_with)[idx])
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(fn(values) if paired_with is None else fn(values, paired_with)),
            float(lo), float(hi))


# ============================================================================
# Bland-Altman
# ============================================================================

def bland_altman(y_ref, y_pred):
    y_ref = np.asarray(y_ref); y_pred = np.asarray(y_pred)
    mean = (y_ref + y_pred) / 2
    diff = y_pred - y_ref
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    loa_low  = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd
    # 95% CIs for bias and LoA
    n = len(diff)
    se_bias = sd / np.sqrt(n)
    bias_ci = (bias - 1.96 * se_bias, bias + 1.96 * se_bias)
    return {
        'mean':  mean.tolist(),
        'diff':  diff.tolist(),
        'bias':  bias,
        'bias_95ci': bias_ci,
        'sd':    sd,
        'loa_lower': loa_low,
        'loa_upper': loa_high
    }


# ============================================================================
# Intraclass correlation (two-way random, single measurement)
# ============================================================================

def icc_2_1(rater_a, rater_b):
    """ICC(2,1) per Shrout & Fleiss (1979) for 2 raters and single measure."""
    a = np.asarray(rater_a, dtype=float)
    b = np.asarray(rater_b, dtype=float)
    n = len(a)
    if n < 3:
        return float('nan')
    Y = np.stack([a, b], axis=1)  # n x 2
    k = 2
    mean_subject = Y.mean(axis=1)
    mean_rater   = Y.mean(axis=0)
    grand_mean   = Y.mean()
    ss_between_subjects = k * np.sum((mean_subject - grand_mean) ** 2)
    ss_between_raters   = n * np.sum((mean_rater   - grand_mean) ** 2)
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_residual = ss_total - ss_between_subjects - ss_between_raters
    ms_b_subjects = ss_between_subjects / (n - 1)
    ms_b_raters   = ss_between_raters   / (k - 1)
    ms_residual   = ss_residual         / ((n - 1) * (k - 1))
    icc = (ms_b_subjects - ms_residual) / (
        ms_b_subjects + (k - 1) * ms_residual +
        k * (ms_b_raters - ms_residual) / n
    )
    return float(icc)


# ============================================================================
# Cohen's kappa (linear weights for ordinal DMFT bins)
# ============================================================================

def cohens_kappa_weighted(a, b, weight='linear', max_class=None):
    a = np.asarray(a, dtype=int); b = np.asarray(b, dtype=int)
    if max_class is None:
        max_class = max(a.max(), b.max())
    K = max_class + 1
    M = np.zeros((K, K))
    for ai, bi in zip(a, b):
        M[ai, bi] += 1
    n = M.sum()
    po = M.sum() / n
    # weight matrix
    W = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if weight == 'linear':
                W[i, j] = 1 - abs(i - j) / (K - 1)
            elif weight == 'quadratic':
                W[i, j] = 1 - ((i - j) / (K - 1)) ** 2
            else:
                W[i, j] = 1.0 if i == j else 0.0
    row_marg = M.sum(axis=1) / n
    col_marg = M.sum(axis=0) / n
    pe = np.outer(row_marg, col_marg)
    obs = (M * W).sum() / n
    exp = (pe * W).sum()
    return float((obs - exp) / max(1 - exp, 1e-9))


# ============================================================================
# Calibration: Expected Calibration Error (ECE)
# ============================================================================

def expected_calibration_error(confs, correct, n_bins=15):
    """Standard ECE (Guo et al., 2017). `correct` is 0/1 array.
    Returns ECE in [0, 1]."""
    confs = np.asarray(confs); correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ============================================================================
# Temperature scaling (post-hoc calibration)
# ============================================================================

def temperature_scaling_fit(logits, targets, max_iter=200, lr=0.01):
    """Fit a single temperature T (>0) on logits to minimise NLL on (logits, targets).
    Returns the fitted T as a float."""
    import torch
    logits = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T.clamp(min=0.05), targets)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().item())


# ============================================================================
# Per-class detection metrics
# ============================================================================

def per_class_metrics_from_matches(matches_per_class, totals_per_class):
    """For each class, compute precision/recall/F1 with bootstrap CIs.
    matches_per_class is a dict: class -> list of (is_tp 0/1, is_fp 0/1, label_present 0/1)
    """
    out = {}
    for cls, M in matches_per_class.items():
        tp = np.array([m[0] for m in M])
        fp = np.array([m[1] for m in M])
        gt = np.array([m[2] for m in M])
        if len(M) == 0:
            continue
        def precision(t, f, g): return (t.sum() / max((t.sum() + f.sum()), 1e-9))
        def recall   (t, f, g): return (t.sum() / max(g.sum(), 1e-9))
        def f1       (t, f, g):
            p = precision(t, f, g); r = recall(t, f, g)
            return 2 * p * r / max(p + r, 1e-9)
        out[cls] = {
            'precision': bootstrap_ci(tp, lambda x: tp.sum()/max(tp.sum()+fp.sum(),1e-9)),
            'recall':    bootstrap_ci(tp, lambda x: tp.sum()/max(gt.sum(),1e-9)),
            'f1':        bootstrap_ci(tp, lambda x: 2*(tp.sum()/max(tp.sum()+fp.sum(),1e-9))*(tp.sum()/max(gt.sum(),1e-9))/max((tp.sum()/max(tp.sum()+fp.sum(),1e-9))+(tp.sum()/max(gt.sum(),1e-9)),1e-9))
        }
    return out


# ============================================================================
# DMFT per OPG
# ============================================================================

def mae(y_ref, y_pred):
    y_ref = np.asarray(y_ref, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_ref - y_pred)))


def pearson_r(y_ref, y_pred):
    y_ref = np.asarray(y_ref, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.corrcoef(y_ref, y_pred)[0, 1])


# ============================================================================
# Paired bootstrap CIs for two-array metrics + their differences
# ============================================================================

def boot_pair(x, y, fn, n_boot=N_BOOT):
    """Bootstrap 95% CI of a symmetric two-array metric fn(x, y) (e.g. icc, mae)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); n = len(x)
    est = fn(x, y)
    samp = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        samp[i] = fn(x[idx], y[idx])
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return (float(est), float(lo), float(hi))


def boot_diff(xa, ya, xb, yb, fn, n_boot=N_BOOT):
    """Bootstrap 95% CI of the paired difference fn(xa, ya) - fn(xb, yb),
    resampling OPGs with the SAME indices for both terms (paired)."""
    xa, ya, xb, yb = (np.asarray(v, dtype=float) for v in (xa, ya, xb, yb))
    n = len(xa)
    est = fn(xa, ya) - fn(xb, yb)
    samp = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        samp[i] = fn(xa[idx], ya[idx]) - fn(xb[idx], yb[idx])
    lo, hi = np.percentile(samp, [2.5, 97.5])
    return (float(est), float(lo), float(hi))


def cumulative_agreement(diffs, ks=range(0, 9)):
    """Fraction of OPGs whose |difference| is <= k DMFT units, for each k."""
    d = np.abs(np.asarray(diffs, dtype=float))
    return {int(k): float(np.mean(d <= k)) for k in ks}


# ============================================================================
# Top-level orchestration: external DMFT validation
# ============================================================================

def external_dmft_analysis(ai_preds_json, rater_a_xlsx, rater_b_xlsx, out_dir):
    """Compare AI DMFT predictions against rater-derived ground truth for the
    100 Düsseldorf OPGs."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    with open(ai_preds_json, encoding='utf-8') as fh:
        ai = json.load(fh)
    ai_df = pd.DataFrame([
        {'image_id': r['image_id'],
         'AI_D': r['dmft']['D'], 'AI_M': r['dmft']['M'], 'AI_F': r['dmft']['F'],
         'AI_DMFT': r['dmft']['DMFT']}
        for r in ai
    ])

    a_df = pd.read_excel(rater_a_xlsx, sheet_name='annotations')[
        ['image_id', 'D', 'M', 'F', 'DMFT', 'excluded']]
    b_df = pd.read_excel(rater_b_xlsx, sheet_name='annotations')[
        ['image_id', 'D', 'M', 'F', 'DMFT', 'excluded']]
    a_df.columns = ['image_id', 'A_D', 'A_M', 'A_F', 'A_DMFT', 'A_excluded']
    b_df.columns = ['image_id', 'B_D', 'B_M', 'B_F', 'B_DMFT', 'B_excluded']

    df = ai_df.merge(a_df, on='image_id', how='inner').merge(b_df, on='image_id', how='inner')
    # exclude OPGs marked as uninterpretable by either rater
    df = df[(df['A_excluded'].fillna(0) == 0) & (df['B_excluded'].fillna(0) == 0)].copy()

    # adjudicated ground truth = mean of rater A and B for D, M, F (rounded)
    for c in ['D', 'M', 'F']:
        df[f'GT_{c}'] = ((df[f'A_{c}'].astype(float) + df[f'B_{c}'].astype(float)) / 2).round().astype(int)
    df['GT_DMFT'] = df['GT_D'] + df['GT_M'] + df['GT_F']

    df.to_csv(out_dir / 'external_dmft_combined.csv', index=False)

    # Inter-rater (between rater A and B)
    inter_rater = {
        'ICC_2_1': icc_2_1(df['A_DMFT'], df['B_DMFT']),
        'kappa_linear_DMFT': cohens_kappa_weighted(df['A_DMFT'], df['B_DMFT']),
        'pearson_r': pearson_r(df['A_DMFT'], df['B_DMFT']),
        'mae': mae(df['A_DMFT'], df['B_DMFT'])
    }

    # AI vs GT
    ai_vs_gt = {
        'pearson_r': pearson_r(df['GT_DMFT'], df['AI_DMFT']),
        'icc_2_1':   icc_2_1(df['GT_DMFT'], df['AI_DMFT']),
        'mae_dmft':  bootstrap_ci(np.abs(df['AI_DMFT'] - df['GT_DMFT'])),
        'bland_altman': bland_altman(df['GT_DMFT'], df['AI_DMFT'])
    }
    # per-component
    for comp in ('D', 'M', 'F'):
        ai_vs_gt[f'mae_{comp}'] = bootstrap_ci(np.abs(df[f'AI_{comp}'] - df[f'GT_{comp}']))
        ai_vs_gt[f'pearson_r_{comp}'] = pearson_r(df[f'GT_{comp}'], df[f'AI_{comp}'])

    # ------------------------------------------------------------------
    # Symmetric agreement: AI vs EACH rater and vs the adjudicated score,
    # alongside the human inter-rater agreement, each with bootstrap CIs.
    # This is the honest framing for the "comparable to (not exceeding)
    # inter-rater" claim required by the reviewers.
    # ------------------------------------------------------------------
    def agreement_block(ref, test):
        return {
            'icc_2_1':           boot_pair(ref, test, icc_2_1),
            'pearson_r':         pearson_r(ref, test),
            'mae':               boot_pair(ref, test, mae),
            'bland_altman_bias': bland_altman(ref, test)['bias'],
        }
    symmetric = {
        'inter_rater_A_vs_B': agreement_block(df['A_DMFT'], df['B_DMFT']),
        'AI_vs_A':            agreement_block(df['A_DMFT'], df['AI_DMFT']),
        'AI_vs_B':            agreement_block(df['B_DMFT'], df['AI_DMFT']),
        'AI_vs_adjudicated':  agreement_block(df['GT_DMFT'], df['AI_DMFT']),
        # Overclaim test: AI-vs-adjudicated MINUS human inter-rater (paired).
        # A CI overlapping 0 means AI agreement is COMPARABLE, not superior.
        'delta_icc_AIadj_minus_interrater': boot_diff(
            df['GT_DMFT'], df['AI_DMFT'], df['A_DMFT'], df['B_DMFT'], icc_2_1),
        'delta_mae_AIadj_minus_interrater': boot_diff(
            df['GT_DMFT'], df['AI_DMFT'], df['A_DMFT'], df['B_DMFT'], mae),
    }

    # Cumulative agreement (fraction of OPGs within k DMFT units)
    cumulative = {
        'AI_vs_adjudicated':  cumulative_agreement(df['AI_DMFT'] - df['GT_DMFT']),
        'AI_vs_A':            cumulative_agreement(df['AI_DMFT'] - df['A_DMFT']),
        'AI_vs_B':            cumulative_agreement(df['AI_DMFT'] - df['B_DMFT']),
        'inter_rater_A_vs_B': cumulative_agreement(df['A_DMFT'] - df['B_DMFT']),
    }

    summary = {
        'n_included':  int(len(df)),
        'inter_rater': inter_rater,
        'ai_vs_gt':    ai_vs_gt,
        'symmetric':   symmetric,
        'cumulative_agreement': cumulative,
    }
    with open(out_dir / 'external_dmft_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ai_external', required=True)
    ap.add_argument('--rater_a',     required=True)
    ap.add_argument('--rater_b',     required=True)
    ap.add_argument('--out',         default=str(P2_RESULTS))
    args = ap.parse_args()
    external_dmft_analysis(args.ai_external, args.rater_a, args.rater_b, args.out)
