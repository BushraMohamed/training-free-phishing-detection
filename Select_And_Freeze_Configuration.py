"""
Select_And_Freeze_Configuration.py
=============================================================================
Choose the final configuration on DEVELOPMENT data only, FREEZE it, and only
then evaluate it on held-out TEST data.

Costs nothing: it reuses the per-email scores already saved in
score_cache_reasoning_only/ (TF-IDF code version). No API calls.

PROTOCOL (fixed before any result is looked at)
  1. Every dataset is split 50/50, stratified by class, with a fixed seed
     -> a development half and a test half.
  2. Candidate settings: Perspective weight in {0,10,20,30,40,50}% with
     Reasoning weight = 100% - Perspective weight and Pattern weight = 0%
     (the three weights always sum to 100%), crossed with threshold T in
     {20,25,30,35,40}.
  3. Design constraint: Perspective weight >= 10%, so the Perspective Agent
     stays an active part of the multi-agent system.
  4. Selection rule: maximise the mean F1 over the development halves of
     Enron, Salloum English, Salloum Arabic, MeAJOR and the synthetic Arabic
     set. (Nazario is phishing-only, so its F1 is recall only: it is reported
     but not used to choose.) Settings within 0.25pp of the best are
     treated as tied, and the tie is broken by fewer false positives.
  5. The choice is written to frozen_config.json BEFORE the test halves are
     evaluated. Do not edit that file by hand.
  6. The test halves give the reported numbers, with paired-bootstrap 95%
     intervals against the equal-weight and Reasoning-only settings.

HONEST LIMIT: you have already seen results on the full datasets, so these
halves are not blind. For a genuinely unseen check, run
Confirm_On_Fresh_MeAJOR.py afterwards (it reads frozen_config.json).

Run in the folder that holds score_cache_reasoning_only/ and the six CSV
files (Enron.csv, Nazario.csv, Salloum_English_300.csv, Salloum_Arabic_300.csv,
meajor_cleaned_preprocessed.csv, UAE_Synthetic_Arabic_Combined_600.csv):

    python3 Select_And_Freeze_Configuration.py
"""
import os, sys, json, csv, hashlib
from datetime import datetime
import numpy as np
import pandas as pd

CACHE_DIR = 'score_cache_reasoning_only'
OUT_DIR = 'final_configuration'
SEED = 2026
PE_GRID = [0, 10, 20, 30, 40, 50]
T_GRID = [20, 25, 30, 35, 40]
MIN_PE = 10
TIE_TOL = 0.25
N_BOOT = 1000
EQUAL_WEIGHT = (50, 30)      # (Perspective %, threshold): the earlier reported setting
REASONING_ONLY = (0, 30)

# name, csv file, cache key, used for selection?
DATASETS = [
    ('Enron Full',           'Enron.csv',                             'Enron_Full',           True),
    ('Nazario Full',         'Nazario.csv',                           'Nazario_Full',         False),
    ('Salloum English 300',  'Salloum_English_300.csv',               'Salloum_English_300',  True),
    ('Salloum Arabic 300',   'Salloum_Arabic_300.csv',                'Salloum_Arabic_300',   True),
    ('MeAJOR 2025 (3,000)',  'meajor_cleaned_preprocessed.csv',       'MeAJOR_2025',          True),
    ('UAE Synthetic Arabic', 'UAE_Synthetic_Arabic_Combined_600.csv', 'UAE_Synthetic_Arabic', True),
]


def load_labels(name, path):
    df = pd.read_csv(path)
    if name.startswith('MeAJOR'):                       # reproduce the exact 3,000-email sample (seed 42)
        en = df[df['language'] == 'en'].copy()
        en = en[en['body'].notna()]
        en = en[~en['body'].str.contains(r'<\|EMOJI\|>', na=False)]
        en = en[en['body'].str.len() > 50]
        legit = en[en['label'] == 0].sample(1500, random_state=42)
        phish = en[en['label'] == 1].sample(1500, random_state=42)
        s = pd.concat([legit, phish]).sample(frac=1, random_state=42).reset_index(drop=True)
        return s['label'].astype(int).to_numpy()
    lc = next(c for c in df.columns if c.strip().lower() in ('label', 'class', 'type', 'category'))
    return df[lc].astype(int).to_numpy()


def load_scores(cache_key, n):
    path = os.path.join(CACHE_DIR, f'{cache_key}_scores.json')
    if not os.path.exists(path):
        return None, None, path
    cache = json.load(open(path))
    r = np.full(n, np.nan); pe = np.full(n, np.nan)
    for k, v in cache.items():
        i = int(k)
        if i < n:
            r[i] = v['r']; pe[i] = v['pe']
    return r, pe, path


def stratified_split(y, seed):
    rng = np.random.RandomState(seed)
    dev = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; rng.shuffle(idx); dev[idx[:len(idx) // 2]] = True
    return dev


def prf(y, pred):
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {'F1': 100 * f, 'Precision': 100 * p, 'Recall': 100 * r, 'FP': fp, 'FN': fn}


def combined(d, pe_pct):
    w = pe_pct / 100.0
    return (1 - w) * d['r'] + w * d['pe']


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    print("\n" + "=" * 78)
    print("  SELECT AND FREEZE THE FINAL CONFIGURATION  (development halves only)")
    print(f"  {datetime.now():%Y-%m-%d %H:%M} | cached scores, no API calls, cost $0")
    print("=" * 78)
    if not os.path.isdir(CACHE_DIR):
        sys.exit(f"\n{CACHE_DIR}/ not found. Run this in the folder that contains it.")
    os.makedirs(OUT_DIR, exist_ok=True)

    data, cache_files = {}, {}
    for i, (name, csv_file, key, use_for_sel) in enumerate(DATASETS):
        if not os.path.exists(csv_file):
            print(f"  skip {name}: {csv_file} not found"); continue
        y = load_labels(name, csv_file)
        r, pe, cpath = load_scores(key, len(y))
        if r is None:
            print(f"  skip {name}: no cache file {cpath}"); continue
        ok = ~np.isnan(r) & ~np.isnan(pe)
        if ok.sum() < len(y):
            print(f"  WARNING {name}: only {int(ok.sum())}/{len(y)} emails have scores; using those")
        y, r, pe = y[ok], r[ok], pe[ok]
        dev = stratified_split(y, SEED + i)
        data[name] = {'y': y, 'r': r, 'pe': pe, 'dev': dev, 'sel': use_for_sel}
        cache_files[name] = {'file': cpath, 'md5': md5(cpath)}
        print(f"  loaded {name:<24} {len(y):>6} emails  (dev {int(dev.sum())} / test {int((~dev).sum())})")
    sel_names = [n for n, d in data.items() if d['sel']]
    if len(sel_names) < 2:
        sys.exit("\nNeed at least two selection datasets with cached scores.")

    # ---------- 1. development grid ----------
    rows = []
    for pe_pct in PE_GRID:
        for T in T_GRID:
            f1s, fps, recs = [], 0, []
            for n in sel_names:
                d = data[n]; m = d['dev']
                res = prf(d['y'][m], combined(d, pe_pct)[m] >= T)
                f1s.append(res['F1']); fps += res['FP']; recs.append(res['Recall'])
            rows.append({'Perspective_%': pe_pct, 'Reasoning_%': 100 - pe_pct, 'Threshold': T,
                         'Dev_macro_F1': float(np.mean(f1s)), 'Dev_total_FP': fps, 'Dev_mean_recall': float(np.mean(recs))})
    with open(os.path.join(OUT_DIR, 'dev_grid.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    eligible = [r for r in rows if r['Perspective_%'] >= MIN_PE]
    best_f1 = max(r['Dev_macro_F1'] for r in eligible)
    tied = [r for r in eligible if r['Dev_macro_F1'] >= best_f1 - TIE_TOL]
    chosen = sorted(tied, key=lambda r: (r['Dev_total_FP'], -r['Dev_macro_F1']))[0]
    overall_best = max(rows, key=lambda r: r['Dev_macro_F1'])

    print(f"\nTop eligible settings on the development halves (Perspective weight >= {MIN_PE}%):")
    print(f"  {'Pe %':>5} {'R %':>5} {'T':>3} {'dev macro-F1':>13} {'dev FP':>7} {'dev recall':>11}")
    for r in sorted(eligible, key=lambda r: -r['Dev_macro_F1'])[:8]:
        mark = '  <-- chosen' if r is chosen else ('  (tied)' if r in tied else '')
        print(f"  {r['Perspective_%']:>5} {r['Reasoning_%']:>5} {r['Threshold']:>3} {r['Dev_macro_F1']:>12.2f}% {r['Dev_total_FP']:>7} {r['Dev_mean_recall']:>10.2f}%{mark}")
    print(f"\n  For information, best of all settings incl. Perspective 0%: "
          f"Pe={overall_best['Perspective_%']}% T={overall_best['Threshold']} -> {overall_best['Dev_macro_F1']:.2f}%")

    # ---------- 2. freeze BEFORE touching the test halves ----------
    frozen = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'weights_percent': {'pattern': 0, 'reasoning': chosen['Reasoning_%'], 'perspective': chosen['Perspective_%']},
        'threshold': chosen['Threshold'],
        'rule': (f"max mean F1 on development halves of {sel_names}; Perspective weight >= {MIN_PE}%; "
                 f"settings within {TIE_TOL}pp of best are tied; ties broken by fewer false positives"),
        'split_seed': SEED, 'dev_macro_F1_of_chosen': round(chosen['Dev_macro_F1'], 2),
        'cache_files_md5': cache_files,
    }
    fpath = os.path.join(OUT_DIR, 'frozen_config.json')
    json.dump(frozen, open(fpath, 'w'), indent=2)
    print(f"\n  FROZEN -> {fpath}")
    print(f"  Reasoning {chosen['Reasoning_%']}% + Perspective {chosen['Perspective_%']}% + Pattern 0%, threshold {chosen['Threshold']}")
    print("  The test halves have not been looked at yet. Do not edit this file.")

    # ---------- 3. test halves ----------
    cfgs = {'Selected (frozen)': (chosen['Perspective_%'], chosen['Threshold']),
            'Equal weight (earlier setting)': EQUAL_WEIGHT, 'Reasoning only': REASONING_ONLY}
    out_rows, summary = [], {k: [] for k in cfgs}
    print("\n" + "=" * 78 + "\n  HELD-OUT TEST HALVES\n" + "=" * 78)
    print(f"  {'Dataset':<24} {'configuration':<31} {'F1':>7} {'Prec':>7} {'Recall':>7} {'FP':>5}")
    for n, d in data.items():
        m = ~d['dev']
        for cname, (pe_pct, T) in cfgs.items():
            res = prf(d['y'][m], combined(d, pe_pct)[m] >= T)
            out_rows.append({'Dataset': n, 'Configuration': cname, 'Perspective_%': pe_pct, 'Threshold': T, **{k: round(v, 2) for k, v in res.items()}})
            if d['sel']: summary[cname].append(res['F1'])
            print(f"  {n:<24} {cname:<31} {res['F1']:>6.2f}% {res['Precision']:>6.2f}% {res['Recall']:>6.2f}% {res['FP']:>5}")
        print()
    print("  Mean F1 over the selection datasets (test halves):")
    for cname, v in summary.items():
        print(f"    {cname:<31} {np.mean(v):6.2f}%")

    # paired bootstrap of the macro-F1 difference
    rng = np.random.RandomState(SEED)
    def boot(cfg_a, cfg_b):
        pre = []
        for n in sel_names:
            d = data[n]; m = ~d['dev']
            pre.append((d['y'][m], combined(d, cfg_a[0])[m] >= cfg_a[1], combined(d, cfg_b[0])[m] >= cfg_b[1]))
        diffs = []
        for _ in range(N_BOOT):
            per = []
            for y, a, b in pre:
                idx = rng.randint(0, len(y), len(y))
                per.append(prf(y[idx], a[idx])['F1'] - prf(y[idx], b[idx])['F1'])
            diffs.append(np.mean(per))
        lo, hi = np.percentile(diffs, [2.5, 97.5]); return float(np.mean(diffs)), float(lo), float(hi)
    print("\n  Paired-bootstrap 95% interval for the difference in mean F1 (selected minus other):")
    boots = {}
    for other in ('Equal weight (earlier setting)', 'Reasoning only'):
        mean_d, lo, hi = boot(cfgs['Selected (frozen)'], cfgs[other]); boots[other] = (mean_d, lo, hi)
        verdict = 'interval excludes 0' if (lo > 0 or hi < 0) else 'interval includes 0 (no clear difference)'
        print(f"    vs {other:<31} {mean_d:+.2f}pp  [{lo:+.2f}, {hi:+.2f}]  {verdict}")

    with open(os.path.join(OUT_DIR, 'test_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    json.dump({k: {'diff_pp': v[0], 'ci95_low': v[1], 'ci95_high': v[2]} for k, v in boots.items()},
              open(os.path.join(OUT_DIR, 'bootstrap.json'), 'w'), indent=2)
    print(f"\n  Saved: {OUT_DIR}/dev_grid.csv, frozen_config.json, test_results.csv, bootstrap.json")
    print("  Next: python3 Confirm_On_Fresh_MeAJOR.py  (unseen data, about $1.3)\n")


if __name__ == '__main__':
    main()
