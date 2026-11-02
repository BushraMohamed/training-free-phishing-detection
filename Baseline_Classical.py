"""
Baseline_Classical.py
=============================================================================
trained baselines compared with the proposed
system on EXACTLY the same development / test halves.

What it does (no API calls, no cost, runs on a laptop):

  1. Rebuilds the development/test split of every dataset exactly as
     Select_And_Freeze_Configuration.py did (seed 2026, stratified 50/50).
     If that script is in the folder it is imported and used directly.
  2. Trains two classical baselines on each DEVELOPMENT half:
        - TF-IDF (word 1-2 grams + character 3-5 grams) + Logistic Regression
        - the same features + Linear SVM
     The regularisation strength C is chosen by 5-fold cross-validation on
     the development half only. The test half is never used for training or
     tuning.
  3. Evaluates every model once on the TEST half and compares it with the
     frozen system configuration (read from final_configuration/
     frozen_config.json and the saved agent scores), including paired-
     bootstrap 95% intervals for the F1 difference.
  4. Two extra comparisons that matter for a training-free system:
        - cross-corpus: train on one dataset, test on another
        - low-data: train on only 20, 50, 100, 200 or 500 labelled emails
  5. Reports macro- AND micro-averaged results.
  6. Counts identical emails within and between the datasets, since MeAJOR
     contains the Nazario corpus and TREC-05 (whose legitimate emails come
     from Enron).

Run it in the repository folder, which holds the six CSV files,
score_cache_reasoning_only/ and final_configuration/:

    python3 Baseline_Classical.py

Expected time: about 5-20 minutes (Enron is the slow part).
Needs: pandas, numpy, scikit-learn  (pip3 install scikit-learn)

Outputs go to baseline_results/.
"""
import os, sys, json, csv, time, hashlib, platform, warnings
from datetime import datetime
import numpy as np
import pandas as pd

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

warnings.filterwarnings('ignore')

OUT_DIR = 'baseline_results'
MAX_CHARS = 20000                     # very long emails are truncated (same for every model)
N_BOOT = 1000
LOW_DATA_SIZES = [20, 50, 100, 200, 500]
LOW_DATA_REPEATS = 5
LR_GRID = {'C': [0.1, 1.0, 10.0]}
SVM_GRID = {'C': [0.01, 0.1, 1.0]}

# ---------------------------------------------------------------------------
# Split and labels: identical to Select_And_Freeze_Configuration.py.
# The embedded copies below are used only if that script is not in the folder.
# ---------------------------------------------------------------------------
_CACHE_DIR = 'score_cache_reasoning_only'
_SEED = 2026
_DATASETS = [
    ('Enron Full',           'Enron.csv',                             'Enron_Full',           True),
    ('Nazario Full',         'Nazario.csv',                           'Nazario_Full',         False),
    ('Salloum English 300',  'Salloum_English_300.csv',               'Salloum_English_300',  True),
    ('Salloum Arabic 300',   'Salloum_Arabic_300.csv',                'Salloum_Arabic_300',   True),
    ('MeAJOR 2025 (3,000)',  'meajor_cleaned_preprocessed.csv',       'MeAJOR_2025',          True),
    ('UAE Synthetic Arabic', 'UAE_Synthetic_Arabic_Combined_600.csv', 'UAE_Synthetic_Arabic', True),
]


def _meajor_sample(df):
    en = df[df['language'] == 'en'].copy()
    en = en[en['body'].notna()]
    en = en[~en['body'].str.contains(r'<\|EMOJI\|>', na=False)]
    en = en[en['body'].str.len() > 50]
    legit = en[en['label'] == 0].sample(1500, random_state=42)
    phish = en[en['label'] == 1].sample(1500, random_state=42)
    return pd.concat([legit, phish]).sample(frac=1, random_state=42).reset_index(drop=True)


def _load_labels(name, path):
    df = pd.read_csv(path)
    if name.startswith('MeAJOR'):
        return _meajor_sample(df)['label'].astype(int).to_numpy()
    lc = next(c for c in df.columns if c.strip().lower() in ('label', 'class', 'type', 'category'))
    return df[lc].astype(int).to_numpy()


def _load_scores(cache_key, n):
    path = os.path.join(_CACHE_DIR, f'{cache_key}_scores.json')
    if not os.path.exists(path):
        return None, None, path
    cache = json.load(open(path))
    r = np.full(n, np.nan); pe = np.full(n, np.nan)
    for k, v in cache.items():
        i = int(k)
        if i < n:
            r[i] = v['r']; pe[i] = v['pe']
    return r, pe, path


def _stratified_split(y, seed):
    rng = np.random.RandomState(seed)
    dev = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]; rng.shuffle(idx); dev[idx[:len(idx) // 2]] = True
    return dev


try:
    import Select_And_Freeze_Configuration as _sel           # the script that made the split
    DATASETS, SEED, CACHE_DIR = _sel.DATASETS, _sel.SEED, _sel.CACHE_DIR
    load_labels, load_scores, stratified_split = _sel.load_labels, _sel.load_scores, _sel.stratified_split
    SPLIT_SOURCE = 'imported from Select_And_Freeze_Configuration.py'
except Exception:
    DATASETS, SEED, CACHE_DIR = _DATASETS, _SEED, _CACHE_DIR
    load_labels, load_scores, stratified_split = _load_labels, _load_scores, _stratified_split
    SPLIT_SOURCE = 'embedded copy of Select_And_Freeze_Configuration.py (seed 2026)'

# ---------------------------------------------------------------------------
TEXT_PREF = ['body', 'text', 'message (preview)', 'email_text', 'email', 'content', 'message', 'email_body']


def _is_text(series):
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def load_texts(name, path):
    df = pd.read_csv(path)
    if name.startswith('MeAJOR'):
        df = _meajor_sample(df)
    cols = {c.strip().lower(): c for c in df.columns}
    label_col = next(cols[k] for k in cols if k in ('label', 'class', 'type', 'category'))
    tcol = next((cols[k] for k in TEXT_PREF if k in cols), None)
    if tcol is None:
        cands = [c for c in df.columns if c != label_col and _is_text(df[c])]
        tcol = max(cands, key=lambda c: df[c].fillna('').astype(str).str.len().sum())
    scol = cols.get('subject')
    body = df[tcol].fillna('').astype(str)
    text = (df[scol].fillna('').astype(str) + ' \n ' + body) if scol else body
    return [t[:MAX_CHARS] for t in text.tolist()], tcol, scol


def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(y, pred):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(bool)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / len(y) if len(y) else 0.0
    return {'Accuracy': 100 * acc, 'Precision': 100 * p, 'Recall': 100 * r, 'F1': 100 * f,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def f1_only(y, pred):
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return 100.0 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def make_features(min_df=2):
    word = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=min_df, max_df=0.95,
                           sublinear_tf=True, max_features=200000, token_pattern=r'(?u)\b\w+\b')
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=min_df,
                           sublinear_tf=True, max_features=200000)
    return FeatureUnion([('word', word), ('char', char)])


def fit_models(texts_dev, y_dev):
    """Fit the features on the development half only, then tune C by 5-fold CV on it."""
    feats = make_features(min_df=2 if len(texts_dev) >= 200 else 1)
    X = feats.fit_transform(texts_dev)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {}
    for mname, est, grid in [('TF-IDF + Logistic Regression', LogisticRegression(solver='liblinear', max_iter=5000), LR_GRID),
                             ('TF-IDF + Linear SVM', LinearSVC(max_iter=20000), SVM_GRID)]:
        gs = GridSearchCV(est, grid, scoring='f1', cv=cv, n_jobs=-1, refit=True)
        gs.fit(X, y_dev)
        models[mname] = (gs.best_estimator_, gs.best_params_['C'])
    return feats, models


def paired_boot(y, pred_a, pred_b, rng, n_boot=N_BOOT):
    n = len(y); d = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        d[b] = f1_only(y[idx], pred_a[idx]) - f1_only(y[idx], pred_b[idx])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), d


def fmt_ci(m, lo, hi):
    verdict = 'clear difference' if (lo > 0 or hi < 0) else 'no clear difference'
    return f'{m:+.2f}pp [{lo:+.2f}, {hi:+.2f}] {verdict}'


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    t0 = time.time()
    print('\n' + '=' * 78)
    print('  CLASSICAL BASELINES ON THE SAME DEVELOPMENT / TEST HALVES')
    print(f'  {datetime.now():%Y-%m-%d %H:%M} | no API calls, cost $0 | split: {SPLIT_SOURCE}')
    print('=' * 78)
    os.makedirs(os.path.join(OUT_DIR, 'predictions'), exist_ok=True)

    frozen_path = os.path.join('final_configuration', 'frozen_config.json')
    if os.path.exists(frozen_path):
        fz = json.load(open(frozen_path)); SYS_PE = fz['weights_percent']['perspective']; SYS_T = fz['threshold']
        sys_src = 'final_configuration/frozen_config.json'
    else:
        SYS_PE, SYS_T = 10, 30; sys_src = 'default (Perspective 10%, threshold 30); frozen_config.json not found'
    SYS_NAME = f'Proposed system (R {100 - SYS_PE}% / Pe {SYS_PE}%, T={SYS_T})'
    print(f'  system configuration: {SYS_NAME}  [{sys_src}]\n')

    # ---------- load every dataset with the original split ----------
    data, info = {}, {'datasets': {}}
    for i, (name, csv_file, key, use_for_sel) in enumerate(DATASETS):
        if not os.path.exists(csv_file):
            print(f'  skip {name}: {csv_file} not found'); continue
        y_all = load_labels(name, csv_file)
        texts, tcol, scol = load_texts(name, csv_file)
        if len(texts) != len(y_all):
            sys.exit(f'{name}: text rows ({len(texts)}) and label rows ({len(y_all)}) differ; stopping rather than guessing.')
        r, pe, cpath = load_scores(key, len(y_all))
        if r is not None:
            ok = ~np.isnan(r) & ~np.isnan(pe)
            if ok.sum() < len(y_all):
                print(f'  NOTE {name}: {int(ok.sum())}/{len(y_all)} emails have system scores; using those, as the selection script did')
        else:
            ok = np.ones(len(y_all), dtype=bool)
            print(f'  NOTE {name}: no saved system scores ({cpath}); baselines only, split assumes all emails were scored')
        y = y_all[ok]; texts = [t for t, k in zip(texts, ok) if k]
        dev = stratified_split(y, SEED + i)
        d = {'y': y, 'texts': texts, 'dev': dev, 'sel': use_for_sel, 'key': key}
        if r is not None:
            comb = (1 - SYS_PE / 100.0) * r[ok] + (SYS_PE / 100.0) * pe[ok]
            d['sys_pred'] = comb >= SYS_T
        data[name] = d
        info['datasets'][name] = {'file': csv_file, 'md5': md5_file(csv_file), 'text_column': tcol, 'subject_column': scol,
                                  'n': int(len(y)), 'dev': int(dev.sum()), 'test': int((~dev).sum()),
                                  'phishing': int(y.sum()), 'legitimate': int((y == 0).sum())}
        print(f'  loaded {name:<24} {len(y):>6} emails (dev {int(dev.sum())} / test {int((~dev).sum())})'
              f'  text="{tcol}"' + (f' + subject="{scol}"' if scol else ''))

    if not data:
        sys.exit('\nNo dataset files found. Run this in the folder that holds the CSV files.')

    rng = np.random.RandomState(SEED)
    in_rows, boot_rows, fitted = [], [], {}
    boot_store = {}                                  # for the macro bootstrap

    # ---------- 1. in-domain ----------
    print('\n' + '-' * 78 + '\n  1. IN-DOMAIN: train on the development half, test on the test half\n' + '-' * 78)
    for name, d in data.items():
        y = d['y']; dev = d['dev']; te = ~dev
        if len(np.unique(y[dev])) < 2:
            print(f'  {name}: one class only (phishing-only corpus), so no in-domain training; used in cross-corpus tests.')
            continue
        t1 = time.time()
        tr_texts = [t for t, k in zip(d['texts'], dev) if k]
        te_texts = [t for t, k in zip(d['texts'], te) if k]
        feats, models = fit_models(tr_texts, y[dev])
        X_te = feats.transform(te_texts)
        fitted[name] = (feats, models)
        preds = {m: est.predict(X_te).astype(bool) for m, (est, _) in models.items()}
        if 'sys_pred' in d:
            preds[SYS_NAME] = d['sys_pred'][te]
        pred_df = pd.DataFrame({'test_row': np.where(te)[0], 'label': y[te]})
        for m, p in preds.items():
            res = metrics(y[te], p)
            row = {'Dataset': name, 'Model': m, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()}}
            row['Best_C'] = models[m][1] if m in models else ''
            in_rows.append(row)
            pred_df[m] = p.astype(int)
        pred_df.to_csv(os.path.join(OUT_DIR, 'predictions', f"{d['key']}_test_predictions.csv"), index=False)
        print(f'\n  {name}  (test {int(te.sum())} emails, {time.time() - t1:.0f}s)')
        print(f"    {'model':<44}{'F1':>8}{'Prec':>8}{'Recall':>8}{'FP':>6}{'FN':>6}")
        for m, p in preds.items():
            res = metrics(y[te], p)
            print(f"    {m:<44}{res['F1']:>7.2f}%{res['Precision']:>7.2f}%{res['Recall']:>7.2f}%{res['FP']:>6}{res['FN']:>6}")
        if 'sys_pred' in d:
            for m in models:
                mean_d, lo, hi, draws = paired_boot(y[te], preds[m], preds[SYS_NAME], rng)
                boot_store.setdefault(m, {})[name] = (y[te], preds[m], preds[SYS_NAME])
                boot_rows.append({'Dataset': name, 'Comparison': f'{m} minus system', 'F1_diff_pp': round(mean_d, 2),
                                  'CI95_low': round(lo, 2), 'CI95_high': round(hi, 2),
                                  'Verdict': 'clear difference' if (lo > 0 or hi < 0) else 'no clear difference'})
                print(f'    {m} minus system: {fmt_ci(mean_d, lo, hi)}')

    # ---------- macro and micro over the selection datasets ----------
    sel_names = [n for n, d in data.items() if d['sel'] and n in fitted]
    summary_rows = []
    if sel_names:
        print('\n' + '-' * 78 + f'\n  MACRO and MICRO averages over the {len(sel_names)} selection datasets (test halves)\n' + '-' * 78)
        model_names = list(fitted[sel_names[0]][1].keys()) + ([SYS_NAME] if all('sys_pred' in data[n] for n in sel_names) else [])
        for m in model_names:
            rows_m = [r for r in in_rows if r['Model'] == m and r['Dataset'] in sel_names]
            macro = {k: float(np.mean([r[k] for r in rows_m])) for k in ('Accuracy', 'Precision', 'Recall', 'F1')}
            tp = sum(r['TP'] for r in rows_m); fp = sum(r['FP'] for r in rows_m)
            fn = sum(r['FN'] for r in rows_m); tn = sum(r['TN'] for r in rows_m)
            p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
            micro_f1 = 100 * 2 * p * rc / (p + rc) if p + rc else 0
            summary_rows.append({'Model': m, 'Macro_F1': round(macro['F1'], 2), 'Macro_Precision': round(macro['Precision'], 2),
                                 'Macro_Recall': round(macro['Recall'], 2), 'Macro_Accuracy': round(macro['Accuracy'], 2),
                                 'Micro_F1': round(micro_f1, 2), 'Micro_Precision': round(100 * p, 2), 'Micro_Recall': round(100 * rc, 2),
                                 'Total_TP': tp, 'Total_FP': fp, 'Total_FN': fn, 'Total_TN': tn})
            print(f"    {m:<44} macro-F1 {macro['F1']:6.2f}%   micro-F1 {micro_f1:6.2f}%   total FP {fp}")
        for m, per in boot_store.items():
            names = [n for n in sel_names if n in per]
            if len(names) < 2:
                continue
            diffs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                vals = []
                for n in names:
                    y_, a_, s_ = per[n]; idx = rng.randint(0, len(y_), len(y_))
                    vals.append(f1_only(y_[idx], a_[idx]) - f1_only(y_[idx], s_[idx]))
                diffs[b] = np.mean(vals)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            boot_rows.append({'Dataset': 'MACRO (selection datasets)', 'Comparison': f'{m} minus system',
                              'F1_diff_pp': round(float(diffs.mean()), 2), 'CI95_low': round(float(lo), 2),
                              'CI95_high': round(float(hi), 2), 'Verdict': 'clear difference' if (lo > 0 or hi < 0) else 'no clear difference'})
            print(f'    macro {m} minus system: {fmt_ci(float(diffs.mean()), float(lo), float(hi))}')

    # ---------- 2. cross-corpus ----------
    print('\n' + '-' * 78 + '\n  2. CROSS-CORPUS: train on one dataset, test on another (system needs no training)\n' + '-' * 78)
    pairs = [('Enron Full', 'MeAJOR 2025 (3,000)'), ('MeAJOR 2025 (3,000)', 'Enron Full'),
             ('Enron Full', 'Salloum English 300'), ('MeAJOR 2025 (3,000)', 'Salloum English 300'),
             ('Enron Full', 'Nazario Full'),
             ('Salloum Arabic 300', 'UAE Synthetic Arabic'), ('UAE Synthetic Arabic', 'Salloum Arabic 300')]
    overlap_note = {('MeAJOR 2025 (3,000)', 'Enron Full'): 'MeAJOR includes TREC-05, whose legitimate emails come from Enron: possible overlap',
                    ('Enron Full', 'MeAJOR 2025 (3,000)'): 'MeAJOR includes TREC-05, whose legitimate emails come from Enron: possible overlap'}
    cross_rows = []
    for src, tgt in pairs:
        if src not in fitted or tgt not in data:
            continue
        feats, models = fitted[src]; dt = data[tgt]; te = ~dt['dev']
        X_te = feats.transform([t for t, k in zip(dt['texts'], te) if k])
        recall_only = len(np.unique(dt['y'])) < 2
        print(f'\n  train {src}  ->  test {tgt}' + ('  (phishing-only: recall is the measure)' if recall_only else ''))
        entries = [(m, est.predict(X_te).astype(bool)) for m, (est, _) in models.items()]
        if 'sys_pred' in dt:
            entries.append((SYS_NAME + ' [no training]', dt['sys_pred'][te]))
        for m, p in entries:
            res = metrics(dt['y'][te], p)
            cross_rows.append({'Train_on': src, 'Test_on': tgt, 'Model': m,
                               **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()},
                               'Note': ('phishing-only target: report Recall' if recall_only else '') +
                                       ('; ' if recall_only and (src, tgt) in overlap_note else '') + overlap_note.get((src, tgt), '')})
            key_metric = f"Recall {res['Recall']:6.2f}%" if recall_only else f"F1 {res['F1']:6.2f}%"
            print(f'    {m:<52} {key_metric}   FP {res["FP"]}')

    # ---------- 3. low-data ----------
    print('\n' + '-' * 78 + '\n  3. LOW-DATA: Logistic Regression trained on only n labelled emails (5 repeats)\n' + '-' * 78)
    low_rows = []
    for name in sel_names:
        d = data[name]; y = d['y']; dev_idx = np.where(d['dev'])[0]; te = ~d['dev']
        te_texts = [t for t, k in zip(d['texts'], te) if k]
        sys_f1 = metrics(y[te], d['sys_pred'][te])['F1'] if 'sys_pred' in d else None
        line = f'  {name:<24}'
        for n_tr in LOW_DATA_SIZES:
            per_class = n_tr // 2
            pos = dev_idx[y[dev_idx] == 1]; neg = dev_idx[y[dev_idx] == 0]
            if per_class > len(pos) or per_class > len(neg):
                continue
            f1s = []
            for rep in range(LOW_DATA_REPEATS):
                rr = np.random.RandomState(SEED * 100 + rep)
                idx = np.concatenate([rr.choice(pos, per_class, replace=False), rr.choice(neg, per_class, replace=False)])
                feats = make_features(min_df=1)
                X = feats.fit_transform([d['texts'][i] for i in idx])
                clf = LogisticRegression(solver='liblinear', C=10.0, max_iter=5000).fit(X, y[idx])
                f1s.append(metrics(y[te], clf.predict(feats.transform(te_texts)).astype(bool))['F1'])
            low_rows.append({'Dataset': name, 'Train_emails': n_tr, 'Mean_F1': round(float(np.mean(f1s)), 2),
                             'SD_F1': round(float(np.std(f1s, ddof=1)), 2), 'Repeats': LOW_DATA_REPEATS,
                             'System_F1_no_training': round(sys_f1, 2) if sys_f1 is not None else ''})
            line += f'  n={n_tr}: {np.mean(f1s):5.1f}±{np.std(f1s, ddof=1):3.1f}'
        if sys_f1 is not None:
            line += f'   | system (no training): {sys_f1:5.1f}'
        print(line)

    # ---------- 4. overlap between datasets (feedback: are the corpora independent?) ----------
    print('\n' + '-' * 78 + '\n  4. OVERLAP CHECK: identical emails shared between datasets (normalised text)\n' + '-' * 78)

    def norm(t):
        return ' '.join(str(t).lower().split())[:1000]
    fps = {n: {} for n in data}
    for n, d in data.items():
        for j, t in enumerate(d['texts']):
            h = hashlib.md5(norm(t).encode('utf-8')).hexdigest()
            fps[n].setdefault(h, []).append(j)
    overlap_rows = []
    names = list(data)
    for a_i, a in enumerate(names):
        dup_within = sum(len(v) - 1 for v in fps[a].values() if len(v) > 1)
        overlap_rows.append({'Dataset_A': a, 'Dataset_B': '(within itself)', 'Shared_emails': dup_within,
                             'Share_of_A_%': round(100 * dup_within / len(data[a]['texts']), 2)})
        if dup_within:
            print(f'  {a:<24} duplicates within the dataset: {dup_within}')
        for b in names[a_i + 1:]:
            shared = set(fps[a]) & set(fps[b])
            n_a = sum(len(fps[a][h]) for h in shared)
            overlap_rows.append({'Dataset_A': a, 'Dataset_B': b, 'Shared_emails': n_a,
                                 'Share_of_A_%': round(100 * n_a / len(data[a]['texts']), 2)})
            if n_a:
                print(f'  {a:<24} and {b:<24} share {n_a} emails ({100 * n_a / len(data[a]["texts"]):.2f}% of {a})')
    if not any(r['Shared_emails'] for r in overlap_rows):
        print('  no identical emails found within or between the datasets')

    # ---------- save ----------
    write_csv(os.path.join(OUT_DIR, 'dataset_overlap.csv'), overlap_rows)
    write_csv(os.path.join(OUT_DIR, 'in_domain_results.csv'), in_rows)
    write_csv(os.path.join(OUT_DIR, 'summary_macro_micro.csv'), summary_rows)
    write_csv(os.path.join(OUT_DIR, 'bootstrap_vs_system.csv'), boot_rows)
    write_csv(os.path.join(OUT_DIR, 'cross_corpus_results.csv'), cross_rows)
    write_csv(os.path.join(OUT_DIR, 'low_data_results.csv'), low_rows)
    info.update({'created': datetime.now().isoformat(timespec='seconds'), 'split_source': SPLIT_SOURCE, 'seed': SEED,
                 'system_configuration': {'perspective_pct': SYS_PE, 'threshold': SYS_T, 'source': sys_src},
                 'features': 'TF-IDF word 1-2 grams + char_wb 3-5 grams, sublinear tf, fitted on the development half only',
                 'tuning': 'C by 5-fold stratified CV (F1) on the development half; LR C in %s, SVM C in %s' % (LR_GRID['C'], SVM_GRID['C']),
                 'max_chars': MAX_CHARS, 'bootstrap_resamples': N_BOOT,
                 'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                              'numpy': np.__version__, 'pandas': pd.__version__},
                 'runtime_minutes': round((time.time() - t0) / 60, 1)})
    json.dump(info, open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
    print('\n' + '=' * 78)
    print(f'  Saved to {OUT_DIR}/ : in_domain_results.csv, summary_macro_micro.csv, bootstrap_vs_system.csv,')
    print('  cross_corpus_results.csv, low_data_results.csv, dataset_overlap.csv, predictions/, run_info.json')
    print(f'  Finished in {(time.time() - t0) / 60:.1f} minutes.\n')


if __name__ == '__main__':
    main()
