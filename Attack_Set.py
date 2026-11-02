"""
Attack_Set.py
=============================================================================
Rebuilds EXACTLY the emails and attack conditions used by Robustness_Test.py,
so that other models (the trained baselines) can be tested on the same data.

    items, conds = build_attack_set()

items : one dict per sampled test-half email (dataset, row, label, lang, subject, sender, text)
conds : one dict per item, condition name -> attacked email body
        ('Clean', 'A1 visible injection', 'A1 visible injection + sanitiser', ...,
         'A4 paraphrase' when robustness_results/paraphrase_cache.json is present)

No API calls. The paraphrases are read from the cache written by Robustness_Test.py.
"""
import os, json
import numpy as np

import Baseline_Classical as bc
import Robustness_Test as rt

PARA_CACHE = os.path.join('robustness_results', 'paraphrase_cache.json')


def build_attack_set(verbose=True):
    items = []
    for name, n_per_class in rt.SAMPLES:
        i = next(k for k, ds in enumerate(bc.DATASETS) if ds[0] == name)
        csv_file = bc.DATASETS[i][1]
        if not os.path.exists(csv_file):
            if verbose:
                print(f'  skip {name}: {csv_file} not found')
            continue
        y = bc.load_labels(name, csv_file)
        body, subj, send = rt.load_fields(name, csv_file)
        r, pe, _ = bc.load_scores(bc.DATASETS[i][2], len(y))
        ok = (~np.isnan(r) & ~np.isnan(pe)) if r is not None else np.ones(len(y), dtype=bool)
        rows = np.where(ok)[0]; yk = y[ok]
        test = rows[~bc.stratified_split(yk, bc.SEED + i)]
        rng = np.random.RandomState(bc.SEED)                       # identical to Robustness_Test.py
        for cls in (1, 0):
            pool = test[y[test] == cls]
            pick = rng.choice(pool, min(n_per_class, len(pool)), replace=False)
            for j in sorted(pick):
                items.append({'dataset': name, 'dataset_index': i, 'row': int(j), 'label': cls, 'text': body[j],
                              'subject': subj[j], 'sender': send[j],
                              'lang': 'ar' if rt.is_arabic(body[j] + subj[j]) else 'en'})
    para = json.load(open(PARA_CACHE, encoding='utf-8')) if os.path.exists(PARA_CACHE) else {}
    conds = []
    for it in items:
        c = {'Clean': it['text']}
        if it['label'] == 1:
            L = it['lang']
            attacked = {'A1 visible injection': it['text'] + rt.INJ[L], 'A2 hidden injection': rt.HIDDEN[L] + it['text'],
                        'A3 benign stuffing': it['text'] + rt.BENIGN[L]}
            for k, t in attacked.items():
                c[k] = t
                c[k + ' + sanitiser'] = rt.sanitize(t)
            p = para.get(rt.keyof(it['text'], '', ''))
            if p:
                c['A4 paraphrase'] = p
        conds.append(c)
    if verbose:
        n_para = sum('A4 paraphrase' in c for c in conds)
        print(f'  attack set: {len(items)} emails ({sum(it["label"] for it in items)} phishing), '
              f'{n_para} paraphrases loaded from {PARA_CACHE if para else "(no paraphrase cache found)"}')
    return items, conds


def check_against_robustness_results(items):
    """Confirms the rebuilt sample is the one Robustness_Test.py scored."""
    path = os.path.join('robustness_results', 'per_email_scores.csv')
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    old = set(map(tuple, df[df['Condition'] == 'Clean'][['Dataset', 'Row']].itertuples(index=False, name=None)))
    new = {(it['dataset'], it['row']) for it in items}
    return old == new


def summarize(preds, items, conds, n_boot=1000):
    """preds: {(model, item index, condition): 0/1}. Recall per condition, drop from clean with a paired-bootstrap 95% CI."""
    import pandas as pd
    rng = np.random.RandomState(bc.SEED)
    model_names = list(dict.fromkeys(m for m, _, _ in preds))
    cond_names = list(dict.fromkeys(c for cs in conds for c in cs))
    rows = []
    for ds in list(dict.fromkeys(it['dataset'] for it in items)) + ['ALL']:
        ph = [k for k, it in enumerate(items) if it['label'] == 1 and (ds == 'ALL' or it['dataset'] == ds)]
        lg = [k for k, it in enumerate(items) if it['label'] == 0 and (ds == 'ALL' or it['dataset'] == ds)]
        for m in model_names:
            if not ph or not all((m, k, 'Clean') in preds for k in ph):
                continue
            fp = int(sum(preds.get((m, k, 'Clean'), 0) for k in lg))
            for c in cond_names:
                ks = [k for k in ph if (m, k, c) in preds]
                if not ks:
                    continue
                a = np.array([preds[(m, k, 'Clean')] for k in ks]); b = np.array([preds[(m, k, c)] for k in ks])
                n = len(ks)
                if c == 'Clean':
                    lo = hi = 0.0
                else:
                    draws = [(a[ix].mean() - b[ix].mean()) * 100 for ix in (rng.randint(0, n, n) for _ in range(n_boot))]
                    lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append({'Dataset': ds, 'Model': m, 'Condition': c, 'Phishing_emails': n,
                             'Recall_%': round(100 * b.mean(), 2), 'Recall_drop_pp': round(100 * (a.mean() - b.mean()), 2),
                             'Drop_CI95_low': round(float(lo), 2), 'Drop_CI95_high': round(float(hi), 2),
                             'FP_on_clean_legitimate': fp, 'Legitimate_emails': len(lg)})
    return pd.DataFrame(rows)


def print_summary(df):
    for ds in list(dict.fromkeys(df['Dataset'])):
        sub = df[df['Dataset'] == ds]
        models = list(dict.fromkeys(sub['Model']))
        print('\n' + '-' * 78 + f'\n  {ds}: recall % (change from clean, points)\n' + '-' * 78)
        print('  ' + f"{'condition':<36}" + ''.join(f'{m[:21]:>24}' for m in models))
        for c in dict.fromkeys(sub['Condition']):
            line = '  ' + f'{c:<36}'
            for m in models:
                r = sub[(sub['Model'] == m) & (sub['Condition'] == c)]
                line += f"{r['Recall_%'].iloc[0]:>14.1f}% ({-r['Recall_drop_pp'].iloc[0]:+6.1f})" if len(r) else f'{"-":>24}'
            print(line)
        fps = sub.groupby('Model', sort=False)['FP_on_clean_legitimate'].first()
        print('  false positives on the clean legitimate emails: ' + ', '.join(f'{m}: {v}' for m, v in fps.items()))
