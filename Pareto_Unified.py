"""
Pareto_Unified.py
=============================================================================
re-run the agent-combination Pareto analysis
(Study 1, configurations C1-C7) on ONE code version, so that every study in the
paper uses the same final code.

No API calls, no cost, about 2-5 minutes.

What it uses
  - Reasoning and Perspective scores: score_cache_reasoning_only/ (the same
    final-code scores used for selection, held-out test and baselines).
  - Pattern scores: the corrected BM25 Pattern Agent (email = document, keyword
    lists and k1/b chosen on the DEVELOPMENT halves only), rebuilt exactly as in
    Pattern_BM25_TFIDF_Bilingual.py (same seed, same lists).
  - Evaluation: the TEST halves only (seed 2026), threshold T = 30, so no email
    used to build the keyword lists is scored.

Configurations (Pattern / Reasoning / Perspective weight, %)
  C1 100/0/0   C2 0/100/0   C3 0/0/100   C4 50/50/0   C5 50/0/50   C6 0/50/50   C7 33/33/34

Run in the repository folder (needs the six CSV files,
score_cache_reasoning_only/, Baseline_Classical.py and
Pattern_BM25_TFIDF_Bilingual.py):

    python3 Pareto_Unified.py

Results: pareto_unified_results/
"""
import os, json, time, platform
from datetime import datetime
import numpy as np

import Baseline_Classical as bc
import Pattern_BM25_TFIDF_Bilingual as pb

OUT_DIR = 'pareto_unified_results'
T = 30
COST_PER_1000 = 0.42          # GPT-4o-mini, only when the Reasoning weight is above zero
CONFIGS = [('C1', 'Pattern only', 100, 0, 0), ('C2', 'Reasoning only', 0, 100, 0), ('C3', 'Perspective only', 0, 0, 100),
           ('C4', 'Pattern + Reasoning', 50, 50, 0), ('C5', 'Pattern + Perspective', 50, 0, 50),
           ('C6', 'Reasoning + Perspective', 0, 50, 50), ('C7', 'All three, equal', 33, 33, 34)]


def pareto_front(rows, key='F1'):
    front = set()
    for a in rows:
        dominated = any((b[key] >= a[key] and b['Cost_per_1000'] <= a['Cost_per_1000'])
                        and (b[key] > a[key] or b['Cost_per_1000'] < a['Cost_per_1000']) for b in rows)
        if not dominated:
            front.add(a['Config'])
    return front


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '=' * 78)
    print('  PARETO ANALYSIS ON ONE CODE VERSION  |  no API calls')
    print(f'  {datetime.now():%Y-%m-%d %H:%M} | split: {bc.SPLIT_SOURCE} | T = {T}')
    print('=' * 78)

    # ---------- data (same loading as Pattern_BM25_TFIDF_Bilingual.py)
    data = {}
    for i, (name, f, key, _) in enumerate(bc.DATASETS):
        if not os.path.exists(f):
            continue
        y_all = bc.load_labels(name, f); texts, _, _ = bc.load_texts(name, f)
        r, pe, _ = bc.load_scores(key, len(y_all))
        if r is None:
            print(f'  skip {name}: no saved agent scores'); continue
        ok = ~np.isnan(r) & ~np.isnan(pe)
        y = y_all[ok]
        data[name] = {'y': y, 'texts': [t for t, k in zip(texts, ok) if k], 'dev': bc.stratified_split(y, bc.SEED + i),
                      'r': r[ok], 'pe': pe[ok]}
        print(f'  loaded {name:<24} {len(y):>6} emails ({int((~data[name]["dev"]).sum())} in the test half)')

    # ---------- Pattern scores: rebuild the corrected BM25 agent exactly as in Pattern_BM25_TFIDF_Bilingual.py
    rng = np.random.RandomState(bc.SEED)
    P, info = {}, {}
    for lang, L in pb.LANGS.items():
        names = [n for n in L['datasets'] if n in data]
        if not names:
            continue
        pool_t, pool_y = [], []
        for n in names:
            d = data[n]; dev_idx = np.where(d['dev'])[0]
            for cls in (1, 0):
                ids = dev_idx[d['y'][dev_idx] == cls]
                pick = rng.choice(ids, min(pb.MAX_PER_CLASS, len(ids)), replace=False) if len(ids) else []
                pool_t += [d['texts'][j] for j in pick]; pool_y += [cls] * len(pick)
        pool_y = np.array(pool_y)
        ph, lg = pb.select_terms(pool_t, pool_y, L['token'])
        sc = pb.BM25Pattern(ph, lg, pool_t, lang)
        grid = []
        for k1 in pb.K1_GRID:
            for b in pb.B_GRID:
                aucs = [pb.safe_auc(data[n]['y'][data[n]['dev']], sc.raw([t for t, k in zip(data[n]['texts'], data[n]['dev']) if k], k1, b)) for n in names]
                grid.append((float(np.nanmean(aucs)), k1, b))
        _, k1, b = max(grid)
        dev_raw = np.concatenate([sc.raw([t for t, k in zip(data[n]['texts'], data[n]['dev']) if k], k1, b) for n in names])
        s = float(np.std(dev_raw)) or 1.0
        for n in names + [m for m in L['recall_only'] if m in data]:
            P[n] = pb.to_score(sc.raw(data[n]['texts'], k1, b), s)
        info[lang] = {'k1': k1, 'b': b, 'phishing_terms': ph, 'legitimate_terms': lg}
        print(f'  {lang}: BM25 k1={k1}, b={b}; {len(ph)} + {len(lg)} terms from {len(pool_y)} development emails')
        # consume the random numbers that Pattern_BM25_TFIDF_Bilingual.py uses for its bootstrap here,
        # so that the next language gets exactly the same keyword lists as in that script
        for _scorer in ('BM25', 'TF-IDF'):
            for _w in (3, 10):
                for _ in range(pb.N_BOOT):
                    for n in names:
                        m = int((~data[n]['dev']).sum()); rng.randint(0, m, m)

    ref = os.path.join('pattern_bilingual_results', 'term_lists.json')
    if os.path.exists(ref):
        old = json.load(open(ref, encoding='utf-8'))
        for lang in info:
            same = (old.get(lang, {}).get('phishing_terms') == info[lang]['phishing_terms']
                    and old.get(lang, {}).get('legitimate_terms') == info[lang]['legitimate_terms'])
            print(f'  {lang} keyword lists identical to pattern_bilingual_results: {"yes" if same else "NO"}')
            info[lang]['identical_to_pattern_bilingual_results'] = same

    # ---------- evaluate C1-C7 on the TEST halves
    rows = []
    for name, d in data.items():
        if name not in P:
            continue
        te = ~d['dev']; y = d['y'][te]
        for cid, label, wp, wr, wpe in CONFIGS:
            score = (wp * P[name] + wr * d['r'] + wpe * d['pe'])[te] / 100.0
            m = bc.metrics(y, score >= T)
            rows.append({'Dataset': name, 'Config': cid, 'Agents': label, 'Weights_P_R_Pe': f'{wp}/{wr}/{wpe}',
                         'Cost_per_1000': COST_PER_1000 if wr > 0 else 0.0,
                         'F1': round(m['F1'], 2), 'Precision': round(m['Precision'], 2), 'Recall': round(m['Recall'], 2),
                         'FP': m['FP'], 'FN': m['FN'], 'TP': m['TP'], 'TN': m['TN'],
                         'AUC': round(pb.safe_auc(y, score), 4) if len(np.unique(y)) == 2 else '',
                         'Test_emails': int(te.sum()),
                         'Note': 'phishing-only: use Recall' if name.startswith('Nazario') else ''})
    labelled = [n for n in data if n in P and not n.startswith('Nazario')]
    summary = []
    for cid, label, wp, wr, wpe in CONFIGS:
        sub = [r for r in rows if r['Config'] == cid and r['Dataset'] in labelled]
        naz = next((r for r in rows if r['Config'] == cid and r['Dataset'].startswith('Nazario')), None)
        Y = sum(r['TP'] + r['FN'] for r in sub); tp = sum(r['TP'] for r in sub); fp = sum(r['FP'] for r in sub)
        summary.append({'Config': cid, 'Agents': label, 'Weights_P_R_Pe': f'{wp}/{wr}/{wpe}', 'Cost_per_1000': COST_PER_1000 if wr > 0 else 0.0,
                        'F1': round(float(np.mean([r['F1'] for r in sub])), 2),        # macro F1 (key 'F1' for the Pareto front)
                        'Micro_F1': round(100 * 2 * tp / (tp + fp + Y), 2) if (tp + fp + Y) else 0.0,
                        'Mean_recall': round(float(np.mean([r['Recall'] for r in sub])), 2),
                        'Total_FP': fp, 'Nazario_recall': naz['Recall'] if naz else ''})
    enron = [r for r in rows if r['Dataset'] == 'Enron Full']
    fe = pareto_front(enron) if enron else set(); fm = pareto_front(summary)
    for r in enron:
        r['Pareto_front'] = '★' if r['Config'] in fe else ''
    for r in summary:
        r['Pareto_front'] = '★' if r['Config'] in fm else ''

    # ---------- print
    if enron:
        print('\n' + '-' * 78 + f'\n  ENRON TEST HALF ({enron[0]["Test_emails"]:,} emails), T = {T}\n' + '-' * 78)
        print(f"  {'':4}{'agents':<26}{'P/R/Pe':>10}{'cost':>7}{'F1':>8}{'Prec':>8}{'Rec':>8}{'FP':>7}{'FN':>7}{'AUC':>8}")
        for r in enron:
            print(f"  {r['Config']:<4}{r['Agents']:<26}{r['Weights_P_R_Pe']:>10}{r['Cost_per_1000']:>7.2f}{r['F1']:>7.2f}%{r['Precision']:>7.2f}%"
                  f"{r['Recall']:>7.2f}%{r['FP']:>7}{r['FN']:>7}{r['AUC']:>8} {r['Pareto_front']}")
    print('\n' + '-' * 78 + f'\n  ALL FIVE LABELLED TEST HALVES: macro F1, micro F1, total FP; Nazario recall\n' + '-' * 78)
    for r in summary:
        print(f"  {r['Config']:<4}{r['Agents']:<26}{r['Weights_P_R_Pe']:>10}  macro {r['F1']:6.2f}%  micro {r['Micro_F1']:6.2f}%  "
              f"FP {r['Total_FP']:>5}  Nazario {r['Nazario_recall']}% {r['Pareto_front']}")

    bc.write_csv(os.path.join(OUT_DIR, 'pareto_enron_test_half.csv'), enron)
    bc.write_csv(os.path.join(OUT_DIR, 'pareto_all_datasets.csv'), rows)
    bc.write_csv(os.path.join(OUT_DIR, 'pareto_summary_macro_micro.csv'), summary)
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'seed': bc.SEED, 'split_source': bc.SPLIT_SOURCE,
               'threshold': T, 'configs': CONFIGS, 'pattern_agent': info, 'evaluated_on': 'test halves only',
               'python': platform.python_version(), 'runtime_minutes': round((time.time() - t0) / 60, 1)},
              open(os.path.join(OUT_DIR, 'run_info.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n  Saved to {OUT_DIR}/ ({(time.time() - t0) / 60:.1f} minutes).\n')


if __name__ == '__main__':
    main()
