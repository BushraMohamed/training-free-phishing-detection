"""
Pattern_BM25_TFIDF_Bilingual.py
=============================================================================
BM25 and TF-IDF keyword scoring for
BOTH English and Arabic, built and compared in exactly the same way.

No API calls, no cost. It reuses the saved Reasoning and Perspective scores
(score_cache_reasoning_only/) and the same development / test halves as the
final configuration (seed 2026).

Design (identical for both languages)
  1. Keyword lists: for each language, the 50 terms most typical of phishing
     and the 50 most typical of legitimate email, chosen by the log-odds of
     their document frequency in the DEVELOPMENT halves only (balanced: at most
     300 emails per class per dataset, so Enron does not dominate).
     Test emails are never used to build lists or tune anything.
  2. BM25 (Robertson et al., 1994) in its standard form: the EMAIL is the
     document and each keyword list is a query.
        P_raw = BM25(email, phishing terms) - BM25(email, legitimate terms)
     k1 (term-frequency saturation) and b (email-length normalisation) are
     tuned per language on the development halves (pattern-alone AUC).
  3. TF-IDF: cosine(email, phishing terms) - cosine(email, legitimate terms),
     with IDF from the same development emails.
  4. Both raw differences are mapped to 0-100 with the same logistic rule,
     P = 100 / (1 + exp(-raw / s)), s = standard deviation of raw on the
     development halves of that language.
  5. Weight sweep: Pattern weight 0%, 3%, 10%; the remaining weight keeps the
     frozen Reasoning:Perspective ratio (90:10):
        score = w_P x P + (1 - w_P) x (0.9 R + 0.1 Pe)
     reported (a) at the frozen threshold T=30 and (b) with T chosen jointly on
     the development halves from {20,25,30,35,40}.
  6. Diagnostic: the previous Arabic BM25 set-up (keywords as one-word
     documents; lists built from the whole Salloum Arabic file when it was in
     the folder) is re-run, to show how much the earlier result was affected.

Run in the repository folder (needs the six CSV files, score_cache_reasoning_only/,
final_configuration/, Baseline_Classical.py; AI_Phishing_Detection_System_
v50_BM25.py is optional, only for the diagnostic):

    pip3 install rank_bm25 scikit-learn
    python3 Pattern_BM25_TFIDF_Bilingual.py

About 3-10 minutes. Results: pattern_bilingual_results/
"""
import os, re, ast, json, csv, math, time, platform
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

import Baseline_Classical as bc

OUT_DIR = 'pattern_bilingual_results'
N_TERMS = 50
MAX_PER_CLASS = 300
MIN_DF = 5
K1_GRID = [0.5, 0.9, 1.2, 1.5, 1.7, 2.0]
B_GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
PATTERN_WEIGHTS = [0, 3, 10]
T_GRID = [20, 25, 30, 35, 40]
N_BOOT = 1000
LANGS = {
    'English': {'datasets': ['Enron Full', 'Salloum English 300', 'MeAJOR 2025 (3,000)'], 'recall_only': ['Nazario Full'],
                'token': r'(?u)\b[a-z]{3,}\b', 'all_tokens': r'[a-z]+'},
    'Arabic': {'datasets': ['Salloum Arabic 300', 'UAE Synthetic Arabic'], 'recall_only': [],
               'token': r'[؀-ۿ]{3,}', 'all_tokens': r'[؀-ۿ]+'},
}


# --------------------------------------------------------------------------- helpers
def doc_lengths(texts, pattern):
    rx = re.compile(pattern)
    return np.array([len(rx.findall(t.lower())) for t in texts], dtype=float)


def select_terms(texts, y, token_pattern, n_terms=N_TERMS, min_df=MIN_DF):
    cv = CountVectorizer(token_pattern=token_pattern, lowercase=True, binary=True, min_df=min_df)
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    df_ph = np.asarray(X[y == 1].sum(0)).ravel(); df_lg = np.asarray(X[y == 0].sum(0)).ravel()
    n_ph, n_lg = max(1, int((y == 1).sum())), max(1, int((y == 0).sum()))
    lo = np.log((df_ph + 1) / (n_ph + 2)) - np.log((df_lg + 1) / (n_lg + 2))
    order = np.argsort(-lo)
    ph = [vocab[i] for i in order[:n_terms] if lo[i] > 0]
    lg = [vocab[i] for i in order[::-1][:n_terms] if lo[i] < 0]
    return ph, lg


class BM25Pattern:
    """Standard BM25: email = document, keyword list = query."""

    def __init__(self, ph_terms, lg_terms, dev_texts, lang):
        self.lang = LANGS[lang]
        self.vocab = list(dict.fromkeys(ph_terms + lg_terms))
        self.cv = CountVectorizer(vocabulary=self.vocab, token_pattern=self.lang['token'], lowercase=True)
        self.is_ph = np.array([t in set(ph_terms) for t in self.vocab]); self.is_lg = np.array([t in set(lg_terms) for t in self.vocab])
        tf_dev = self.cv.transform(dev_texts)
        n = tf_dev.shape[0]; nq = np.asarray((tf_dev > 0).sum(0)).ravel()
        self.idf = np.log((n - nq + 0.5) / (nq + 0.5) + 1.0)            # non-negative BM25 IDF
        self.avgdl = max(1.0, float(doc_lengths(dev_texts, self.lang['all_tokens']).mean()))

    def raw(self, texts, k1, b):
        tf = self.cv.transform(texts).tocsr().astype(float)
        dl = doc_lengths(texts, self.lang['all_tokens'])
        norm = k1 * (1 - b + b * dl / self.avgdl)                         # one value per email
        tf = tf.tocoo()
        val = self.idf[tf.col] * tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        M = sparse.csr_matrix((val, (tf.row, tf.col)), shape=(len(texts), len(self.vocab)))
        return np.asarray(M[:, self.is_ph].sum(1)).ravel() - np.asarray(M[:, self.is_lg].sum(1)).ravel()


class TFIDFPattern:
    def __init__(self, ph_terms, lg_terms, dev_texts, lang):
        self.lang = LANGS[lang]
        self.vocab = list(dict.fromkeys(ph_terms + lg_terms))
        self.cv = CountVectorizer(vocabulary=self.vocab, token_pattern=self.lang['token'], lowercase=True)
        tf_dev = self.cv.transform(dev_texts); n = tf_dev.shape[0]; dfq = np.asarray((tf_dev > 0).sum(0)).ravel()
        self.idf = np.log((1 + n) / (1 + dfq)) + 1.0
        self.q_ph = np.array([1.0 if t in set(ph_terms) else 0.0 for t in self.vocab]) * self.idf
        self.q_lg = np.array([1.0 if t in set(lg_terms) else 0.0 for t in self.vocab]) * self.idf
        self.q_ph /= np.linalg.norm(self.q_ph) or 1; self.q_lg /= np.linalg.norm(self.q_lg) or 1

    def raw(self, texts, *_):
        tf = self.cv.transform(texts).astype(float)
        tf.data = 1 + np.log(tf.data)                                    # sublinear tf
        X = tf.multiply(self.idf).tocsr()
        nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nrm[nrm == 0] = 1
        return (X @ self.q_ph) / nrm - (X @ self.q_lg) / nrm


def to_score(raw, s):
    return 100.0 / (1.0 + np.exp(-np.clip(raw / s, -50, 50)))


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float('nan')


# --------------------------------------------------------------- previous Arabic set-up (diagnostic)
def previous_arabic_scorer(ph_docs, lg_docs, k1=1.7, b=0.9):
    from rank_bm25 import BM25Okapi
    pb, lb = BM25Okapi(ph_docs, k1=k1, b=b), BM25Okapi(lg_docs, k1=k1, b=b)

    def score(text):
        toks = re.sub('[^؀-ۿ ]', ' ', str(text)).split()
        if not toks:
            return 0.0
        diff = float(sum(pb.get_scores(toks))) - float(sum(lb.get_scores(toks)))
        return min(100.0, max(0.0, 100.0 / (1.0 + math.exp(-diff * 0.5))))
    return score


def previous_lists_from_file(path):
    """The earlier code's lists: top-50 most frequent Arabic words of each class in the WHOLE file."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    df = pd.read_csv(path)
    lcol = next(c for c in df.columns if c.lower() in ['label', 'class', 'type', 'category'])
    tcol = next(c for c in df.columns if c.lower() in ['body', 'text', 'email', 'content', 'message', 'message (preview)'])

    def top50(texts):
        vec = TfidfVectorizer(max_features=50, min_df=2, token_pattern='[؀-ۿ]{3,}')
        vec.fit([str(t) for t in texts]); return [[w] for w in vec.get_feature_names_out()]
    pos = df[df[lcol].astype(str).str.lower().isin(['1', 'phishing', 'spam'])][tcol].tolist()
    neg = df[df[lcol].astype(str).str.lower().isin(['0', 'legitimate', 'ham'])][tcol].tolist()
    return top50(pos), top50(neg)


def previous_fallback_lists(bm25_file):
    tree = ast.parse(open(bm25_file, encoding='utf-8').read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'ArabicBM25PatternAgent':
            for st in node.body:
                if isinstance(st, ast.Assign) and getattr(st.targets[0], 'id', '') in ('FALLBACK_PHISHING', 'FALLBACK_LEGIT'):
                    out[st.targets[0].id] = ast.literal_eval(st.value)
    return out.get('FALLBACK_PHISHING'), out.get('FALLBACK_LEGIT')


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '=' * 78)
    print('  PATTERN AGENT: BM25 AND TF-IDF FOR ENGLISH AND ARABIC')
    print(f'  {datetime.now():%Y-%m-%d %H:%M} | no API calls | split: {bc.SPLIT_SOURCE}')
    print('=' * 78)
    fz_path = os.path.join('final_configuration', 'frozen_config.json')
    fz = json.load(open(fz_path)) if os.path.exists(fz_path) else {'weights_percent': {'perspective': 10}, 'threshold': 30}
    PE = fz['weights_percent']['perspective'] / 100.0; T_FIX = fz['threshold']

    data = {}
    for i, (name, f, key, _) in enumerate(bc.DATASETS):
        if not os.path.exists(f):
            continue
        y_all = bc.load_labels(name, f); texts, _, _ = bc.load_texts(name, f)
        r, pe, _ = bc.load_scores(key, len(y_all))
        if r is None:
            print(f'  skip {name}: no saved agent scores'); continue
        ok = ~np.isnan(r) & ~np.isnan(pe)
        y = y_all[ok]; texts = [t for t, k in zip(texts, ok) if k]
        data[name] = {'y': y, 'texts': texts, 'dev': bc.stratified_split(y, bc.SEED + i),
                      'base': (1 - PE) * r[ok] + PE * pe[ok], 'key': key}
        print(f'  loaded {name:<24} {len(y):>6} emails')

    term_lists, tuning_rows, alone_rows, sweep_rows, boot_rows, diag_rows = {}, [], [], [], [], []
    rng = np.random.RandomState(bc.SEED)
    for lang, L in LANGS.items():
        names = [n for n in L['datasets'] if n in data]
        if not names:
            continue
        # ---- balanced development pool for this language
        pool_t, pool_y = [], []
        for n in names:
            d = data[n]; dev_idx = np.where(d['dev'])[0]
            for cls in (1, 0):
                ids = dev_idx[d['y'][dev_idx] == cls]
                pick = rng.choice(ids, min(MAX_PER_CLASS, len(ids)), replace=False) if len(ids) else []
                pool_t += [d['texts'][j] for j in pick]; pool_y += [cls] * len(pick)
        pool_y = np.array(pool_y)
        ph, lg = select_terms(pool_t, pool_y, L['token'])
        term_lists[lang] = {'phishing_terms': ph, 'legitimate_terms': lg, 'development_emails_used': len(pool_y)}
        print(f'\n  {lang}: {len(ph)} phishing terms, {len(lg)} legitimate terms from {len(pool_y)} development emails')
        print(f'    phishing: {", ".join(ph[:12])} ...')
        print(f'    legitimate: {", ".join(lg[:12])} ...')

        scorers = {'BM25': BM25Pattern(ph, lg, pool_t, lang), 'TF-IDF': TFIDFPattern(ph, lg, pool_t, lang)}
        # ---- tune k1, b on development halves (pattern-alone AUC, mean over datasets)
        best = {'BM25': None, 'TF-IDF': (None, None)}
        grid = []
        for k1 in K1_GRID:
            for b in B_GRID:
                aucs = [safe_auc(data[n]['y'][data[n]['dev']], scorers['BM25'].raw([t for t, k in zip(data[n]['texts'], data[n]['dev']) if k], k1, b)) for n in names]
                grid.append((float(np.nanmean(aucs)), k1, b))
                tuning_rows.append({'Language': lang, 'k1': k1, 'b': b, 'Dev_mean_AUC': round(float(np.nanmean(aucs)), 4),
                                    'Previous_default': 'yes' if (k1, b) == (1.7, 0.9) else ''})
        auc_best, k1b, bb = max(grid)
        best['BM25'] = (k1b, bb)
        prev = next(a for a, k1, b in grid if (k1, b) == (1.7, 0.9))
        print(f'    BM25 tuned on development halves: k1={k1b}, b={bb} (dev AUC {auc_best:.3f}; previous k1=1.7, b=0.9: {prev:.3f})')

        for sname, sc in scorers.items():
            k1, b = best[sname]
            dev_raw = np.concatenate([sc.raw([t for t, k in zip(data[n]['texts'], data[n]['dev']) if k], k1, b) for n in names])
            s = float(np.std(dev_raw)) or 1.0
            P = {}
            for n in names + [m for m in L['recall_only'] if m in data]:
                P[n] = to_score(sc.raw(data[n]['texts'], k1, b), s)
            # pattern alone, test halves
            for n in names:
                te = ~data[n]['dev']
                alone_rows.append({'Language': lang, 'Scorer': sname, 'Dataset': n, 'Test_AUC_pattern_alone': round(safe_auc(data[n]['y'][te], P[n][te]), 4),
                                   'k1': k1 if sname == 'BM25' else '', 'b': b if sname == 'BM25' else ''})
            # threshold chosen on development halves for each pattern weight
            for w in PATTERN_WEIGHTS:
                def comb(n):
                    return (w / 100.0) * P[n] + (1 - w / 100.0) * data[n]['base']
                dev_f1 = {T: np.mean([bc.metrics(data[n]['y'][data[n]['dev']], comb(n)[data[n]['dev']] >= T)['F1'] for n in names]) for T in T_GRID}
                T_dev = max(T_GRID, key=lambda T: (round(dev_f1[T], 6), -abs(T - T_FIX)))
                for mode, T in (('T fixed at 30', T_FIX), ('T chosen on development', T_dev)):
                    for n in names + [m for m in L['recall_only'] if m in data]:
                        te = ~data[n]['dev']; res = bc.metrics(data[n]['y'][te], comb(n)[te] >= T)
                        sweep_rows.append({'Language': lang, 'Scorer': sname, 'Pattern_weight_%': w, 'Threshold_mode': mode, 'T': T,
                                           'Dataset': n, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()},
                                           'Note': 'phishing-only: use Recall' if n in L['recall_only'] else ''})
            # paired bootstrap: w = 3 and 10 vs 0, macro over this language's datasets, T fixed
            for w in (3, 10):
                per = []
                for n in names:
                    te = ~data[n]['dev']; yy = data[n]['y'][te]
                    a = ((w / 100.0) * P[n] + (1 - w / 100.0) * data[n]['base'])[te] >= T_FIX
                    z = data[n]['base'][te] >= T_FIX
                    per.append((yy, a, z))
                diffs = np.empty(N_BOOT)
                for bi in range(N_BOOT):
                    vals = []
                    for yy, a, z in per:
                        ix = rng.randint(0, len(yy), len(yy)); vals.append(bc.f1_only(yy[ix], a[ix]) - bc.f1_only(yy[ix], z[ix]))
                    diffs[bi] = np.mean(vals)
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                boot_rows.append({'Language': lang, 'Scorer': sname, 'Comparison': f'Pattern {w}% minus Pattern 0% (T=30, macro F1)',
                                  'F1_diff_pp': round(float(diffs.mean()), 2), 'CI95_low': round(float(lo), 2), 'CI95_high': round(float(hi), 2),
                                  'Verdict': 'clear difference' if (lo > 0 or hi < 0) else 'no clear difference'})

    # ---- diagnostic: previous Arabic BM25 set-up on the Salloum Arabic TEST half
    sa = data.get('Salloum Arabic 300')
    if sa is not None:
        te = ~sa['dev']; yy = sa['y'][te]; tt = [t for t, k in zip(sa['texts'], te) if k]
        setups = []
        bm25_file = 'AI_Phishing_Detection_System_v50_BM25.py'
        if os.path.exists(bm25_file):
            fp, fl = previous_fallback_lists(bm25_file)
            if fp and fl:
                setups.append(('Previous scorer, hand-written lists (30/29)', fp, fl))
        if os.path.exists('Salloum_Arabic_300.csv'):
            lp, ll = previous_lists_from_file('Salloum_Arabic_300.csv')
            setups.append(('Previous scorer, lists built from the WHOLE Salloum Arabic file (includes the test emails)', lp, ll))
        for label, p_docs, l_docs in setups:
            f = previous_arabic_scorer(p_docs, l_docs)
            s_te = np.array([f(t) for t in tt])
            diag_rows.append({'Setup': label, 'Test_AUC_pattern_alone': round(safe_auc(yy, s_te), 4)})
        new = next((r for r in alone_rows if r['Scorer'] == 'BM25' and r['Dataset'] == 'Salloum Arabic 300'), None)
        if new:
            diag_rows.append({'Setup': 'New BM25 (email = document), lists from development halves only', 'Test_AUC_pattern_alone': new['Test_AUC_pattern_alone']})

    # ---- print summary
    print('\n' + '-' * 78 + '\n  PATTERN ALONE (test halves): AUC, 0.5 = no better than chance\n' + '-' * 78)
    for r in alone_rows:
        print(f"    {r['Language']:<8} {r['Scorer']:<7} {r['Dataset']:<24} AUC {r['Test_AUC_pattern_alone']:.3f}")
    print('\n' + '-' * 78 + '\n  WEIGHT SWEEP (test halves, T fixed at 30): macro F1 per language\n' + '-' * 78)
    for lang in LANGS:
        for sname in ('BM25', 'TF-IDF'):
            line = f'    {lang:<8} {sname:<7}'
            for w in PATTERN_WEIGHTS:
                rows = [r for r in sweep_rows if r['Language'] == lang and r['Scorer'] == sname and r['Pattern_weight_%'] == w
                        and r['Threshold_mode'] == 'T fixed at 30' and not r['Note']]
                if rows:
                    line += f"   Pattern {w:>2}%: {np.mean([r['F1'] for r in rows]):6.2f}%"
            print(line)
    for r in boot_rows:
        print(f"    {r['Language']:<8} {r['Scorer']:<7} {r['Comparison']}: {r['F1_diff_pp']:+.2f}pp [{r['CI95_low']:+.2f}, {r['CI95_high']:+.2f}] {r['Verdict']}")
    if diag_rows:
        print('\n' + '-' * 78 + '\n  DIAGNOSTIC: previous Arabic BM25 set-up vs new (Salloum Arabic test half)\n' + '-' * 78)
        for r in diag_rows:
            print(f"    AUC {r['Test_AUC_pattern_alone']:.3f}  {r['Setup']}")

    bc.write_csv(os.path.join(OUT_DIR, 'bm25_tuning.csv'), tuning_rows)
    bc.write_csv(os.path.join(OUT_DIR, 'pattern_alone.csv'), alone_rows)
    bc.write_csv(os.path.join(OUT_DIR, 'weight_sweep.csv'), sweep_rows)
    bc.write_csv(os.path.join(OUT_DIR, 'bootstrap_pattern_weight.csv'), boot_rows)
    bc.write_csv(os.path.join(OUT_DIR, 'diagnostic_previous_setup.csv'), diag_rows)
    json.dump(term_lists, open(os.path.join(OUT_DIR, 'term_lists.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'seed': bc.SEED, 'split_source': bc.SPLIT_SOURCE,
               'n_terms': N_TERMS, 'max_per_class_per_dataset': MAX_PER_CLASS, 'min_df': MIN_DF, 'k1_grid': K1_GRID, 'b_grid': B_GRID,
               'pattern_weights_percent': PATTERN_WEIGHTS, 'threshold_fixed': T_FIX, 'threshold_grid': T_GRID,
               'frozen_reasoning_perspective': [1 - PE, PE], 'python': platform.python_version(),
               'runtime_minutes': round((time.time() - t0) / 60, 1)}, open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
    print(f'\n  Saved to {OUT_DIR}/ ({(time.time() - t0) / 60:.1f} minutes).\n')


if __name__ == '__main__':
    main()
