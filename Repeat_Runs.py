"""
Repeat_Runs.py
=============================================================================
GPT-4o-mini is not fully deterministic, and
every configuration was scored once. This script scores the 300-email datasets
two more times with the final system, so that every result on them can be
reported as mean +/- standard deviation over three runs (run 1 = the scores
already saved in score_cache_reasoning_only/).

Datasets re-scored: Salloum English 300, Salloum Arabic 300, Synthetic Arabic 600
(1,200 emails x 2 runs = 2,400 GPT-4o-mini calls, about US$1, 20-40 minutes).
Enron and MeAJOR are large enough that run-to-run variation is small; they keep
their saved scores.

It reports
  1. Held-out test halves: F1 of the frozen configuration (R 90 / Pe 10, T 30),
     equal weight (50/50) and Reasoning alone, per run and mean +/- SD.
  2. Perspective-weight sweep (0, 10, 20, 30, 50%) on the full datasets,
     mean +/- SD, and whether the 10%-vs-0% difference exceeds run-to-run noise.
  3. Selection stability: the pre-specified selection rule re-applied with each
     run's scores (Enron and MeAJOR keep run-1 scores). Does it still pick 90/10, T=30?
  4. Agreement between runs: share of emails whose decision changes.
  Check: the Perspective Agent is rule-based, so its new scores must equal the
  saved ones; if not, the script warns that the inputs differ.

Run in the repository folder (needs AI_Phishing_Detection_System_Final.py,
Baseline_Classical.py, the CSV files, score_cache_reasoning_only/):

    export OPENAI_API_KEY="your-new-key"
    caffeinate -i python3 Repeat_Runs.py

If it stops, run the same command again: saved scores are kept.
Results: repeat_runs_results/
"""
import os, re, sys, json, time, types, getpass, argparse, platform, collections, threading, logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd

import Baseline_Classical as bc

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)

OUT_DIR = 'repeat_runs_results'   # a separate folder is used when --datasets is given
SMALL = ['Salloum English 300', 'Salloum Arabic 300', 'UAE Synthetic Arabic']
SELECTION_SETS = ['Enron Full', 'Salloum English 300', 'Salloum Arabic 300', 'MeAJOR 2025 (3,000)', 'UAE Synthetic Arabic']
PE_GRID = [0, 10, 20, 30, 40, 50]
T_GRID = [20, 25, 30, 35, 40]
SWEEP = [0, 10, 20, 30, 50]
TIE_BAND = 0.25
FROZEN = (10, 30)
COST_PER_EMAIL = 0.00042
MAX_CONCURRENT = 3
MAX_PASSES = 5
PASS_PAUSE_SECONDS = 60
SYSTEM_FILE = 'AI_Phishing_Detection_System_Final.py'


def load_fields(name, path):
    df = pd.read_csv(path)
    if name.startswith('MeAJOR'):
        df = bc._meajor_sample(df)
    cols = {c.strip().lower(): c for c in df.columns}
    label_col = next(cols[k] for k in cols if k in ('label', 'class', 'type', 'category'))
    tcol = next((cols[k] for k in bc.TEXT_PREF if k in cols), None)
    if tcol is None:
        cands = [c for c in df.columns if c != label_col and bc._is_text(df[c])]
        tcol = max(cands, key=lambda c: df[c].fillna('').astype(str).str.len().sum())
    get = lambda c: df[c].fillna('').astype(str).tolist() if c else [''] * len(df)
    return get(tcol), get(cols.get('subject')), get(cols.get('sender'))


def load_detector(system_file):
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API key (typing is hidden): ')
    src = open(system_file, encoding='utf-8').read()
    cut = src.find('\ndef create_gradio_interface():')
    if cut == -1:
        sys.exit('Unexpected structure in the system file; stopping rather than guessing.')
    mod = types.ModuleType('agents')
    exec(compile(src[:cut], system_file, 'exec'), mod.__dict__)
    return mod.PhishingDetector(mod.SystemConfig())


def f1(y, pred):
    return bc.f1_only(y, pred)


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def select(scores_by_ds):
    """Pre-specified rule: highest mean dev F1 (Pe >= 10%), ties within 0.25pp -> fewest dev FP."""
    rows = []
    for pe in PE_GRID:
        for T in T_GRID:
            f1s, fps = [], 0
            for name, (y, dev, r, p) in scores_by_ds.items():
                pred = ((1 - pe / 100) * r + (pe / 100) * p)[dev] >= T
                f1s.append(f1(y[dev], pred)); fps += int((pred & (y[dev] == 0)).sum())
            rows.append({'Pe': pe, 'T': T, 'dev_mean_F1': float(np.mean(f1s)), 'dev_FP': fps})
    elig = [r for r in rows if r['Pe'] >= 10]
    best = max(r['dev_mean_F1'] for r in elig)
    tied = [r for r in elig if r['dev_mean_F1'] >= best - TIE_BAND]
    chosen = min(tied, key=lambda r: (r['dev_FP'], -r['dev_mean_F1']))
    return chosen, tied, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=2, help='extra runs to add (default 2, giving 3 in total)')
    ap.add_argument('--workers', type=int, default=MAX_CONCURRENT)
    ap.add_argument('--system', default=SYSTEM_FILE)
    ap.add_argument('--datasets', nargs='*', default=None, help='datasets to re-score (default: the three 300/600-email sets); e.g. "MeAJOR 2025 (3,000)"')
    args = ap.parse_args()
    global SMALL
    if args.datasets:
        SMALL = list(args.datasets)
        bad = [n for n in SMALL if n not in SELECTION_SETS]
        if bad:
            sys.exit(f'Unknown dataset name(s): {bad}. Use: {SELECTION_SETS}')
    if not os.path.exists(args.system):
        sys.exit(f'{args.system} not found in this folder.')
    global OUT_DIR
    if args.datasets:
        OUT_DIR = 'repeat_runs_results_' + '_'.join(re.sub(r'\W+', '', n.split()[0]) for n in SMALL)
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '=' * 78)
    print('  REPEATED SCORING RUNS: run-to-run variation of GPT-4o-mini')
    print(f'  {datetime.now():%Y-%m-%d %H:%M} | system: {args.system} | split: {bc.SPLIT_SOURCE}')
    print('=' * 78)

    # ---------- data and run-1 scores
    D = {}
    for i, (name, f, key, _) in enumerate(bc.DATASETS):
        if name not in SELECTION_SETS or not os.path.exists(f):
            continue
        y_all = bc.load_labels(name, f)
        r, pe, _ = bc.load_scores(key, len(y_all))
        if r is None:
            sys.exit(f'No saved scores for {name} in {bc.CACHE_DIR}/')
        ok = ~np.isnan(r) & ~np.isnan(pe); rows = np.where(ok)[0]
        y = y_all[ok]
        D[name] = {'i': i, 'key': key, 'file': f, 'rows': rows, 'y': y, 'dev': bc.stratified_split(y, bc.SEED + i),
                   'runs': [(r[ok], pe[ok])]}
    missing = [n for n in SMALL if n not in D]
    if missing:
        sys.exit(f'Missing datasets: {missing}')

    # ---------- extra runs (cached per run)
    detector = load_detector(args.system)
    errors = collections.Counter(); lock = threading.Lock()

    def one(job):
        k, text, subject, sender = job
        try:
            res = detector.analyze_gpt_reasoning_agent(text, sender, subject)
            if 'Fallback' in str(res.get('agent_name', '')):
                return k, None
            p = detector.analyze_multiperspective_agent(text, sender, subject)['score']
            return k, {'r': float(res['score']), 'pe': float(p)}
        except Exception as e:
            msg = f'{type(e).__name__}: {e}'
            if 'insufficient_quota' in msg:
                msg = 'OpenAI says the account has no credit left (insufficient_quota). Add credit under Settings > Billing.'
            elif 'rate_limit' in msg or 'RateLimit' in msg or '429' in msg:
                msg = 'OpenAI rate limit reached (too many requests or tokens per minute).'
            with lock:
                errors[msg[:300]] += 1
            return k, None

    pe_mismatch = collections.Counter()
    for name in SMALL:
        d = D[name]; body, subj, send = load_fields(name, d['file'])
        for run in range(2, 2 + args.runs):
            path = os.path.join(OUT_DIR, f'{d["key"]}_run{run}.json')
            cache = json.load(open(path)) if os.path.exists(path) else {}
            for attempt in range(1, MAX_PASSES + 1):
                todo = [int(j) for j in d['rows'] if str(int(j)) not in cache]
                if not todo:
                    break
                if attempt > 1:
                    print(f'    waiting {PASS_PAUSE_SECONDS} s before the next pass ...'); time.sleep(PASS_PAUSE_SECONDS)
                print(f'  {name}, run {run}: scoring {len(todo)} emails (about ${len(todo) * COST_PER_EMAIL:.2f}), pass {attempt}')
                errors.clear(); done = 0
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    for k, sc in ex.map(one, [(j, body[j], subj[j], send[j]) for j in todo]):
                        done += 1
                        if sc is not None:
                            cache[str(k)] = sc
                        if done % 50 == 0:
                            json.dump(cache, open(path, 'w')); print(f'    {done}/{len(todo)}', end='\r')
                json.dump(cache, open(path, 'w')); print()
                for msg, n in errors.most_common(3):
                    print(f'    reason ({n}x): {msg}')
                if any('no credit left' in m for m in errors):
                    sys.exit('Stopped: no OpenAI credit. Add credit and run the same command again.')
            left = [int(j) for j in d['rows'] if str(int(j)) not in cache]
            if left:
                sys.exit(f'{len(left)} emails in {name} run {run} still have no genuine score. Run the same command again.')
            r_new = np.array([cache[str(int(j))]['r'] for j in d['rows']])
            pe_new = np.array([cache[str(int(j))]['pe'] for j in d['rows']])
            pe_mismatch[name] += int((np.abs(pe_new - d['runs'][0][1]) > 1e-6).sum())
            d['runs'].append((r_new, pe_new))
    for name in SMALL:
        if pe_mismatch[name]:
            print(f'  WARNING: {name}: {pe_mismatch[name]} Perspective scores differ from the saved run. The email text given to the '
                  f'agents may differ from the original scoring; tell me before using these results.')

    # ---------- 1. held-out test halves, mean +/- SD
    held, per_run = [], []
    settings = [('Selected (R 90 / Pe 10, T 30)', 10, 30), ('Equal weight (50/50, T 30)', 50, 30), ('Reasoning only (T 30)', 0, 30)]
    for label, pe_w, T in settings:
        for name in SMALL:
            d = D[name]; te = ~d['dev']; vals, fps = [], []
            for k, (r, p) in enumerate(d['runs'], 1):
                pred = ((1 - pe_w / 100) * r + (pe_w / 100) * p)[te] >= T
                m = bc.metrics(d['y'][te], pred); vals.append(m['F1']); fps.append(m['FP'])
                per_run.append({'Part': 'held-out test half', 'Setting': label, 'Dataset': name, 'Run': k,
                                'F1': round(m['F1'], 2), 'Recall': round(m['Recall'], 2), 'FP': m['FP']})
            mu, sd = mean_sd(vals)
            held.append({'Setting': label, 'Dataset': name, 'F1_mean': round(mu, 2), 'F1_SD': round(sd, 2),
                         'F1_min': round(min(vals), 2), 'F1_max': round(max(vals), 2), 'FP_mean': round(float(np.mean(fps)), 1)})

    # ---------- 2. Perspective-weight sweep on full datasets
    sweep = []
    for name in SMALL:
        d = D[name]
        for pe_w in SWEEP:
            vals = [f1(d['y'], ((1 - pe_w / 100) * r + (pe_w / 100) * p) >= 30) for r, p in d['runs']]
            mu, sd = mean_sd(vals)
            sweep.append({'Dataset': name, 'Perspective_weight_%': pe_w, 'F1_mean': round(mu, 2), 'F1_SD': round(sd, 2),
                          'F1_runs': ' / '.join(f'{v:.2f}' for v in vals)})
        diffs = [f1(d['y'], (0.9 * r + 0.1 * p) >= 30) - f1(d['y'], r >= 30) for r, p in d['runs']]
        mu, sd = mean_sd(diffs)
        sweep.append({'Dataset': name, 'Perspective_weight_%': '10 minus 0', 'F1_mean': round(mu, 2), 'F1_SD': round(sd, 2),
                      'F1_runs': ' / '.join(f'{v:+.2f}' for v in diffs)})

    # ---------- 3. selection stability
    sel = []
    n_runs = len(D[SMALL[0]]['runs'])
    for k in range(n_runs):
        sbd = {}
        for name in SELECTION_SETS:
            d = D[name]; r, p = d['runs'][k] if name in SMALL else d['runs'][0]
            sbd[name] = (d['y'], d['dev'], r, p)
        chosen, tied, best = select(sbd)
        frozen_in_tie = any(t['Pe'] == FROZEN[0] and t['T'] == FROZEN[1] for t in tied)
        sel.append({'Run': k + 1, 'Chosen_Pe_%': chosen['Pe'], 'Chosen_T': chosen['T'], 'Chosen_dev_mean_F1': round(chosen['dev_mean_F1'], 2),
                    'Best_dev_mean_F1': round(best, 2), 'Tied_settings': len(tied),
                    'Frozen_90_10_T30_within_tie_band': 'yes' if frozen_in_tie else 'no',
                    'Same_as_frozen': 'yes' if (chosen['Pe'], chosen['T']) == FROZEN else 'no'})

    # ---------- 4. agreement between runs
    agree = []
    for name in SMALL:
        d = D[name]; r1, p1 = d['runs'][0]; base = (0.9 * r1 + 0.1 * p1) >= 30
        for k, (r, p) in enumerate(d['runs'][1:], 2):
            dec = (0.9 * r + 0.1 * p) >= 30
            agree.append({'Dataset': name, 'Run_vs_run1': k, 'Decisions_changed': int((dec != base).sum()),
                          'Emails': len(dec), 'Changed_%': round(100 * float((dec != base).mean()), 2),
                          'Mean_abs_Reasoning_diff': round(float(np.abs(r - r1).mean()), 2),
                          'Perspective_scores_identical': 'yes' if not pe_mismatch[name] else f'NO ({pe_mismatch[name]} differ)'})

    # ---------- print and save
    print('\n' + '-' * 78 + '\n  HELD-OUT TEST HALVES: F1 mean +/- SD over runs\n' + '-' * 78)
    for h in held:
        print(f"  {h['Setting']:<32}{h['Dataset']:<24}{h['F1_mean']:6.2f} +/- {h['F1_SD']:.2f}   (range {h['F1_min']:.2f}-{h['F1_max']:.2f})")
    print('\n' + '-' * 78 + '\n  PERSPECTIVE SWEEP (full datasets, T = 30): F1 mean +/- SD\n' + '-' * 78)
    for s in sweep:
        print(f"  {s['Dataset']:<24} Pe {str(s['Perspective_weight_%']):<10} {s['F1_mean']:7.2f} +/- {s['F1_SD']:.2f}   runs: {s['F1_runs']}")
    print('\n' + '-' * 78 + '\n  SELECTION RULE RE-APPLIED WITH EACH RUN\n' + '-' * 78)
    for s in sel:
        print(f"  run {s['Run']}: chosen Pe {s['Chosen_Pe_%']}%, T {s['Chosen_T']} | 90/10 T30 in tie band: {s['Frozen_90_10_T30_within_tie_band']}"
              f" | {s['Tied_settings']} tied settings")
    if sel and sel[0]['Same_as_frozen'] != 'yes':
        print('  NOTE: with the saved run-1 scores the rule did not reproduce the frozen choice (Pe 10%, T 30).'
              ' Please tell me: the dataset files or the score cache may differ from those used for the selection.')
    print('\n' + '-' * 78 + '\n  AGREEMENT WITH RUN 1 (frozen configuration)\n' + '-' * 78)
    for a in agree:
        print(f"  {a['Dataset']:<24} run {a['Run_vs_run1']}: {a['Decisions_changed']} of {a['Emails']} decisions changed "
              f"({a['Changed_%']}%), mean |dR| {a['Mean_abs_Reasoning_diff']}, Perspective identical: {a['Perspective_scores_identical']}")

    bc.write_csv(os.path.join(OUT_DIR, 'heldout_mean_sd.csv'), held)
    bc.write_csv(os.path.join(OUT_DIR, 'heldout_per_run.csv'), per_run)
    bc.write_csv(os.path.join(OUT_DIR, 'perspective_sweep_mean_sd.csv'), sweep)
    bc.write_csv(os.path.join(OUT_DIR, 'selection_stability.csv'), sel)
    bc.write_csv(os.path.join(OUT_DIR, 'run_agreement.csv'), agree)
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'system_file': args.system, 'runs_total': n_runs,
               'datasets_rescored': SMALL, 'seed': bc.SEED, 'split_source': bc.SPLIT_SOURCE, 'model': 'gpt-4o-mini',
               'python': platform.python_version()}, open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
    print(f'\n  Saved to {OUT_DIR}/ . The *_run2.json / *_run3.json files are the new scores.\n')


if __name__ == '__main__':
    main()
