"""
Check_Perspective_And_Sanitiser.py
=============================================================================
No API calls, no cost, about 5-15 minutes (Enron is the slow part).

PART 1 - Dr. Ahmed's R2: why did the rule-based Perspective scores of
14 Salloum Arabic and 74 synthetic emails differ between the saved run
(score_cache_reasoning_only/) and the repeated runs?

  The Perspective Agent has no model call and no randomness, so the saved
  score must have been computed from different input (which fields were
  passed, or text cut to a length) or with a different code file.
  For every dataset this script:
    a) recomputes the Perspective score with the final code and the inputs the
       final system uses (body, sender, subject) and lists every email whose
       score differs from the saved one, with the four experts' scores;
    b) tries other inputs on those emails (body only, body + sender,
       body + subject, other CSV columns used as sender, text cut to a length)
       and the older code files if they are in the folder, and reports which
       one reproduces the saved scores;
    c) checks the result is stable when Python's hash seed changes.

PART 2 - MOM 41, Action 3: the input sanitiser inside the pipeline.
  Counts, for all six datasets, how many emails the sanitiser would change.
  If none, every reported result is unchanged by adding it to the pipeline.

Run in the folder with the CSV files, score_cache_reasoning_only/ and the scripts:

    python3 Check_Perspective_And_Sanitiser.py

Results: perspective_sanitiser_check/
"""
import os, sys, re, json, time, types, platform, subprocess
from datetime import datetime
import numpy as np
import pandas as pd

import Baseline_Classical as bc
import Robustness_Test as rt

OUT_DIR = 'perspective_sanitiser_check'
CODE_FILES = ['AI_Phishing_Detection_System_Final.py', 'AI_Phishing_Detection_System_v50_TFIDF.py', 'AI_Phishing_Detection_System_v50_BM25.py']
TRUNCATIONS = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
SAMPLE_MATCHED = 200


def load_agent(path):
    sys.modules.setdefault('gradio', types.ModuleType('gradio'))
    os.environ.setdefault('OPENAI_API_KEY', 'not-needed-for-this-check')
    src = open(path, encoding='utf-8').read()
    cut = src.find('\ndef create_gradio_interface():')
    mod = types.ModuleType(os.path.basename(path)[:-3])
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src[:cut], path, 'exec'), mod.__dict__)
        return mod.PhishingDetector(mod.SystemConfig())


def experts(det, body, sender, subject):
    res = det.analyze_multiperspective_agent(body, sender, subject)
    ex = {e['expert'].split(':')[0]: e['score'] for e in res.get('expert_assessments', [])}
    return float(res['score']), ex


def all_columns(name, path):
    df = pd.read_csv(path)
    if name.startswith('MeAJOR'):
        df = bc._meajor_sample(df)
    return df


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, f))
    print('\n' + '=' * 78)
    print('  PERSPECTIVE-SCORE CHECK (R2) AND SANITISER CHECK (MOM 41, Action 3) | no API calls')
    print(f'  {datetime.now():%Y-%m-%d %H:%M}')
    print('=' * 78)
    agents = {f: load_agent(f) for f in CODE_FILES if os.path.exists(f)}
    if CODE_FILES[0] not in agents:
        sys.exit(f'{CODE_FILES[0]} not found in this folder.')
    final = agents[CODE_FILES[0]]
    print('  code files found: ' + ', '.join(agents))

    summary, mismatch_rows, hyp_rows, san_rows = [], [], [], []
    rng = np.random.RandomState(bc.SEED)
    for name, csv_file, key, _ in bc.DATASETS:
        if not os.path.exists(csv_file):
            print(f'  skip {name}: {csv_file} not found'); continue
        y = bc.load_labels(name, csv_file)
        body, subj, send = rt.load_fields(name, csv_file)
        r, pe, _ = bc.load_scores(key, len(y))
        df = all_columns(name, csv_file)
        t1 = time.time()
        # ---------- PART 1a: final code, final inputs
        new = np.full(len(y), np.nan)
        for j in range(len(y)):
            if pe is not None and not np.isnan(pe[j]):
                new[j] = final.analyze_multiperspective_agent(body[j], send[j], subj[j])['score']
        has = ~np.isnan(new)
        diff = has & (np.abs(new - pe) > 1e-6) if pe is not None else np.zeros(len(y), bool)
        bad = np.where(diff)[0]
        print(f'\n  {name}: {int(has.sum())} emails checked, {len(bad)} Perspective scores differ from the saved run ({time.time() - t1:.0f}s)')
        for j in bad:
            s_new, ex = experts(final, body[j], send[j], subj[j])
            mismatch_rows.append({'Dataset': name, 'Row': int(j), 'Label': int(y[j]), 'Saved_Pe': round(float(pe[j]), 4),
                                  'Recomputed_Pe': round(s_new, 4), **{f'{k}_now': v for k, v in ex.items()},
                                  'Reasoning_score_saved': round(float(r[j]), 2) if r is not None else '',
                                  'Body_length': len(body[j]), 'Has_Arabic': bool(re.search(r'[؀-ۿ]', body[j])),
                                  'Sender': send[j][:60], 'Subject': subj[j][:60], 'Body_start': body[j][:120].replace('\n', ' ')})
        # ---------- PART 1b: which inputs / code reproduce the saved scores?
        if len(bad):
            bad = rng.choice(bad, 300, replace=False) if len(bad) > 300 else bad   # enough to identify the cause
            good = np.where(has & ~diff)[0]
            ctrl = rng.choice(good, min(SAMPLE_MATCHED, len(good)), replace=False) if len(good) else np.array([], int)
            hyps = {'body + sender + subject (final system)': lambda j: (body[j], send[j], subj[j]),
                    'body only': lambda j: (body[j], '', ''),
                    'body + sender': lambda j: (body[j], send[j], ''),
                    'body + subject': lambda j: (body[j], '', subj[j]),
                    'subject + body as text, no sender': lambda j: (subj[j] + ' \n ' + body[j], '', '')}
            used = {c.strip().lower() for c in df.columns}
            for col in df.columns:
                if col.strip().lower() in ('label', 'class', 'type', 'category') or df[col].dtype != object:
                    continue
                vals = df[col].fillna('').astype(str).tolist()
                if vals == body or vals == subj or vals == send:
                    continue
                hyps[f'body + column "{col}" as sender'] = (lambda v: (lambda j: (body[j], v[j], subj[j])))(vals)
                hyps[f'body + column "{col}" as subject'] = (lambda v: (lambda j: (body[j], send[j], v[j])))(vals)
            for L in TRUNCATIONS:
                hyps[f'body cut to {L} characters'] = (lambda L: (lambda j: (body[j][:L], send[j], subj[j])))(L)
            for hname, fn in hyps.items():
                for fname, det in agents.items():
                    if fname != CODE_FILES[0] and not hname.startswith('body + sender + subject') and not hname == 'body only':
                        continue                      # older code files: test only the two main input variants
                    hit_bad = sum(abs(det.analyze_multiperspective_agent(*fn(j))['score'] - pe[j]) < 1e-6 for j in bad)
                    hit_ctrl = sum(abs(det.analyze_multiperspective_agent(*fn(j))['score'] - pe[j]) < 1e-6 for j in ctrl)
                    hyp_rows.append({'Dataset': name, 'Code_file': fname, 'Inputs': hname,
                                     'Reproduces_differing_emails': f'{hit_bad} of {len(bad)}',
                                     'Still_matches_control_emails': f'{hit_ctrl} of {len(ctrl)}',
                                     'Explains_all': 'YES' if hit_bad == len(bad) and hit_ctrl == len(ctrl) else ''})
        # ---------- PART 2: sanitiser
        changed = [j for j in range(len(y)) if rt.sanitize(body[j]) != body[j].strip()]
        for j in changed:
            removed = [ln for ln in body[j].split('\n') if ln not in rt.sanitize(body[j]).split('\n')]
            p_after = final.analyze_multiperspective_agent(rt.sanitize(body[j]), send[j], subj[j])['score']
            san_rows.append({'Dataset': name, 'Row': j, 'Label': int(y[j]), 'Removed_text': ' | '.join(removed)[:200],
                             'HTML_comment': bool(re.search(r'<!--.*?-->', body[j], re.S)),
                             'Pe_before': round(float(new[j]), 4) if not np.isnan(new[j]) else '', 'Pe_after': round(p_after, 4)})
        print(f'    sanitiser would change {len(changed)} of {len(y)} emails')
        summary.append({'Dataset': name, 'Emails': len(y), 'Perspective_differs_from_saved': len(bad),
                        'Sanitiser_changes': len(changed)})

    # ---------- PART 1c: hash-seed stability of the final code (fresh processes)
    stab = 'not run'
    if mismatch_rows:
        probe = json.dumps([[m['Dataset'], m['Row']] for m in mismatch_rows[:200]])
        code = ("import json,sys,Robustness_Test as rt,Baseline_Classical as bc,Check_Perspective_And_Sanitiser as c;"
                "d=c.load_agent('AI_Phishing_Detection_System_Final.py');p=json.loads(sys.argv[1]);out=[];cache={}\n"
                "for n,j in p:\n"
                " f=next(x[1] for x in bc.DATASETS if x[0]==n)\n"
                " if n not in cache: cache[n]=rt.load_fields(n,f)\n"
                " b,s,u=cache[n]; out.append(d.analyze_multiperspective_agent(b[j],u[j],s[j])['score'])\n"
                "print(json.dumps(out))")
        res = []
        for seed in ('1', '2', '3'):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            o = subprocess.run([sys.executable, '-c', code, probe], capture_output=True, text=True, env=env)
            try:
                res.append(json.loads(o.stdout.strip().splitlines()[-1]))
            except Exception:
                res.append(None)
        stab = 'identical across hash seeds' if res[0] is not None and all(x == res[0] for x in res) else 'DIFFERS across hash seeds'
        print(f'\n  Hash-seed check on the differing emails: {stab}')

    bc.write_csv(os.path.join(OUT_DIR, 'summary.csv'), summary)
    bc.write_csv(os.path.join(OUT_DIR, 'perspective_differences.csv'), mismatch_rows)
    bc.write_csv(os.path.join(OUT_DIR, 'perspective_hypotheses.csv'), hyp_rows)
    bc.write_csv(os.path.join(OUT_DIR, 'sanitiser_changes.csv'), san_rows)
    print('\n' + '-' * 78 + '\n  SUMMARY\n' + '-' * 78)
    for s in summary:
        print(f"  {s['Dataset']:<24} Perspective differs: {s['Perspective_differs_from_saved']:>5}   sanitiser changes: {s['Sanitiser_changes']:>5}")
    expl = [h for h in hyp_rows if h['Explains_all']]
    if expl:
        print('\n  Inputs/code that reproduce the saved scores exactly:')
        for h in expl:
            print(f"    {h['Dataset']}: {h['Inputs']}  [{h['Code_file']}]")
    elif hyp_rows:
        print('\n  No tested input reproduces all saved scores; see perspective_hypotheses.csv.')
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'code_files': list(agents), 'hash_seed_check': stab,
               'python': platform.python_version(), 'runtime_minutes': round((time.time() - t0) / 60, 1)},
              open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
    print(f'\n  Saved to {OUT_DIR}/ ({(time.time() - t0) / 60:.1f} minutes).\n')


if __name__ == '__main__':
    main()
