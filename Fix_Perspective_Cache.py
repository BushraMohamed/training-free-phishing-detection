"""
Fix_Perspective_Cache.py
=============================================================================
Dr. Ahmed's R2 - the fix. Check_Perspective_And_Sanitiser.py showed that the
saved (run-1) Perspective scores of 7 Salloum Arabic and 37 synthetic emails were
produced by the earlier code file AI_Phishing_Detection_System_v50_TFIDF.py, not
by AI_Phishing_Detection_System_Final.py. The Perspective Agent itself is
deterministic.

This script:
  1. backs up score_cache_reasoning_only/ to score_cache_reasoning_only_run1_original/
     (only the first time; the backup is never overwritten);
  2. recomputes the rule-based Perspective score of EVERY saved email with the
     final code and writes it into the cache (Reasoning scores are not touched);
  3. explains the difference: which expert rule fires differently in the two
     code files, and which keyword lists differ (perspective_fix/cause.txt);
  4. MOM 41 Action 3: the sanitiser changes 3 legitimate emails (2 Enron, 1 MeAJOR).
     If an OpenAI key is set, it re-scores those 3 sanitised emails with the
     reasoning agent (3 calls, less than US$0.01) and stores the new scores, so the
     cache matches the final pipeline exactly. Without a key this part is skipped.

No other API calls. About 5-10 minutes (Enron is the slow part).

    python3 Fix_Perspective_Cache.py
    (optional, for part 4:  export OPENAI_API_KEY="your-key"  first)

Then run (free, they reuse saved scores):
    python3 Pareto_Unified.py
    python3 Repeat_Runs.py          (press Enter if it asks for a key: nothing new is scored)

Results: perspective_fix/
"""
import os, sys, json, shutil, time, platform
from datetime import datetime
import numpy as np

import Baseline_Classical as bc
import Robustness_Test as rt
from Check_Perspective_And_Sanitiser import load_agent

OUT = 'perspective_fix'
FINAL, OLD = 'AI_Phishing_Detection_System_Final.py', 'AI_Phishing_Detection_System_v50_TFIDF.py'
CACHE = bc.CACHE_DIR
BACKUP = CACHE + '_run1_original'
LISTS = ['infrastructure_keywords', 'infrastructure_keywords_english', 'trusted_domains']


def experts(det, b, s, u):
    res = det.analyze_multiperspective_agent(b, s, u)
    return res['score'], {e['expert'].split(':')[0]: (e['score'], e.get('indicators', [])) for e in res.get('expert_assessments', [])}


def main():
    t0 = time.time()
    key_set = os.environ.get('OPENAI_API_KEY', '').startswith('sk-')   # checked before the agents load
    os.makedirs(OUT, exist_ok=True)
    print('\n' + '=' * 78 + '\n  FIX: Perspective scores in the saved cache recomputed with the final code (R2)\n' + '=' * 78)
    if not os.path.exists(BACKUP):
        shutil.copytree(CACHE, BACKUP)
        print(f'  backup made: {BACKUP}/')
    else:
        print(f'  backup already exists, kept as it is: {BACKUP}/')
    final = load_agent(FINAL)
    old = load_agent(OLD) if os.path.exists(OLD) else None

    changed_rows, summary, cause = [], [], []
    for name, csv_file, key, _ in bc.DATASETS:
        path = os.path.join(CACHE, f'{key}_scores.json')
        if not os.path.exists(csv_file) or not os.path.exists(path):
            print(f'  skip {name}'); continue
        body, subj, send = rt.load_fields(name, csv_file)
        cache = json.load(open(path))
        n_changed = 0
        for k, v in cache.items():
            j = int(k)
            if j >= len(body) or v.get('pe') is None:
                continue
            new = float(final.analyze_multiperspective_agent(body[j], send[j], subj[j])['score'])
            if abs(new - float(v['pe'])) > 1e-6:
                n_changed += 1
                row = {'Dataset': name, 'Row': j, 'Pe_saved': round(float(v['pe']), 4), 'Pe_final_code': round(new, 4)}
                if old is not None and len(cause) < 60:
                    _, ex_new = experts(final, body[j], send[j], subj[j])
                    _, ex_old = experts(old, body[j], send[j], subj[j])
                    for e in sorted(set(ex_new) | set(ex_old)):
                        a, b = ex_new.get(e, (0, [])), ex_old.get(e, (0, []))
                        if a[0] != b[0]:
                            cause.append(f'{name} row {j}: {e} final={a[0]} {a[1]} | v50_TFIDF={b[0]} {b[1]}')
                changed_rows.append(row)
                v['pe'] = new
        if n_changed:
            json.dump(cache, open(path, 'w'))
        summary.append({'Dataset': name, 'Saved_emails': len(cache), 'Perspective_scores_corrected': n_changed})
        print(f'  {name:<24} {len(cache):>6} saved emails, {n_changed} Perspective scores corrected')

    # ---------- keyword lists that differ between the two code files
    if old is not None:
        for attr in LISTS:
            a, b = set(map(str, getattr(final, attr, []) or [])), set(map(str, getattr(old, attr, []) or []))
            if a != b:
                cause.append(f'LIST {attr}: only in final = {sorted(a - b)} | only in v50_TFIDF = {sorted(b - a)}')
        for attr in ('LEGITIMATE_SIGNALS',):
            a = set(map(str, getattr(getattr(final, 'config', None), attr, []) or []))
            b = set(map(str, getattr(getattr(old, 'config', None), attr, []) or []))
            if a != b:
                cause.append(f'LIST {attr}: only in final = {sorted(a - b)} | only in v50_TFIDF = {sorted(b - a)}')
    open(os.path.join(OUT, 'cause.txt'), 'w', encoding='utf-8').write('\n'.join(cause) or 'v50_TFIDF file not found or no difference')

    # ---------- sanitiser: the legitimate emails it changes
    san = []
    for name, csv_file, key, _ in bc.DATASETS:
        path = os.path.join(CACHE, f'{key}_scores.json')
        if not os.path.exists(csv_file) or not os.path.exists(path):
            continue
        body, subj, send = rt.load_fields(name, csv_file)
        cache = json.load(open(path)); dirty = False
        for k, v in cache.items():
            j = int(k)
            if j >= len(body) or rt.sanitize(body[j]) == body[j].strip():
                continue
            clean = rt.sanitize(body[j])
            row = {'Dataset': name, 'Row': j, 'R_saved': round(float(v['r']), 2), 'Pe': round(float(v['pe']), 4)}
            if key_set:
                try:
                    r_new = float(final.analyze_gpt_reasoning_agent(clean, send[j], subj[j])['score'])
                except Exception as e:
                    print(f'    could not re-score {name} row {j}: {type(e).__name__}: {str(e)[:150]}')
                    san.append(row); continue
                old_dec = 0.9 * v['r'] + 0.1 * v['pe'] >= 30
                v['r'] = r_new; dirty = True
                row.update({'R_after_sanitiser': round(r_new, 2), 'Decision_before': int(old_dec),
                            'Decision_after': int(0.9 * r_new + 0.1 * v['pe'] >= 30)})
            san.append(row)
        if dirty:
            json.dump(cache, open(path, 'w'))
    if san:
        print(f'\n  sanitiser changes {len(san)} emails' + ('' if key_set else ' (no OpenAI key set: not re-scored)'))
        for s in san:
            print('   ', s)

    bc.write_csv(os.path.join(OUT, 'summary.csv'), summary)
    bc.write_csv(os.path.join(OUT, 'corrected_scores.csv'), changed_rows)
    bc.write_csv(os.path.join(OUT, 'sanitiser_rescored.csv'), san)
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'backup': BACKUP, 'sanitiser_rescored': key_set,
               'python': platform.python_version(), 'runtime_minutes': round((time.time() - t0) / 60, 1)},
              open(os.path.join(OUT, 'run_info.json'), 'w'), indent=2)
    print(f'\n  Saved to {OUT}/ ({(time.time() - t0) / 60:.1f} minutes).')
    print('  Next (free): python3 Pareto_Unified.py   then   python3 Repeat_Runs.py\n')


if __name__ == '__main__':
    main()
