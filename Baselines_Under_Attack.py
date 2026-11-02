"""
Baselines_Under_Attack.py
=============================================================================
MOM 41, Action 4: test the trained TF-IDF baselines on the SAME attacked emails
as the proposed system (Robustness_Test.py), under identical conditions, and
compare their drop in recall with the system's.

No API calls, no cost, about 5 minutes.

  - The 250 phishing + 250 legitimate test-half emails and the attacks (visible
    and hidden injection, benign text, paraphrase, each injection also with the
    sanitiser) are rebuilt exactly by Attack_Set.py; paraphrases come from
    robustness_results/paraphrase_cache.json.
  - TF-IDF + logistic regression and TF-IDF + linear SVM are trained on the
    development half of each dataset exactly as in Baseline_Classical.py
    (in-domain), then classify the clean and attacked versions.
  - The system's decisions are read from robustness_results/per_email_scores.csv.

Run in the folder with the CSV files, score_cache_reasoning_only/,
robustness_results/ and the scripts:

    python3 Baselines_Under_Attack.py

Results: baseline_attack_results/
"""
import os, json, time, platform
from datetime import datetime
import numpy as np
import pandas as pd

import Baseline_Classical as bc
import Attack_Set as ats

OUT_DIR = 'baseline_attack_results'
N_BOOT = 1000
SYSTEM = 'Proposed system'


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print('\n' + '=' * 78)
    print('  TRAINED BASELINES ON THE ATTACKED EMAILS (MOM 41, Action 4) | no API calls')
    print(f'  {datetime.now():%Y-%m-%d %H:%M} | split: {bc.SPLIT_SOURCE}')
    print('=' * 78)
    items, conds = ats.build_attack_set()
    same = ats.check_against_robustness_results(items)
    print(f'  same emails as robustness_results/per_email_scores.csv: {"yes" if same else ("NO - stop and tell me" if same is False else "file not found")}')
    sys_path = os.path.join('robustness_results', 'per_email_scores.csv')
    sysdf = pd.read_csv(sys_path) if os.path.exists(sys_path) else None
    sys_flag = {} if sysdf is None else {(r.Dataset, int(r.Row), r.Condition): int(r.Flagged_final) for r in sysdf.itertuples()}

    preds = {}            # (model, item index, condition) -> 0/1
    checks = []
    for name in dict.fromkeys(it['dataset'] for it in items):
        i = next(k for k, ds in enumerate(bc.DATASETS) if ds[0] == name)
        csv_file, key = bc.DATASETS[i][1], bc.DATASETS[i][2]
        y_all = bc.load_labels(name, csv_file)
        texts_all, tcol, scol = bc.load_texts(name, csv_file)
        r, pe, _ = bc.load_scores(key, len(y_all))
        ok = (~np.isnan(r) & ~np.isnan(pe)) if r is not None else np.ones(len(y_all), dtype=bool)
        y = y_all[ok]; texts = [t for t, k in zip(texts_all, ok) if k]
        dev = bc.stratified_split(y, bc.SEED + i)
        t1 = time.time()
        feats, models = bc.fit_models([t for t, k in zip(texts, dev) if k], y[dev])
        print(f'\n  {name}: trained on {int(dev.sum())} development emails ({time.time() - t1:.0f}s); C = '
              + ', '.join(f'{m.split("+ ")[-1]} {c}' for m, (_, c) in models.items()))
        idx = [k for k, it in enumerate(items) if it['dataset'] == name]
        mism = 0
        for k in idx:
            it = items[k]
            build = lambda body: ((it['subject'] + ' \n ' + body) if scol else body)[:bc.MAX_CHARS]
            if build(it['text']) != texts_all[it['row']]:
                mism += 1
            names = list(conds[k].keys())
            X = feats.transform([build(conds[k][c]) for c in names])
            for m, (est, _) in models.items():
                for c, p in zip(names, est.predict(X)):
                    preds[(m, k, c)] = int(p)
            for c in names:
                if (name, it['row'], c) in sys_flag:
                    preds[(SYSTEM, k, c)] = sys_flag[(name, it['row'], c)]
        checks.append({'Dataset': name, 'Emails': len(idx), 'Text_rebuilt_differently': mism})
        if mism:
            print(f'  NOTE: {mism} rebuilt texts differ from the training text format (tell me).')

    df = ats.summarize(preds, items, conds)
    df.to_csv(os.path.join(OUT_DIR, 'recall_by_condition.csv'), index=False)
    bc.write_csv(os.path.join(OUT_DIR, 'integrity_checks.csv'), checks)
    ats.print_summary(df)
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'seed': bc.SEED, 'split_source': bc.SPLIT_SOURCE,
               'emails': len(items), 'same_sample_as_robustness_results': same, 'python': platform.python_version(),
               'runtime_minutes': round((time.time() - t0) / 60, 1)}, open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
    print(f'\n  Saved to {OUT_DIR}/ ({(time.time() - t0) / 60:.1f} minutes).\n')


if __name__ == '__main__':
    main()
