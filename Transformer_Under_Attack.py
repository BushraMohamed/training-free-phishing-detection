"""
Transformer_Under_Attack.py
=============================================================================
MOM 41, Action 4 (transformer part): fine-tune XLM-R (and AraBERT for Arabic)
exactly as in Baseline_Transformer.py on the development half of each dataset
used in the robustness test, then classify the same clean and attacked emails
as the proposed system (rebuilt by Attack_Set.py).

Run on Google Colab (GPU), after uploading the same files as for
Baseline_Transformer.py PLUS robustness_results/ and Attack_Set.py,
Robustness_Test.py:

    !python Transformer_Under_Attack.py
    !python Transformer_Under_Attack.py --model aubmindlab/bert-base-arabertv02 --datasets "Salloum Arabic 300"

About 10 minutes on a T4 GPU. Results: transformer_attack_<model>/
"""
import os, sys, json, time, argparse, platform
from datetime import datetime
import numpy as np

import Baseline_Classical as bc
import Baseline_Transformer as bt
import Attack_Set as ats

import torch
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='xlm-roberta-base')
    ap.add_argument('--datasets', nargs='*', default=None)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    args = ap.parse_args()
    short = args.model.rstrip('/').split('/')[-1]
    out = f'transformer_attack_{short}'
    os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s); log_f.write(s + '\n'); log_f.flush()

    t0 = time.time()
    bt.set_seed(bc.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log('\n' + '=' * 78 + f'\n  {args.model} ON THE ATTACKED EMAILS (MOM 41, Action 4) | device {device}\n' + '=' * 78)
    items, conds = ats.build_attack_set()
    same = ats.check_against_robustness_results(items)
    log(f'  same emails as robustness_results/per_email_scores.csv: {same}')
    tok = AutoTokenizer.from_pretrained(args.model)
    preds = {}
    names = [n for n in dict.fromkeys(it['dataset'] for it in items) if args.datasets is None or n in args.datasets]
    for name in names:
        i = next(k for k, ds in enumerate(bc.DATASETS) if ds[0] == name)
        csv_file, key = bc.DATASETS[i][1], bc.DATASETS[i][2]
        y_all = bc.load_labels(name, csv_file)
        texts_all, tcol, scol = bc.load_texts(name, csv_file)
        r, pe, _ = bc.load_scores(key, len(y_all))
        ok = (~np.isnan(r) & ~np.isnan(pe)) if r is not None else np.ones(len(y_all), dtype=bool)
        y = y_all[ok]; texts = [t for t, k in zip(texts_all, ok) if k]
        d = {'y': y, 'texts': texts, 'dev': bc.stratified_split(y, bc.SEED + i)}
        log(f'\n  {name}: fine-tuning on {int(d["dev"].sum())} development emails')
        bt.set_seed(bc.SEED)
        model, best_ep, va_f1, use_amp = bt.train_one(name, d, args, device, tok, log)
        idx = [k for k, it in enumerate(items) if it['dataset'] == name]
        keys, texts_att = [], []
        for k in idx:
            it = items[k]
            for c, body in conds[k].items():
                keys.append((k, c))
                texts_att.append(((it['subject'] + ' \n ' + body) if scol else body)[:bc.MAX_CHARS])
        enc = bt.encode(tok, texts_att, args.max_len)
        p = bt.predict(model, tok, enc, list(range(len(texts_att))), args.batch, device, use_amp)
        for (k, c), v in zip(keys, p):
            preds[(args.model, k, c)] = int(v)
        log(f'    best epoch {best_ep}, validation F1 {va_f1:.2f}%; classified {len(texts_att)} clean/attacked texts')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    sub_items = [it for it in items if it['dataset'] in names]
    keep = [k for k, it in enumerate(items) if it['dataset'] in names]
    remap = {k: n for n, k in enumerate(keep)}
    preds2 = {(m, remap[k], c): v for (m, k, c), v in preds.items()}
    df = ats.summarize(preds2, sub_items, [conds[k] for k in keep])
    df.to_csv(os.path.join(out, 'recall_by_condition.csv'), index=False)
    ats.print_summary(df)
    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'model': args.model, 'datasets': names,
               'epochs': args.epochs, 'max_len': args.max_len, 'batch': args.batch, 'lr': args.lr, 'seed': bc.SEED,
               'same_sample_as_robustness_results': same, 'python': platform.python_version(),
               'runtime_minutes': round((time.time() - t0) / 60, 1)}, open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    log(f'\n  Saved to {out}/ ({(time.time() - t0) / 60:.1f} minutes).')


if __name__ == '__main__':
    main()
