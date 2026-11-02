"""
Baseline_Transformer.py
=============================================================================
fine-tuned transformer baselines on EXACTLY
the same development / test halves as the proposed system.

  * Default model: XLM-RoBERTa base (multilingual: English and Arabic).
  * Arabic-specific model (AraBERT) as a second run, see below.
  * Each model is fine-tuned on the DEVELOPMENT half only (90% train / 10%
    validation, best epoch chosen on validation F1) and evaluated once on the
    TEST half. It is compared with the frozen system configuration, with
    paired-bootstrap 95% intervals for the F1 difference.
  * Cross-corpus: the model trained on one dataset is also tested on the
    other datasets' test halves (training-free systems should be compared
    here too).

Needs a GPU. Easiest: Google Colab (Runtime > Change runtime type > T4 GPU).
Put these in one folder and upload it (zip) to Colab:
    Baseline_Transformer.py, Baseline_Classical.py,
    Select_And_Freeze_Configuration.py, the six CSV files,
    score_cache_reasoning_only/, final_configuration/

Then in Colab:
    !pip -q install transformers
    !python Baseline_Transformer.py                       # XLM-R on all datasets (~30-45 min on a T4)
    !python Baseline_Transformer.py --model aubmindlab/bert-base-arabertv02 \
            --datasets "Salloum Arabic 300" "UAE Synthetic Arabic"   # AraBERT, Arabic only (~5 min)

Results go to transformer_results_<model>/.
"""
import os, sys, json, time, argparse, random, platform
from datetime import datetime
import numpy as np
import pandas as pd

import Baseline_Classical as bc            # same loading, split and metrics as the classical baseline

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import transformers

CROSS_PAIRS = [('Enron Full', 'MeAJOR 2025 (3,000)'), ('MeAJOR 2025 (3,000)', 'Enron Full'),
               ('Enron Full', 'Salloum English 300'), ('MeAJOR 2025 (3,000)', 'Salloum English 300'),
               ('Enron Full', 'Nazario Full'),
               ('Salloum Arabic 300', 'UAE Synthetic Arabic'), ('UAE Synthetic Arabic', 'Salloum Arabic 300')]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def encode(tok, texts, max_len):
    return tok(texts, truncation=True, max_length=max_len)


def batches(enc, labels, idx, bs, tok, shuffle, rng=None):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        feats = [{k: enc[k][i] for k in enc.keys()} for i in b]
        out = tok.pad(feats, return_tensors='pt')
        if labels is not None:
            out['labels'] = torch.tensor([int(labels[i]) for i in b])
        yield out


@torch.no_grad()
def predict(model, tok, enc, idx, bs, device, use_amp):
    model.eval(); preds = []
    for batch in batches(enc, None, idx, bs * 2, tok, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds).astype(bool) if preds else np.zeros(0, dtype=bool)


def train_one(name, d, args, device, tok, log):
    y = d['y']; dev_idx = np.where(d['dev'])[0]
    rng = np.random.RandomState(bc.SEED)
    # 90/10 stratified split of the development half into train / validation
    tr, va = [], []
    for cls in (0, 1):
        ids = dev_idx[y[dev_idx] == cls].copy(); rng.shuffle(ids)
        n_va = max(1, int(round(0.1 * len(ids)))); va += list(ids[:n_va]); tr += list(ids[n_va:])
    enc = encode(tok, d['texts'], args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * int(np.ceil(len(tr) / args.batch))
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_f1, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); t1 = time.time(); losses = []
        for step, batch in enumerate(batches(enc, y, tr, args.batch, tok, shuffle=True, rng=np.random.RandomState(bc.SEED + ep))):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); losses.append(float(loss))
        va_pred = predict(model, tok, enc, va, args.batch, device, use_amp)
        va_f1 = bc.metrics(y[va], va_pred)['F1']
        log(f'    epoch {ep}: train loss {np.mean(losses):.4f} | validation F1 {va_f1:.2f}% | {time.time() - t1:.0f}s')
        if va_f1 > best_f1:
            best_f1, best_ep = va_f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_ep, best_f1, use_amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='xlm-roberta-base')
    ap.add_argument('--datasets', nargs='*', default=None, help='names as in the DATASETS list; default all')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    short = args.model.rstrip('/').split('/')[-1]
    out = args.out or f'transformer_results_{short}'
    os.makedirs(os.path.join(out, 'predictions'), exist_ok=True)
    log_f = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')

    def log(s=''):
        print(s); log_f.write(s + '\n'); log_f.flush()

    set_seed(bc.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    log('\n' + '=' * 78)
    log(f'  TRANSFORMER BASELINE: {args.model}  on the same development / test halves')
    log(f'  {datetime.now():%Y-%m-%d %H:%M} | device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ' (no GPU: this will be very slow)'))
    log(f'  split: {bc.SPLIT_SOURCE}')
    log('=' * 78)

    frozen_path = os.path.join('final_configuration', 'frozen_config.json')
    if os.path.exists(frozen_path):
        fz = json.load(open(frozen_path)); SYS_PE = fz['weights_percent']['perspective']; SYS_T = fz['threshold']
    else:
        SYS_PE, SYS_T = 10, 30
    SYS_NAME = f'Proposed system (R {100 - SYS_PE}% / Pe {SYS_PE}%, T={SYS_T})'

    data = {}
    for i, (name, csv_file, key, use_for_sel) in enumerate(bc.DATASETS):
        if not os.path.exists(csv_file):
            continue
        y_all = bc.load_labels(name, csv_file)
        texts, tcol, scol = bc.load_texts(name, csv_file)
        r, pe, _ = bc.load_scores(key, len(y_all))
        ok = (~np.isnan(r) & ~np.isnan(pe)) if r is not None else np.ones(len(y_all), dtype=bool)
        y = y_all[ok]; texts = [t for t, k in zip(texts, ok) if k]
        d = {'y': y, 'texts': texts, 'dev': bc.stratified_split(y, bc.SEED + i), 'key': key, 'sel': use_for_sel}
        if r is not None:
            d['sys_pred'] = ((1 - SYS_PE / 100.0) * r[ok] + (SYS_PE / 100.0) * pe[ok]) >= SYS_T
        data[name] = d
    todo = [n for n in data if (args.datasets is None or n in args.datasets) and len(np.unique(data[n]['y'])) == 2]
    log(f'  datasets to fine-tune on: {todo}')
    tok = AutoTokenizer.from_pretrained(args.model)

    in_rows, cross_rows, boot_rows = [], [], []
    rng = np.random.RandomState(bc.SEED)
    for name in todo:
        d = data[name]; y = d['y']; te = np.where(~d['dev'])[0]
        log(f'\n  {name}: fine-tuning on {int(d["dev"].sum())} development emails, testing on {len(te)}')
        set_seed(bc.SEED)
        model, best_ep, va_f1, use_amp = train_one(name, d, args, device, tok, log)
        enc = encode(tok, d['texts'], args.max_len)
        pred = predict(model, tok, enc, te, args.batch, device, use_amp)
        res = bc.metrics(y[te], pred)
        in_rows.append({'Dataset': name, 'Model': args.model, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()},
                        'Best_epoch': best_ep, 'Validation_F1': round(va_f1, 2)})
        pdf = pd.DataFrame({'test_row': te, 'label': y[te], args.model: pred.astype(int)})
        log(f"    TEST: F1 {res['F1']:.2f}%  Precision {res['Precision']:.2f}%  Recall {res['Recall']:.2f}%  FP {res['FP']}  FN {res['FN']}")
        if 'sys_pred' in d:
            sp = d['sys_pred'][te]; sres = bc.metrics(y[te], sp)
            in_rows.append({'Dataset': name, 'Model': SYS_NAME, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in sres.items()}})
            pdf[SYS_NAME] = sp.astype(int)
            m, lo, hi, _ = bc.paired_boot(y[te], pred, sp, rng)
            boot_rows.append({'Dataset': name, 'Comparison': f'{args.model} minus system', 'F1_diff_pp': round(m, 2),
                              'CI95_low': round(lo, 2), 'CI95_high': round(hi, 2),
                              'Verdict': 'clear difference' if (lo > 0 or hi < 0) else 'no clear difference'})
            log(f"    system on the same test half: F1 {sres['F1']:.2f}%  |  model minus system: {bc.fmt_ci(m, lo, hi)}")
        pdf.to_csv(os.path.join(out, 'predictions', f"{d['key']}_test_predictions.csv"), index=False)
        for src, tgt in CROSS_PAIRS:
            if src != name or tgt not in data:
                continue
            dt = data[tgt]; tte = np.where(~dt['dev'])[0]
            tenc = encode(tok, dt['texts'], args.max_len)
            cp = predict(model, tok, tenc, tte, args.batch, device, use_amp)
            cres = bc.metrics(dt['y'][tte], cp)
            cross_rows.append({'Train_on': src, 'Test_on': tgt, 'Model': args.model,
                               **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in cres.items()}})
            msg = f"Recall {cres['Recall']:.2f}%" if len(np.unique(dt['y'])) < 2 else f"F1 {cres['F1']:.2f}%"
            log(f'    cross-corpus -> {tgt}: {msg}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        bc.write_csv(os.path.join(out, 'in_domain_results.csv'), in_rows)      # saved after every dataset
        bc.write_csv(os.path.join(out, 'cross_corpus_results.csv'), cross_rows)
        bc.write_csv(os.path.join(out, 'bootstrap_vs_system.csv'), boot_rows)

    json.dump({'created': datetime.now().isoformat(timespec='seconds'), 'model': args.model, 'epochs': args.epochs,
               'max_len': args.max_len, 'batch': args.batch, 'lr': args.lr, 'seed': bc.SEED, 'split_source': bc.SPLIT_SOURCE,
               'model_selection': 'best epoch by F1 on a 10% stratified validation slice of the development half',
               'device': str(device), 'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                                                   'transformers': transformers.__version__},
               'runtime_minutes': round((time.time() - t0) / 60, 1)}, open(os.path.join(out, 'run_info.json'), 'w'), indent=2)
    log(f'\n  Saved to {out}/  ({(time.time() - t0) / 60:.1f} minutes).\n')


if __name__ == '__main__':
    main()
