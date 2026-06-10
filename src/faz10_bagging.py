"""
Datathon 2026 — Faz 10: SEED BAGGING (savaş planı #2, bedava+güvenli varyans düşürme).

Faz 9 tuned paramları SABİT; her GBM'i N farklı seed'le yeniden eğit, OOF + test tahminlerini
seed'ler arası ORTALA. Fold yapısı SABİT (SEED=42 fold'ları — hizalama bozulmasın); değişen yalnız
MODEL seed'i (bagging/random_strength/feature-subsample realizasyonu) -> varyans düşer, +0.1-0.3.

Girdi: experiments/faz9_tuning_log.json (best_params). Çıktı: bagged OOF/test + blend + log + submission.
SEEDS env ile (default '42,1337,2024'). Ağır: len(SEEDS)*3*5 model fit.
"""
import os, json, datetime, time
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, metrics
from faz3_ensemble import tune_blend
from faz9_tuning import build_matrices, cat_cv, lgb_cv, xgb_cv, wmse


def cat_params(bp, seed):
    return dict(loss_function='RMSE', eval_metric='RMSE', iterations=3000, random_seed=seed,
                od_type='Iter', od_wait=200, verbose=False, allow_writing_files=False, **bp)

def lgb_params(bp, seed):
    return dict(objective='regression', metric='l2', n_estimators=5000, random_state=seed,
                n_jobs=-1, verbosity=-1, subsample_freq=1, **bp)

def xgb_params(bp, seed):
    return dict(objective='reg:squarederror', eval_metric='rmse', n_estimators=5000, tree_method='hist',
                enable_categorical=True, random_state=seed, n_jobs=-1, early_stopping_rounds=200, **bp)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    bp = json.load(open(os.path.join(exp_dir, 'faz9_tuning_log.json')))['best_params']
    Xc, Xc_t, Xg, Xg_t, cat_cols, cat_idx = build_matrices(train, test, y, folds, exp_dir)
    seeds = [int(s) for s in os.environ.get('SEEDS', '42,1337,2024').split(',')]
    print(f"[bagging] seeds={seeds} | feat {Xc.shape}")

    acc = {k: {'oof': np.zeros(len(y)), 'test': np.zeros(len(test))} for k in ('cat', 'lgb', 'xgb')}
    for si, sd in enumerate(seeds):
        t0 = time.time()
        oc, tc = cat_cv(cat_params(bp['cat'], sd), Xc, y, Xc_t, cat_idx, folds, w, predict_test=True)
        ol, tl = lgb_cv(lgb_params(bp['lgb'], sd), Xg, y, Xg_t, cat_cols, folds, w, predict_test=True)
        ox, tx = xgb_cv(xgb_params(bp['xgb'], sd), Xg, y, Xg_t, folds, w, predict_test=True)
        for k, (o, t) in [('cat', (oc, tc)), ('lgb', (ol, tl)), ('xgb', (ox, tx))]:
            acc[k]['oof'] += o / len(seeds); acc[k]['test'] += t / len(seeds)
        print(f"  seed {sd}: cat {wmse(y,oc,w):.3f} lgb {wmse(y,ol,w):.3f} xgb {wmse(y,ox,w):.3f} ({time.time()-t0:.0f}s)")

    print("\n--- Bagged tekil model (düz | ağırlıklı) ---")
    for k in ('cat', 'lgb', 'xgb'):
        p, wt, _ = metrics(y, np.clip(acc[k]['oof'], 0, 100), w, train[YEAR], te_prop)
        print(f"  {k}: {p:.4f} | {wt:.4f}")
    oc, ol, ox = (np.clip(acc[k]['oof'], 0, 100) for k in ('cat', 'lgb', 'xgb'))
    tc, tl, tx = (np.clip(acc[k]['test'], 0, 100) for k in ('cat', 'lgb', 'xgb'))

    _, (wc, wl, wx) = tune_blend([oc, ol, ox], y, w)
    oof_bl = np.clip(wc*oc + wl*ol + wx*ox, 0, 100)
    test_bl = np.clip(wc*tc + wl*tl + wx*tx, 0, 100)
    bp_, bw, bby = metrics(y, oof_bl, w, train[YEAR], te_prop)
    ref = json.load(open(os.path.join(exp_dir, 'faz9_tuning_log.json')))['blend']['weighted']
    print(f"\n================ FAZ 10 (TUNED+BAGGED) SONUÇ ================")
    print(f"Blend cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f} | wOOF={bw:.4f} (düz {bp_:.4f})")
    print(f"  Faz 9 tuned wOOF {ref:.4f} -> bagging kazancı {ref - bw:+.4f}")
    print("\nYıl-bazlı:"); print(bby.round(3).to_string())

    for k, o, t in [('cat', oc, tc), ('lgb', ol, tl), ('xgb', ox, tx)]:
        np.save(os.path.join(exp_dir, f'oof_bag_{k}.npy'), o); np.save(os.path.join(exp_dir, f'test_bag_{k}.npy'), t)
    np.save(os.path.join(exp_dir, 'oof_bag_blend.npy'), oof_bl)
    log = {'phase': 'faz10_seed_bagging', 'date': datetime.date.today().isoformat(), 'seeds': seeds,
           'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
           'blend': {'plain': bp_, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'ref_faz9_tuned': ref, 'gain_vs_faz9': ref - bw}
    with open(os.path.join(exp_dir, 'faz10_bagging_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = f"sub_{datetime.date.today().isoformat()}_tuned_bagged_woof{bw:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: test_bl}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof/test_bag_*.npy + faz10_bagging_log.json + {fname}")


if __name__ == '__main__':
    main()
