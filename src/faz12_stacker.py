"""
Datathon 2026 — Faz 12: ÖĞRENİLMİŞ STACKER (savaş planı #5, EN RİSKLİ, son).

Basit ağırlık-blend yerine: tüm baz modellerin OOF'ları üstünde meta-model (Ridge / NNLS / küçük LGBM)
öğren. Baz OOF'lar zaten fold-dışı -> meta-CV aynı fold'larda leakage-free.

⚠️ OOF-ŞİŞME riski (embedding dersi): meta-model OOF gürültüsüne overfit edip wOOF'u yapay düşürebilir.
Bu yüzden: (a) güçlü regularizasyon, (b) basit blend ile KIYAS, (c) kazanç PUBLIC'te doğrulanır.

Baz model registry: (oof_*.npy, test_*.npy) çiftleri — VAR OLANLAR otomatik dahil (transformerlar
Kaggle'dan gelince oof_<tag>_train.npy + <tag>_test.npy olarak eklenir, kendiliğinden girer).
"""
import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.optimize import nnls

from faz1_catboost_baseline import resolve_paths, ID, TARGET, YEAR
from faz2_text_meta import make_folds, metrics

# (isim, oof_dosya, test_dosya). Yoksa atlanır. transformer tag'leri sona eklenebilir.
REGISTRY = [
    ('tuned_cat', 'oof_tuned_cat.npy', 'test_tuned_cat.npy'),
    ('tuned_lgb', 'oof_tuned_lgb.npy', 'test_tuned_lgb.npy'),
    ('tuned_xgb', 'oof_tuned_xgb.npy', 'test_tuned_xgb.npy'),
    ('bag_cat', 'oof_bag_cat.npy', 'test_bag_cat.npy'),
    ('bag_lgb', 'oof_bag_lgb.npy', 'test_bag_lgb.npy'),
    ('bag_xgb', 'oof_bag_xgb.npy', 'test_bag_xgb.npy'),
    ('berturk', 'oof_berturk_train.npy', 'berturk_test.npy'),
    ('bert128k', 'oof_bert128k_train.npy', 'bert128k_test.npy'),
    ('electra', 'oof_electra_train.npy', 'electra_test.npy'),
    ('xlmr', 'oof_xlmr_train.npy', 'xlmr_test.npy'),
]


def cv_meta(fit_pred, OOF, y, folds, w):
    """Meta-model'i fold-dışı değerlendir: her fold meta-train'de fit, va'da tahmin -> wOOF."""
    pred = np.zeros(len(y))
    for tr, va in folds:
        pred[va] = fit_pred(OOF[tr], y[tr], OOF[va])
    pred = np.clip(pred, 0, 100)
    return pred, float(np.sum(w * (y - pred) ** 2) / np.sum(w))


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    names, oofs, tests = [], [], []
    for nm, of, tf in REGISTRY:
        op, tp = os.path.join(exp_dir, of), os.path.join(exp_dir, tf)
        if os.path.exists(op) and os.path.exists(tp):
            names.append(nm); oofs.append(np.clip(np.load(op).astype('float64'), 0, 100))
            tests.append(np.clip(np.load(tp).astype('float64'), 0, 100))
    OOF = np.vstack(oofs).T; TST = np.vstack(tests).T
    print(f"[stacker] {len(names)} baz model: {names}")
    print("  OOF korelasyon:"); print(pd.DataFrame(OOF, columns=names).corr().round(3).to_string())

    results = {}
    # 1) Ridge (birkaç alpha)
    for alpha in [1.0, 10.0, 50.0, 100.0]:
        def fp(Xt, yt, Xv, a=alpha):
            m = Ridge(alpha=a, positive=True).fit(Xt, yt); return m.predict(Xv)
        _, wm = cv_meta(fp, OOF, y, folds, w); results[f'ridge_a{alpha:g}'] = wm
    # 2) NNLS (negatif olmayan ağırlık, normalize)
    def fp_nnls(Xt, yt, Xv):
        coef, _ = nnls(Xt, yt); s = coef.sum() or 1.0; return Xv @ (coef / s) * (yt.mean() / ((Xt @ (coef/s)).mean() or 1))
    _, wm_nnls = cv_meta(fp_nnls, OOF, y, folds, w); results['nnls'] = wm_nnls
    # referans: en iyi tek model + eşit blend
    best_single = min((float(np.sum(w*(y-OOF[:,i])**2)/np.sum(w)), names[i]) for i in range(len(names)))
    eq = np.clip(OOF.mean(1), 0, 100); wm_eq = float(np.sum(w*(y-eq)**2)/np.sum(w))

    print("\n================ FAZ 12 STACKER SONUÇ (wOOF) ================")
    print(f"  en iyi TEK model : {best_single[1]} {best_single[0]:.4f}")
    print(f"  eşit-blend       : {wm_eq:.4f}")
    for k, v in sorted(results.items(), key=lambda x: x[1]): print(f"  {k:12s}: {v:.4f}")
    best_name = min(results, key=results.get); best_w = results[best_name]
    print(f"  -> en iyi stacker: {best_name} {best_w:.4f}  (⚠️ OOF şişebilir, public şart)")

    # Final test tahmini: en iyi yöntemi TÜM train'de fit edip test'e uygula
    if best_name.startswith('ridge'):
        a = float(best_name.split('_a')[1]); M = Ridge(alpha=a, positive=True).fit(OOF, y)
        test_pred = np.clip(M.predict(TST), 0, 100); coef = dict(zip(names, M.coef_.round(4)))
    else:
        coef_raw, _ = nnls(OOF, y); s = coef_raw.sum() or 1.0
        test_pred = np.clip(TST @ (coef_raw/s) * (y.mean()/((OOF @ (coef_raw/s)).mean() or 1)), 0, 100)
        coef = dict(zip(names, (coef_raw/s).round(4)))
    print(f"  final ağırlıklar: {coef}")

    log = {'phase': 'faz12_stacker', 'date': datetime.date.today().isoformat(), 'base_models': names,
           'results_woof': results, 'best_single': {'name': best_single[1], 'woof': best_single[0]},
           'equal_blend_woof': wm_eq, 'best_method': best_name, 'best_woof': best_w,
           'final_coef': {k: float(v) for k, v in coef.items()},
           'WARNING': 'meta-model OOF sismesi mumkun; public dogrulamasi sart'}
    with open(os.path.join(exp_dir, 'faz12_stacker_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = f"sub_{datetime.date.today().isoformat()}_stacker_{best_name}_woof{best_w:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: test_pred}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] faz12_stacker_log.json + {fname}")


if __name__ == '__main__':
    main()
