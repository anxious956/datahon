"""
Datathon 2026 — Faz 17: MEGA-STACKER (agresif mod, transformer-ready).

ANCHOR: 0.8*stacker_v2 + 0.2*tabpfn = public 83.185 (KANITLI). Bunun wOOF'u = kalibrasyon.
Hedef: anchor'ı wOOF'ta MEANINGFUL geçen varyant bul (ama wOOF güvenilmez -> sadece BÜYÜK fark anlamlı).

Üyeler (oof+test mevcut olanlar; transformerlar gelince oof_<tag>_train.npy + <tag>_test.npy AUTO girer):
  stacker_v2 (güçlü GBM+text base), tabpfn (bağımsız tabular), berturk (text 0.69 corr),
  tuned_cat (GBM çeşitlilik), [bert128k, electra ...].

Yöntemler: anchor (sabit), NNLS, Ridge(positive), yıl-farkında LGBM-stacker (4).
Her biri için wOOF + anchor'a göre delta. Aday submission'lar yazılır; KARAR public ile.
"""
import os, json, datetime
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from scipy.optimize import nnls

from faz1_catboost_baseline import resolve_paths, ID, TARGET, YEAR
from faz2_text_meta import make_folds, metrics

# (isim, oof_dosya, test_dosya). featB test csv'den; diğerleri npy. Yoksa atlanır.
MEMBERS = [
    ('stacker_v2', 'oof_stacker_v2.npy', 'test_stacker_v2.npy'),
    ('tabpfn',     'oof_tabpfn_train.npy', 'tabpfn_test.npy'),
    ('berturk',    'oof_berturk_train.npy', 'berturk_test.npy'),
    ('tuned_cat',  'oof_tuned_cat.npy', 'test_tuned_cat.npy'),
    ('bert128k',   'oof_bert128k_train.npy', 'bert128k_test.npy'),   # transformer (gelince)
    ('electra',    'oof_electra_train.npy', 'electra_test.npy'),     # transformer (gelince)
]


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values; yr = train[YEAR].values
    folds, _ = make_folds(train)
    trp = train[YEAR].value_counts(normalize=True); tep = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda v: tep.get(v, 0.0) / trp.get(v, np.nan)).values)
    def wm(p): return float(np.sum(w * (y - p) ** 2) / np.sum(w))

    names, OOF, TST = [], [], []
    for nm, of, tf in MEMBERS:
        op, tp = os.path.join(exp_dir, of), os.path.join(exp_dir, tf)
        if os.path.exists(op) and os.path.exists(tp):
            names.append(nm); OOF.append(np.clip(np.load(op).astype('float64'), 0, 100))
            TST.append(np.clip(np.load(tp).astype('float64'), 0, 100))
    OOF = np.vstack(OOF).T; TST = np.vstack(TST).T
    print(f"[mega] {len(names)} üye: {names}")
    print("OOF korelasyon:"); print(pd.DataFrame(OOF, columns=names).corr().round(3).to_string())
    print("\nTekil wOOF:")
    for i, nm in enumerate(names): print(f"  {nm:11s}: {wm(OOF[:,i]):.4f}")

    # ---------- ANCHOR (kanıtlı 83.185) ----------
    si, ti = names.index('stacker_v2'), names.index('tabpfn')
    anc_oof = np.clip(0.8*OOF[:,si] + 0.2*OOF[:,ti], 0, 100)
    anc_w = wm(anc_oof)
    print(f"\n>>> ANCHOR (0.8*stacker_v2+0.2*tabpfn): wOOF={anc_w:.4f}  == public 83.185 (kalibrasyon)")

    results = {'anchor_0.8_0.2': anc_w}
    cands = {}  # isim -> test_pred

    # ---------- NNLS (fold-dışı) ----------
    def cv_pred(fit_fn):
        pr = np.zeros(len(y))
        for tr, va in folds: pr[va] = fit_fn(OOF[tr], y[tr], OOF[va])
        return np.clip(pr, 0, 100)
    def nnls_fit(Xt, yt, Xv):
        c, _ = nnls(Xt, yt); s = c.sum() or 1; return Xv @ (c/s) * (yt.mean()/((Xt@(c/s)).mean() or 1))
    results['nnls'] = wm(cv_pred(nnls_fit))
    c_full, _ = nnls(OOF, y); s = c_full.sum() or 1
    cands['nnls'] = np.clip(TST @ (c_full/s) * (y.mean()/((OOF@(c_full/s)).mean() or 1)), 0, 100)

    # ---------- Ridge(positive) birkaç alpha ----------
    for a in [5.0, 20.0, 60.0]:
        results[f'ridge_a{a:g}'] = wm(cv_pred(lambda Xt,yt,Xv,al=a: Ridge(alpha=al,positive=True).fit(Xt,yt).predict(Xv)))
    bestR = min([k for k in results if k.startswith('ridge')], key=lambda k: results[k])
    aR = float(bestR.split('_a')[1]); M = Ridge(alpha=aR, positive=True).fit(OOF, y)
    cands[bestR] = np.clip(M.predict(TST), 0, 100)
    print(f"  ridge ağırlık ({bestR}): {dict(zip(names, M.coef_.round(3)))}")

    # ---------- YIL-FARKINDA LGBM stacker (4) ----------
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    yr_oh = pd.get_dummies(pd.Series(yr).astype(str)).values.astype('float64')
    te_yr_oh = pd.get_dummies(test[YEAR].astype(str)).reindex(
        columns=pd.Series(yr).astype(str).unique(), fill_value=0).values.astype('float64')
    Xs = np.hstack([OOF, yr_oh]); Xs_t = np.hstack([TST, te_yr_oh])
    pr = np.zeros(len(y)); te_acc = np.zeros(len(test))
    for tr, va in folds:
        m = LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=50,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                          random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xs[tr], y[tr], eval_set=[(Xs[va], y[va])],
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        pr[va] = m.predict(Xs[va]); te_acc += m.predict(Xs_t) / len(folds)
    results['lgb_yearaware'] = wm(np.clip(pr, 0, 100))
    cands['lgb_yearaware'] = np.clip(te_acc, 0, 100)

    print("\n================ FAZ 17 MEGA-STACKER (wOOF) ================")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:18s}: {v:.4f}   (anchor {anc_w:.4f}'e göre {anc_w-v:+.4f})")

    # ---------- Aday submission'lar (anchor'ı geçen wOOF'lular) ----------
    sv2_test = TST[:, si]; tab_test = TST[:, ti]
    cands['anchor'] = np.clip(0.8*sv2_test + 0.2*tab_test, 0, 100)  # = 83.185 (referans)
    saved = []
    for nm, tp in cands.items():
        fn = f"sub_{datetime.date.today().isoformat()}_mega_{nm}.csv"
        pd.DataFrame({ID: test[ID].values, TARGET: tp}).to_csv(os.path.join(sub_dir, fn), index=False)
        saved.append(fn)
    log = {'phase': 'faz17_megastacker', 'date': datetime.date.today().isoformat(), 'members': names,
           'results_woof': results, 'anchor_woof': anc_w,
           'note': 'anchor=public 83.185 kalibrasyon; wOOF guvenilmez, BUYUK delta + public sart'}
    json.dump(log, open(os.path.join(exp_dir, 'faz17_megastacker_log.json'), 'w'), indent=2, ensure_ascii=False)
    print(f"\n[kayıt] {len(saved)} aday + faz17_megastacker_log.json")
    print("Adaylar:", saved)


if __name__ == '__main__':
    main()
