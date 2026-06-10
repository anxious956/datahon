"""
Datathon 2026 — Faz 11: PSEUDO-LABELING (riskli; geç-yıl 2025-2026 hedefler).

FİKİR: Test'te ensemble'ın EN KARARLI olduğu (3 GBM'in std'si düşük) örnekleri pseudo-label
yapıp train'e kat, yeniden eğit. 2025-2026 test örnekleri train'de az → orada değerli olabilir.

⚠️ LEAKAGE-GÜVENLİ TASARIM (kritik):
- Pseudo-label'lar fold-DIŞI üretilir. Fold k için:
    1) cat/lgb/xgb yalnız fold-k-train'de eğitilir (eval_set = fold-k-val, mevcut pipeline gibi),
       test tahmini alınır → bu fold'un pseudo kaynağı.
    2) 3 modelin test-tahmin std'si DÜŞÜK örnekler seçilir (en kararlı); pseudo_y = 3 model ort.
    3) Her model (fold-k-train + güvenli-pseudo-test) ile YENİDEN eğitilir, fold-k-val tahmin edilir.
  Pseudo etiketler yalnız fold-k-train modelinden geldiği için fold-k-val'a SIZMAZ → OOF dürüst.
- Test submission tahmini = fold-augmented modellerin ortalaması (mevcut pipeline ile tutarlı).

⚠️ DİKKAT: Pseudo-labeling OOF'u yine de YAPAY iyileştirebilir (model kendi tahminini doğrular).
   Kazanç MUTLAKA yarın public'te doğrulanır; OOF'a TEK BAŞINA GÜVENME (WARPLAN Adım 4).

GİRDİ: experiments/faz9_tuned_params.json (Adım 1 tuned paramlar). Önce faz9 koş.
ÇIKTI: oof_blend_pseudo.npy, test_blend_pseudo.npy, faz11_pseudo_log.json, submission.
AYAR: CONF_Q (vars. 0.5 = en kararlı %50), PSEUDO_SEED (vars. 42).
"""
import os, json, datetime
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

from faz1_catboost_baseline import YEAR, ID, TARGET, SEED
from faz2_text_meta import metrics
from faz3_ensemble import tune_blend
from faz9_tune import prepare, CAT_ITERS, GBM_ESTIMATORS, OD_WAIT

CONF_Q = float(os.environ.get('CONF_Q', 0.5))      # en kararlı bu oran kadar test örneği
PSEUDO_SEED = int(os.environ.get('PSEUDO_SEED', SEED))


def fit_cat(Xtr, ytr, Xval, yval, Xte, cat_idx, params):
    p = dict(loss_function='RMSE', eval_metric='RMSE', iterations=CAT_ITERS, od_type='Iter',
             od_wait=OD_WAIT, random_seed=PSEUDO_SEED, verbose=False, allow_writing_files=False)
    p.update(params)
    m = CatBoostRegressor(**p)
    m.fit(Pool(Xtr, ytr, cat_features=cat_idx),
          eval_set=Pool(Xval, yval, cat_features=cat_idx), use_best_model=True)
    return np.clip(m.predict(Xval), 0, 100), np.clip(m.predict(Xte), 0, 100)


def fit_lgb(Xtr, ytr, Xval, yval, Xte, cat_cols, params):
    p = dict(objective='regression', metric='l2', n_estimators=GBM_ESTIMATORS,
             random_state=PSEUDO_SEED, n_jobs=-1, verbosity=-1)
    p.update(params)
    m = LGBMRegressor(**p)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], categorical_feature=cat_cols,
          callbacks=[early_stopping(OD_WAIT, verbose=False), log_evaluation(0)])
    return np.clip(m.predict(Xval), 0, 100), np.clip(m.predict(Xte), 0, 100)


def fit_xgb(Xtr, ytr, Xval, yval, Xte, params):
    p = dict(objective='reg:squarederror', eval_metric='rmse', n_estimators=GBM_ESTIMATORS,
             tree_method='hist', enable_categorical=True, random_state=PSEUDO_SEED, n_jobs=-1,
             early_stopping_rounds=OD_WAIT)
    p.update(params)
    m = XGBRegressor(**p)
    m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return np.clip(m.predict(Xval), 0, 100), np.clip(m.predict(Xte), 0, 100)


def main():
    P = prepare()
    y, w, years, te_prop, folds = P['y'], P['w'], P['years'], P['te_prop'], P['folds']
    cat_cols, cat_idx = P['cat_cols'], P['cat_idx']
    Xc, Xc_t, Xg, Xg_t = P['Xc'], P['Xc_t'], P['Xg'], P['Xg_t']
    exp_dir, sub_dir, test = P['exp_dir'], P['sub_dir'], P['test']
    test_years = test[YEAR].values
    n_te = len(Xc_t)

    tp = json.load(open(os.path.join(exp_dir, 'faz9_tuned_params.json')))
    bp = {k: tp[k]['best_params'] for k in ('cat', 'lgb', 'xgb')}
    print(f"[pseudo] tuned paramlar yüklendi | CONF_Q={CONF_Q} (en kararlı %{CONF_Q*100:.0f}) "
          f"| seed={PSEUDO_SEED}")

    base_oof = {k: np.zeros(len(Xc)) for k in ('cat', 'lgb', 'xgb')}
    ps_oof = {k: np.zeros(len(Xc)) for k in ('cat', 'lgb', 'xgb')}
    ps_te = {k: np.zeros(n_te) for k in ('cat', 'lgb', 'xgb')}
    pseudo_count = []; pseudo_late = []

    for fi, (tr, va) in enumerate(folds):
        # 1) Baz modeller (pseudo'suz) — fold-k-train; test tahmini = pseudo kaynağı
        cv_c, ct_c = fit_cat(Xc.iloc[tr], y[tr], Xc.iloc[va], y[va], Xc_t, cat_idx, bp['cat'])
        cv_l, ct_l = fit_lgb(Xg.iloc[tr], y[tr], Xg.iloc[va], y[va], Xg_t, cat_cols, bp['lgb'])
        cv_x, ct_x = fit_xgb(Xg.iloc[tr], y[tr], Xg.iloc[va], y[va], Xg_t, bp['xgb'])
        base_oof['cat'][va] = cv_c; base_oof['lgb'][va] = cv_l; base_oof['xgb'][va] = cv_x

        # 2) Kararlılık: 3 model test-tahmin std'si DÜŞÜK = en kararlı. pseudo_y = ortalama.
        te_stack = np.vstack([ct_c, ct_l, ct_x])
        std_te = te_stack.std(axis=0)
        pseudo_y = te_stack.mean(axis=0)
        thr = np.quantile(std_te, CONF_Q)
        mask = std_te <= thr
        pseudo_count.append(int(mask.sum()))
        pseudo_late.append(int(mask[np.isin(test_years, [2025, 2026])].sum()))

        # 3) Augmented retrain (fold-k-train + güvenli pseudo-test), fold-k-val tahmin (dürüst OOF)
        Xc_aug = pd.concat([Xc.iloc[tr], Xc_t.iloc[mask]], ignore_index=True)
        Xg_aug = pd.concat([Xg.iloc[tr], Xg_t.iloc[mask]], ignore_index=True)
        y_aug = np.concatenate([y[tr], pseudo_y[mask]])
        pv_c, pt_c = fit_cat(Xc_aug, y_aug, Xc.iloc[va], y[va], Xc_t, cat_idx, bp['cat'])
        pv_l, pt_l = fit_lgb(Xg_aug, y_aug, Xg.iloc[va], y[va], Xg_t, cat_cols, bp['lgb'])
        pv_x, pt_x = fit_xgb(Xg_aug, y_aug, Xg.iloc[va], y[va], Xg_t, bp['xgb'])
        ps_oof['cat'][va] = pv_c; ps_oof['lgb'][va] = pv_l; ps_oof['xgb'][va] = pv_x
        ps_te['cat'] += pt_c / len(folds); ps_te['lgb'] += pt_l / len(folds); ps_te['xgb'] += pt_x / len(folds)
        print(f"  fold {fi}: pseudo={mask.sum():4d} (2025-26: {pseudo_late[-1]:3d}, thr_std={thr:.3f})")

    # --- Baz (pseudo'suz) blend referansı ---
    _, (bc, bl, bx) = tune_blend([base_oof['cat'], base_oof['lgb'], base_oof['xgb']], y, w)
    base_blend = np.clip(bc*base_oof['cat'] + bl*base_oof['lgb'] + bx*base_oof['xgb'], 0, 100)
    _, base_w, _ = metrics(y, base_blend, w, years, te_prop)

    # --- Pseudo blend ---
    _, (pc, pl, px) = tune_blend([ps_oof['cat'], ps_oof['lgb'], ps_oof['xgb']], y, w)
    ps_blend = np.clip(pc*ps_oof['cat'] + pl*ps_oof['lgb'] + px*ps_oof['xgb'], 0, 100)
    ps_te_blend = np.clip(pc*ps_te['cat'] + pl*ps_te['lgb'] + px*ps_te['xgb'], 0, 100)
    p_plain, p_w, p_by = metrics(y, ps_blend, w, years, te_prop)

    print("\n================ FAZ 11 SONUÇ (pseudo-labeling, ağırlıklı OOF) ================")
    print(f"Baz blend (pseudo'suz) wOOF : {base_w:.4f}")
    print(f"Pseudo blend wOOF           : {p_w:.4f}  (düz {p_plain:.4f})")
    print(f"  kazanç vs baz             : {base_w - p_w:+.4f}  ⚠️ OOF şişebilir → PUBLIC HAKEMİ")
    print(f"  ort. pseudo örnek/fold    : {np.mean(pseudo_count):.0f} (2025-26: {np.mean(pseudo_late):.0f})")
    print("\nYıl-bazlı pseudo blend OOF MSE (geç-yıl hedefi):")
    print(p_by.round(3).to_string())

    np.save(os.path.join(exp_dir, 'oof_blend_pseudo.npy'), ps_blend)
    np.save(os.path.join(exp_dir, 'test_blend_pseudo.npy'), ps_te_blend)
    log = {'phase': 'faz11_pseudo_labeling', 'date': datetime.date.today().isoformat(),
           'conf_q': CONF_Q, 'seed': PSEUDO_SEED, 'pseudo_count_per_fold': pseudo_count,
           'pseudo_late_per_fold': pseudo_late,
           'base_blend': {'weighted': base_w, 'weights': {'cat': bc, 'lgb': bl, 'xgb': bx}},
           'pseudo_blend': {'plain': p_plain, 'weighted': p_w, 'by_year': p_by.round(4).to_dict(),
                            'weights': {'cat': pc, 'lgb': pl, 'xgb': px}},
           'gain_vs_base': base_w - p_w, 'WARN': 'OOF inflation possible; validate on public.'}
    with open(os.path.join(exp_dir, 'faz11_pseudo_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: ps_te_blend})
    fname = f"sub_{datetime.date.today().isoformat()}_pseudo_woof{p_w:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof/test_blend_pseudo.npy + faz11_pseudo_log.json + {fname}")


if __name__ == '__main__':
    main()
