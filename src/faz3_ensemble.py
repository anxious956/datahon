"""
Datathon 2026 — Faz 3 İSKELET: CatBoost + LightGBM + XGBoost ensemble (OOF-blend).

⚠️ HENÜZ ÇALIŞTIRMA. Faz 2.2 embedding VERDICT'i (KEEP/DROP) bekleniyor.
   (Lokalde çalışır: lightgbm/xgboost pip'ten kurulur, HF gerekmez.)

TASARIM:
- 3 GBM, AYNI feature seti + AYNI leakage-free fold'lar (yıl×hedef-desil, SEED=42):
    sayısal + kategorik + text_meta(word+char, Faz 2.1) [+ emb_meta (KEEP ise)]
  CatBoost/LightGBM/XGBoost farklı hata yapar -> blend kazandırır.
- KEEP/DROP ESNEK: emb_meta dosyaları (experiments/emb_e5_{train,test}.npy) VARSA otomatik
  eklenir (KEEP), yoksa text_meta ile gidilir (DROP). Embedding'i lokalde üretemediğimiz için
  (HF bloklu) KEEP path'inde Kaggle'da üretilen emb cache'i experiments/'e konur.
- Kategorik işleme model-bazlı: CatBoost native (str), LightGBM 'category' dtype,
  XGBoost enable_categorical=True. NaN'ı üçü de native yönetir.
- BLEND: OOF üzerinde (public LB'de DEĞİL — köprümüz var) ağırlık taraması; hedef = test-yıl-
  ağırlıklı OOF MSE (public proxy). Aynı ağırlıklar test tahminine uygulanır. clip(0,100).

ÇIKTILAR (çalıştırılınca):
- experiments/oof_cat.npy, oof_lgb.npy, oof_xgb.npy, oof_blend.npy
- experiments/faz3_ensemble_log.json
- submissions/sub_<tarih>_blend_oof<skor>.csv
"""
import os, json, datetime
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta

EMB_TRAIN_NPY = 'emb_e5_train.npy'   # experiments/ altında aranır (Kaggle'dan indirilen cache)
EMB_TEST_NPY = 'emb_e5_test.npy'

LGBM_PARAMS = dict(objective='regression', metric='l2', n_estimators=4000, learning_rate=0.03,
                   num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   reg_lambda=3.0, random_state=SEED, n_jobs=-1, verbosity=-1)
XGB_PARAMS = dict(objective='reg:squarederror', eval_metric='rmse', n_estimators=4000,
                  learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=3.0, tree_method='hist', enable_categorical=True,
                  random_state=SEED, n_jobs=-1, early_stopping_rounds=200)


def run_lgbm_cv(X, y, X_test, cat_cols, folds):
    oof = np.zeros(len(X)); test = np.zeros(len(X_test)); iters = []
    for tr, va in folds:
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              categorical_feature=cat_cols,
              callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va]); test += m.predict(X_test) / len(folds)
        iters.append(int(m.best_iteration_ or LGBM_PARAMS['n_estimators']))
    return np.clip(oof, 0, 100), np.clip(test, 0, 100), iters


def run_xgb_cv(X, y, X_test, folds):
    oof = np.zeros(len(X)); test = np.zeros(len(X_test)); iters = []
    for tr, va in folds:
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict(X.iloc[va]); test += m.predict(X_test) / len(folds)
        iters.append(int(m.best_iteration or XGB_PARAMS['n_estimators']))
    return np.clip(oof, 0, 100), np.clip(test, 0, 100), iters


def tune_blend(oofs, y, w, step=0.05):
    """3-simplex üzerinde ağırlık taraması; hedef = test-yıl-ağırlıklı OOF MSE (public proxy)."""
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    best = None
    for a in grid:
        for b in grid:
            c = round(1 - a - b, 4)
            if c < -1e-9 or c > 1 + 1e-9:
                continue
            blend = np.clip(a * oofs[0] + b * oofs[1] + c * oofs[2], 0, 100)
            wm = float(np.sum(w * (y - blend) ** 2) / np.sum(w))
            if best is None or wm < best[0]:
                best = (wm, (float(a), float(b), float(c)))
    return best  # (weighted_mse, (w_cat, w_lgb, w_xgb))


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # --- Ortak meta-feature'lar (Faz 2.1 ile aynı) ---
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    meta_cols = ['text_meta']
    meta_tr = {'text_meta': tm_tr}; meta_te = {'text_meta': tm_te}

    # --- emb_meta: KEEP path (dosya varsa) ---
    emb_tr_p = os.path.join(exp_dir, EMB_TRAIN_NPY); emb_te_p = os.path.join(exp_dir, EMB_TEST_NPY)
    use_emb = os.path.exists(emb_tr_p) and os.path.exists(emb_te_p)
    if use_emb:
        em_tr, em_te = build_emb_meta(np.load(emb_tr_p), np.load(emb_te_p), y, folds)
        meta_cols.append('emb_meta'); meta_tr['emb_meta'] = em_tr; meta_te['emb_meta'] = em_te
    print(f"[feat] {len(num_cols)} sayısal + {len(cat_cols)} kategorik + meta={meta_cols} "
          f"(emb {'VAR (KEEP)' if use_emb else 'YOK (DROP)'})")

    # --- Model-bazlı X (kategorik işleme farklı) ---
    base_cols = num_cols + cat_cols
    # CatBoost: kategorik str
    Xc = train[base_cols].copy(); Xc_t = test[base_cols].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    # LGBM/XGB: kategorik 'category' dtype (test'i train kategorileriyle hizala)
    Xg = train[base_cols].copy(); Xg_t = test[base_cols].copy()
    for c in cat_cols:
        cats = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cats); Xg_t[c] = Xg_t[c].astype(cats)
    for mc in meta_cols:
        Xc[mc] = meta_tr[mc]; Xc_t[mc] = meta_te[mc]
        Xg[mc] = meta_tr[mc]; Xg_t[mc] = meta_te[mc]
    feats_c = base_cols + meta_cols
    cat_idx = [feats_c.index(c) for c in cat_cols]

    # --- 3 model CV (aynı fold) ---
    oof_cat, test_cat, _, _ = run_catboost_cv(Xc, y, Xc_t, cat_idx, folds)
    oof_lgb, test_lgb, it_lgb = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    oof_xgb, test_xgb, it_xgb = run_xgb_cv(Xg, y, Xg_t, folds)

    models = {'cat': oof_cat, 'lgb': oof_lgb, 'xgb': oof_xgb}
    tests = {'cat': test_cat, 'lgb': test_lgb, 'xgb': test_xgb}
    print("\n--- Tekil model OOF (düz | ağırlıklı) ---")
    indiv = {}
    for k, o in models.items():
        p, wt, by = metrics(y, o, w, train[YEAR], te_prop)
        indiv[k] = {'plain': p, 'weighted': wt, 'by_year': by.round(4).to_dict()}
        print(f"  {k}: {p:.4f} | {wt:.4f}")

    # --- Blend (OOF ağırlık taraması) ---
    wm, (wc, wl, wx) = tune_blend([oof_cat, oof_lgb, oof_xgb], y, w)
    oof_blend = np.clip(wc * oof_cat + wl * oof_lgb + wx * oof_xgb, 0, 100)
    test_blend = np.clip(wc * test_cat + wl * test_lgb + wx * test_xgb, 0, 100)
    b_plain, b_weighted, b_by_year = metrics(y, oof_blend, w, train[YEAR], te_prop)

    ref = 87.8292  # Faz 2.1 ağırlıklı OOF
    print("\n================ FAZ 3 SONUÇ (ağırlıklı OOF) ================")
    print(f"Blend ağırlıkları: cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f}")
    print(f"Blend ağırlıklı OOF : {b_weighted:.4f}  (düz {b_plain:.4f})")
    print(f"  kazanç vs Faz 2.1 : {ref - b_weighted:+.4f}  (köprü -> proj public ~{b_weighted - 1.17:.2f})")
    print("\nYıl-bazlı blend OOF MSE (2025-2026 ağır test payı):")
    print(b_by_year.round(3).to_string())

    # --- Kayıtlar ---
    for k, o in models.items():
        np.save(os.path.join(exp_dir, f'oof_{k}.npy'), o)
    np.save(os.path.join(exp_dir, 'oof_blend.npy'), oof_blend)
    log = {
        'phase': 'faz3_ensemble_blend', 'date': datetime.date.today().isoformat(),
        'use_emb': use_emb, 'meta_cols': meta_cols,
        'individual': indiv, 'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
        'blend': {'plain': b_plain, 'weighted': b_weighted, 'by_year': b_by_year.round(4).to_dict()},
        'ref_faz2_1_weighted': ref, 'gain_vs_faz2_1': ref - b_weighted,
        'proj_public': b_weighted - 1.17, 'lgb_iters': it_lgb, 'xgb_iters': it_xgb,
    }
    with open(os.path.join(exp_dir, 'faz3_ensemble_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_blend})
    fname = f"sub_{datetime.date.today().isoformat()}_blend_oof{b_plain:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_cat/lgb/xgb/blend.npy + faz3_ensemble_log.json + {fname}")


if __name__ == '__main__':
    main()
