"""
Datathon 2026 — Faz 16: ÖĞRENİLMİŞ STACKER (HAMLE 2) + TabPFN auto-detect (HAMLE 1 girdisi).

NEDEN: 22. sıra (84.057), ilk 10 eşiği ~83.2-83.4. Basit sabit-ağırlık blend yerine TÜM
bağımsız OOF sinyallerini (cat/lgb/xgb featB + lineer Ridge + text_meta + emb_meta +
berturk_meta [+ TabPFN varsa]) ikinci-seviye bir meta-model (Ridge VE küçük-LightGBM) ile
birleştir.

OOF-ŞİŞME KORUMASI (faz9 dersi, KRİTİK):
- Taban OOF'lar zaten fold-dışı (her zamanki gibi). Ama meta-modeli DOĞRUDAN bu OOF'lar
  üzerinde fit edip yine onlarla değerlendirmek iyimser olur (meta-model kendi eğitim
  verisini görür). Bu yüzden NESTED CV: her outer fold'da meta-model SADECE diğer fold'ların
  OOF'larıyla eğitilir, mevcut fold'un OOF'u yalnız tahmin için kullanılır -> meta_oof tam
  fold-dışı. Ridge alpha'sı da bu nested_oof üzerinden seçilir (küçük grid, hafif iyimserlik).
- Final test tahmini: meta-model TÜM OOF üzerinde fit edilip taban-modellerin TEST tahminlerine
  uygulanır (standart stacking).

TabPFN auto-detect: oof_tabpfn_train.npy + tabpfn_test.npy varsa (Kaggle'da üretilip
experiments/'e konursa) otomatik 8. kolon olarak eklenir; iki versiyon (TabPFN'li/'siz)
karşılaştırılır -> wOOF farkı şişme mi gerçek mi raporlanır (TabPFN OOF <-> GBM-blend OOF
korelasyonu ile).

Çıktı: experiments/faz16_stacker_log.json + (kazanan varyant) submissions/sub_<tarih>_stacker_v2_woof<>.csv
"""
import os, json, datetime, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz14_seedbag import build_featB_X
from faz5_linear import run_linear_cv

FIXED_BLEND = (0.7, 0.2, 0.1)   # featB kanıtlı sabit blend (faz13 ile aynı, referans)
RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def nested_ridge_stack(M_oof, M_test, y, w, folds, alphas):
    """Her alpha için nested OOF (her fold meta-model diğer fold'ların OOF'uyla fit), en iyi
    alpha wOOF ile seçilir; final test = seçilen alpha ile TÜM OOF'a fit edilen Ridge."""
    best = None
    for a in alphas:
        nested_oof = np.zeros(len(y))
        for tr, va in folds:
            r = Ridge(alpha=a, random_state=42).fit(M_oof[tr], y[tr])
            nested_oof[va] = r.predict(M_oof[va])
        nested_oof = np.clip(nested_oof, 0, 100)
        _, wt, _ = metrics(y, nested_oof, w, None, None) if False else (None, float(np.sum(w*(y-nested_oof)**2)/np.sum(w)), None)
        if best is None or wt < best[0]:
            best = (wt, a, nested_oof)
    wt, a, nested_oof = best
    final = Ridge(alpha=a, random_state=42).fit(M_oof, y)
    test_pred = np.clip(final.predict(M_test), 0, 100)
    return nested_oof, test_pred, wt, a


def nested_lgb_stack(M_oof, M_test, y, w, folds):
    """Küçük LightGBM meta-model, nested CV (her fold diğer fold'ların OOF'uyla fit)."""
    params = dict(objective='regression', metric='l2', n_estimators=200, learning_rate=0.05,
                   max_depth=3, num_leaves=7, min_child_samples=50, subsample=0.8,
                   colsample_bytree=0.8, reg_lambda=3.0, random_state=42, n_jobs=-1, verbosity=-1)
    nested_oof = np.zeros(len(y)); test_pred = np.zeros(M_test.shape[0])
    for tr, va in folds:
        m = LGBMRegressor(**params)
        m.fit(M_oof[tr], y[tr])
        nested_oof[va] = m.predict(M_oof[va])
        test_pred += m.predict(M_test) / len(folds)
    nested_oof = np.clip(nested_oof, 0, 100); test_pred = np.clip(test_pred, 0, 100)
    wt = float(np.sum(w*(y-nested_oof)**2)/np.sum(w))
    return nested_oof, test_pred, wt


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    emb_tr_raw = np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)); emb_te_raw = np.load(os.path.join(exp_dir, EMB_TEST_NPY))
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    txt_tr = train[TEXT].fillna(''); txt_te = test[TEXT].fillna('')

    # ---------- 1) featB taban modeller (cat/lgb/xgb, sade paramlar, seed=42) ----------
    print("[faz16] adım 1: featB taban modeller (cat/lgb/xgb, sade paramlar)")
    t0 = time.time()
    Xc, Xc_t, Xg, Xg_t, ci, feats = build_featB_X(train_fe, test_fe, txt_tr, txt_te, y, num_cols, cat_cols,
                                                   emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds)
    oof_cat, test_cat, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    oof_lgb, test_lgb, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    oof_xgb, test_xgb, _ = run_xgb_cv(Xg, y, Xg_t, folds)
    print(f"  GBM tabanlar {time.time()-t0:.0f}s")

    # text_meta / emb_meta / berturk_meta -> Xc kolonlarından geri al (build_featB_X içinde üretildi)
    tm_tr, tm_te = Xc['text_meta'].values, Xc_t['text_meta'].values
    em_tr, em_te = Xc['emb_meta'].values, Xc_t['emb_meta'].values

    # ---------- 2) lineer Ridge (faz5 ile aynı pipeline, featB meta seti) ----------
    print("[faz16] adım 2: lineer Ridge (4. üye)")
    t0 = time.time()
    lin_num = num_cols + ['text_meta', 'emb_meta', 'berturk_meta']
    Xl = train_fe[num_cols + cat_cols].copy(); Xl_t = test_fe[num_cols + cat_cols].copy()
    Xl['text_meta'] = tm_tr; Xl['emb_meta'] = em_tr; Xl['berturk_meta'] = bt_tr
    Xl_t['text_meta'] = tm_te; Xl_t['emb_meta'] = em_te; Xl_t['berturk_meta'] = bt_te
    lw, lalpha, oof_lin, test_lin = run_linear_cv(Xl, Xl_t, y, lin_num, cat_cols, folds, w)
    print(f"  lineer wOOF={lw:.4f} (alpha={lalpha}) {time.time()-t0:.0f}s")

    # ---------- 3) TabPFN auto-detect ----------
    tabpfn_op = os.path.join(exp_dir, 'oof_tabpfn_train.npy'); tabpfn_tp = os.path.join(exp_dir, 'tabpfn_test.npy')
    has_tabpfn = os.path.exists(tabpfn_op) and os.path.exists(tabpfn_tp)
    tabpfn_corr = None
    if has_tabpfn:
        oof_tab = np.clip(np.load(tabpfn_op).astype('float64'), 0, 100)
        test_tab = np.clip(np.load(tabpfn_tp).astype('float64'), 0, 100)
        gbm_blend_oof = np.clip(FIXED_BLEND[0]*oof_cat + FIXED_BLEND[1]*oof_lgb + FIXED_BLEND[2]*oof_xgb, 0, 100)
        tabpfn_corr = float(np.corrcoef(oof_tab, gbm_blend_oof)[0, 1])
        print(f"[faz16] TabPFN bulundu. corr(TabPFN-OOF, GBM-blend-OOF) = {tabpfn_corr:.4f} "
              f"({'çeşitlilik VAR -> ensemble katkısı muhtemel' if tabpfn_corr < 0.90 else 'YÜKSEK korelasyon -> katkı sınırlı olabilir'})")
    else:
        print("[faz16] TabPFN bulunamadı (oof_tabpfn_train.npy / tabpfn_test.npy YOK) -> TabPFN'siz devam")

    # ---------- 4) referans: SABİT blend (faz13/featB ile aynı) ----------
    fx_oof = np.clip(FIXED_BLEND[0]*oof_cat + FIXED_BLEND[1]*oof_lgb + FIXED_BLEND[2]*oof_xgb, 0, 100)
    fx_test = np.clip(FIXED_BLEND[0]*test_cat + FIXED_BLEND[1]*test_lgb + FIXED_BLEND[2]*test_xgb, 0, 100)
    _, fx_w, fx_by = metrics(y, fx_oof, w, train[YEAR], te_prop)
    print(f"\n[referans] SABİT blend {FIXED_BLEND} (3-GBM)              -> wOOF={fx_w:.4f}")

    # ---------- 5) Stacker varyantları ----------
    base_cols = ['cat', 'lgb', 'xgb', 'lin', 'text_meta', 'emb_meta', 'berturk_meta']
    M_oof_list = [oof_cat, oof_lgb, oof_xgb, oof_lin, tm_tr, em_tr, bt_tr]
    M_test_list = [test_cat, test_lgb, test_xgb, test_lin, tm_te, em_te, bt_te]
    if has_tabpfn:
        base_cols_full = base_cols + ['tabpfn']
        M_oof_full = np.column_stack(M_oof_list + [oof_tab]); M_test_full = np.column_stack(M_test_list + [test_tab])
    M_oof = np.column_stack(M_oof_list); M_test = np.column_stack(M_test_list)

    results = {}
    for tag, Mo, Mt, cols in [('no_tabpfn', M_oof, M_test, base_cols)] + (
            [('with_tabpfn', M_oof_full, M_test_full, base_cols_full)] if has_tabpfn else []):
        ro, rt, rw, ra = nested_ridge_stack(Mo, Mt, y, w, folds, RIDGE_ALPHAS)
        lo, lt, lwt = nested_lgb_stack(Mo, Mt, y, w, folds)
        print(f"\n[{tag}] Ridge-stacker  (alpha={ra}) -> wOOF={rw:.4f}  (Δ vs sabit blend {fx_w-rw:+.4f})")
        print(f"[{tag}] LightGBM-stacker            -> wOOF={lwt:.4f}  (Δ vs sabit blend {fx_w-lwt:+.4f})")
        results[tag] = {'cols': cols, 'ridge': {'alpha': ra, 'oof': ro, 'test': rt, 'weighted': rw},
                        'lgb': {'oof': lo, 'test': lt, 'weighted': lwt}}

    # ---------- 6) kazananı seç ----------
    candidates = [('fixed_blend_3gbm', fx_oof, fx_test, fx_w)]
    for tag, r in results.items():
        candidates.append((f'{tag}_ridge', r['ridge']['oof'], r['ridge']['test'], r['ridge']['weighted']))
        candidates.append((f'{tag}_lgb', r['lgb']['oof'], r['lgb']['test'], r['lgb']['weighted']))
    winner = min(candidates, key=lambda c: c[3])
    w_oof, w_test, w_wt = winner[1], winner[2], winner[3]
    _, _, w_by = metrics(y, w_oof, w, train[YEAR], te_prop)
    print(f"\n================ FAZ 16 SONUÇ ================")
    print("Aday wOOF (REFERANS — wOOF güvenilmezliği faz9 dersi geçerli, KARAR public ile):")
    for name, _, _, wt in candidates:
        flag = "  <- en düşük wOOF (ref)" if name == winner[0] else ""
        print(f"  {name:24s}: {wt:.4f}{flag}")
    print(f"\nKazanan (referans): {winner[0]} wOOF={w_wt:.4f}")
    print("\nYıl-bazlı (kazanan):"); print(w_by.round(3).to_string())

    # ---------- kayıt ----------
    np.save(os.path.join(exp_dir, 'oof_stacker_v2.npy'), w_oof)
    np.save(os.path.join(exp_dir, 'test_stacker_v2.npy'), w_test)
    log = {'phase': 'faz16_stacker', 'date': datetime.date.today().isoformat(),
           'has_tabpfn': has_tabpfn, 'tabpfn_corr_vs_gbm_blend': tabpfn_corr,
           'fixed_blend_3gbm': {'weighted': fx_w, 'by_year': fx_by.round(4).to_dict()},
           'linear': {'weighted': lw, 'alpha': lalpha},
           'variants': {tag: {'cols': r['cols'], 'ridge': {'alpha': r['ridge']['alpha'], 'weighted': r['ridge']['weighted']},
                              'lgb': {'weighted': r['lgb']['weighted']}} for tag, r in results.items()},
           'winner': winner[0], 'winner_weighted': w_wt, 'winner_by_year': w_by.round(4).to_dict(),
           'note': 'wOOF REFERANS (faz9 dersi); karar PUBLIC ile.'}
    with open(os.path.join(exp_dir, 'faz16_stacker_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = f"sub_{datetime.date.today().isoformat()}_stacker_v2_{winner[0]}_woof{w_wt:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: w_test}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof/test_stacker_v2.npy + faz16_stacker_log.json + {fname}")


if __name__ == '__main__':
    main()
