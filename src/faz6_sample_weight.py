"""
Datathon 2026 — Faz 6: test-yıl sample-weight ile EĞİTİM (geç-yıl odaklı optimizasyon).

NEDEN: Yarışma geç-yıl (2025-2026) tahmininde kazanılıyor; erken-yıl herkese kolay. Şimdiye kadar
test-yıl ağırlığını yalnız DEĞERLENDİRMEDE kullandık. Burada GBM'leri o ağırlıkla EĞİTİYORUZ
(late-year up-weight) -> optimizasyon tam metriğin önemsediği bölgeye odaklanır. Early stopping de
ağırlıklı validation'da. Aynı fold + FE + text_meta + emb_meta. Sabit ağırlık (OOF-tune yok).

Karşılaştırma: Faz 4 ağırlıksız-eğitim 3-yönlü blend wOOF 86.27. Sample-weight bunu düşürür mü?
Özellikle 2025-2026 yıl-bazlı wOOF kritik.
"""
import os, json, datetime
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, metrics, CATBOOST_PARAMS
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import LGBM_PARAMS, XGB_PARAMS, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer


def cat_w(X, y, Xt, cat_idx, folds, sw):
    oof = np.zeros(len(X)); test = np.zeros(len(Xt))
    for tr, va in folds:
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx, weight=sw[tr]),
              eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx, weight=sw[va]), use_best_model=True)
        oof[va] = m.predict(X.iloc[va]); test += m.predict(Xt) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def lgb_w(X, y, Xt, cat_cols, folds, sw):
    oof = np.zeros(len(X)); test = np.zeros(len(Xt))
    for tr, va in folds:
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(X.iloc[tr], y[tr], sample_weight=sw[tr], eval_set=[(X.iloc[va], y[va])],
              eval_sample_weight=[sw[va]], categorical_feature=cat_cols,
              callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(X.iloc[va]); test += m.predict(Xt) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def xgb_w(X, y, Xt, folds, sw):
    oof = np.zeros(len(X)); test = np.zeros(len(Xt))
    for tr, va in folds:
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(X.iloc[tr], y[tr], sample_weight=sw[tr],
              eval_set=[(X.iloc[va], y[va])], sample_weight_eval_set=[sw[va]], verbose=False)
        oof[va] = m.predict(X.iloc[va]); test += m.predict(Xt) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]; num_cols = raw_num + eng_cols
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)
    # eğitim ağırlığı: ort. 1'e normalize (öğrenme oranı/regülarizasyon ölçeği bozulmasın)
    sw = w / w.mean()

    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    meta = ['text_meta', 'emb_meta']
    train_fe['text_meta'] = tm_tr; train_fe['emb_meta'] = em_tr
    test_fe['text_meta'] = tm_te; test_fe['emb_meta'] = em_te

    base = num_cols + cat_cols
    Xc = train_fe[base + meta].copy(); Xc_t = test_fe[base + meta].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[base + meta].copy(); Xg_t = test_fe[base + meta].copy()
    for c in cat_cols:
        cats = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cats); Xg_t[c] = Xg_t[c].astype(cats)
    feats_c = base + meta; cat_idx = [feats_c.index(c) for c in cat_cols]

    res = {}
    for name, fn in [('cat', lambda: cat_w(Xc, y, Xc_t, cat_idx, folds, sw)),
                     ('lgb', lambda: lgb_w(Xg, y, Xg_t, cat_cols, folds, sw)),
                     ('xgb', lambda: xgb_w(Xg, y, Xg_t, folds, sw))]:
        oof, tst = fn(); p, wt, by = metrics(y, oof, w, train[YEAR], te_prop)
        res[name] = (oof, tst, p, wt, by)
        print(f"[sw-eğitim] {name}: düz={p:.4f} | ağırlıklı={wt:.4f}")

    ref = {'cat': 86.5748, 'lgb': 88.1518, 'xgb': 88.9607}  # Faz 4/5 ağırlıksız-eğitim referansı
    print("\n--- Ağırlıksız-eğitim referansıyla kıyas (ağırlıklı OOF) ---")
    for k in ['cat', 'lgb', 'xgb']:
        print(f"  {k}: {ref[k]:.4f} -> {res[k][3]:.4f}  ({ref[k]-res[k][3]:+.4f})")

    wm, (wc, wl, wx) = tune_blend([res['cat'][0], res['lgb'][0], res['xgb'][0]], y, w)
    oof_b = np.clip(wc*res['cat'][0] + wl*res['lgb'][0] + wx*res['xgb'][0], 0, 100)
    test_b = np.clip(wc*res['cat'][1] + wl*res['lgb'][1] + wx*res['xgb'][1], 0, 100)
    bp, bw, bby = metrics(y, oof_b, w, train[YEAR], te_prop)

    print("\n================ FAZ 6 (sample-weight eğitim) — ağırlıklı OOF ================")
    print(f"Faz 4 ağırlıksız-eğitim blend : 86.2748")
    print(f"Faz 6 sample-weight blend     : {bw:.4f}  (düz {bp:.4f})  | kazanç {86.2748-bw:+.4f}")
    print(f"  blend ağırlıkları           : cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f}")
    print("\nYıl-bazlı (Faz 4 ağırlıksız vs Faz 6 sw):")
    f4 = {2019:54.191,2020:57.075,2021:62.975,2022:75.406,2023:79.725,2024:76.380,2025:104.972,2026:106.267}
    for yr in sorted(bby.index):
        d = f4.get(int(yr), np.nan) - bby[yr]
        flag = "  <<< savaş alanı" if yr in (2025, 2026) else ""
        print(f"  {yr}: {f4.get(int(yr), float('nan')):.2f} -> {bby[yr]:.3f}  ({d:+.3f}){flag}")

    np.save(os.path.join(exp_dir, 'oof_blend_sw.npy'), oof_b)
    log = {'phase': 'faz6_sample_weight_training', 'date': datetime.date.today().isoformat(),
           'individual_weighted_oof': {k: res[k][3] for k in res},
           'ref_unweighted': ref, 'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
           'blend': {'plain': bp, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'ref_faz4_blend': 86.2748, 'gain_vs_faz4': 86.2748 - bw}
    with open(os.path.join(exp_dir, 'faz6_sw_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_b})
    fname = f"sub_{datetime.date.today().isoformat()}_blend_sw_oof{bp:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_blend_sw.npy + faz6_sw_log.json + {fname}")


if __name__ == '__main__':
    main()
