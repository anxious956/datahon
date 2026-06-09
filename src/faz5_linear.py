"""
Datathon 2026 — Faz 5: Güçlü lineer model (Ridge) -> 4. ensemble üyesi.

NEDEN (DS-analizi bulgusu): hedef büyük ölçüde LİNEER kurulmuş (sadece sayısal Ridge R²≈0.573;
sentetik formül muhtemelen ağırlıklı-toplam+gürültü+clip). Sonuç:
- GBM'in nonlineer gücü kısmen israf; HİÇ denemediğimiz koz = güçlü lineer model.
- Lineer vs ağaç = GERÇEK model çeşitliliği (stacking-şişmesi DEĞİL, e5 tuzağı gibi değil).
- Bu yüzden 4-yönlü blend ağırlığı wOOF'ta tune edilebilir; diversity kazancı public'te de beklenir
  (yine de submission ile doğrulanacak).

Lineer pipeline (sızıntısız, fold-içi fit):
- sayısal+FE+meta(text/emb): median impute + StandardScaler
- kategorik: OneHotEncoder(handle_unknown='ignore')
- Ridge; alpha küçük taramayla (lineer modelin kendi hiperparametresi — meşru CV seçimi)

Karşılaştırma: Faz 4 (3-yönlü GBM blend, FE'li) wOOF 86.27. Lineer 4. üye ne kadar düşürür?
"""
import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

ALPHAS = [1.0, 5.0, 10.0, 30.0, 100.0]


def run_linear_cv(Xdf, Xdf_t, y, num_feats, cat_feats, folds, w):
    """Ridge OOF (leakage-free, fold-içi preprocess). alpha'yı ağırlıklı OOF ile seç."""
    best = None
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_feats),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
    ])
    for a in ALPHAS:
        oof = np.zeros(len(Xdf)); test = np.zeros(len(Xdf_t))
        for tr, va in folds:
            P = ColumnTransformer(pre.transformers, remainder='drop')
            Xtr = P.fit_transform(Xdf.iloc[tr]); Xva = P.transform(Xdf.iloc[va]); Xte = P.transform(Xdf_t)
            r = Ridge(alpha=a, random_state=SEED).fit(Xtr, y[tr])
            oof[va] = r.predict(Xva); test += r.predict(Xte) / len(folds)
        oof = np.clip(oof, 0, 100); test = np.clip(test, 0, 100)
        wt = float(np.sum(w * (y - oof) ** 2) / np.sum(w))
        if best is None or wt < best[0]:
            best = (wt, a, oof, test)
    return best  # (weighted, alpha, oof, test)


def tune_blend4(oofs, y, w, step=0.05):
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4); best = None
    for a in grid:
        for b in grid:
            if a + b > 1 + 1e-9:
                continue
            for c in grid:
                d = round(1 - a - b - c, 4)
                if d < -1e-9 or d > 1 + 1e-9:
                    continue
                bl = np.clip(a*oofs[0] + b*oofs[1] + c*oofs[2] + d*oofs[3], 0, 100)
                wm = float(np.sum(w*(y-bl)**2)/np.sum(w))
                if best is None or wm < best[0]:
                    best = (wm, (float(a), float(b), float(c), float(d)))
    return best


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

    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    meta_cols = ['text_meta', 'emb_meta']
    for df, a, b in [(train_fe, tm_tr, em_tr)]:
        df['text_meta'] = a; df['emb_meta'] = b
    for df, a, b in [(test_fe, tm_te, em_te)]:
        df['text_meta'] = a; df['emb_meta'] = b

    base_cols = num_cols + cat_cols
    # GBM girdileri
    Xc = train_fe[base_cols + meta_cols].copy(); Xc_t = test_fe[base_cols + meta_cols].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[base_cols + meta_cols].copy(); Xg_t = test_fe[base_cols + meta_cols].copy()
    for c in cat_cols:
        cats = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cats); Xg_t[c] = Xg_t[c].astype(cats)
    feats_c = base_cols + meta_cols; cat_idx = [feats_c.index(c) for c in cat_cols]

    oof_cat, test_cat, _, _ = run_catboost_cv(Xc, y, Xc_t, cat_idx, folds)
    oof_lgb, test_lgb, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    oof_xgb, test_xgb, _ = run_xgb_cv(Xg, y, Xg_t, folds)

    # Lineer (4. üye) — sayısal+FE+meta scale, kategorik OHE
    lin_num = num_cols + meta_cols
    Xl = train_fe[lin_num + cat_cols].copy(); Xl_t = test_fe[lin_num + cat_cols].copy()
    lw, lalpha, oof_lin, test_lin = run_linear_cv(Xl, Xl_t, y, lin_num, cat_cols, folds, w)

    print("\n--- Tekil model OOF (düz | ağırlıklı) ---")
    indiv = {}
    for k, o in [('cat', oof_cat), ('lgb', oof_lgb), ('xgb', oof_xgb), ('lin', oof_lin)]:
        p, wt, by = metrics(y, o, w, train[YEAR], te_prop)
        indiv[k] = {'plain': p, 'weighted': wt, 'by_year': by.round(4).to_dict()}
        print(f"  {k}: {p:.4f} | {wt:.4f}" + (f"  (Ridge alpha={lalpha})" if k == 'lin' else ""))

    # 4-yönlü blend
    wm, (wc, wl, wx, wn) = tune_blend4([oof_cat, oof_lgb, oof_xgb, oof_lin], y, w)
    oof_b = np.clip(wc*oof_cat + wl*oof_lgb + wx*oof_xgb + wn*oof_lin, 0, 100)
    test_b = np.clip(wc*test_cat + wl*test_lgb + wx*test_xgb + wn*test_lin, 0, 100)
    bp, bw, bby = metrics(y, oof_b, w, train[YEAR], te_prop)

    print("\n================ FAZ 5 (lineer 4. üye) — ağırlıklı OOF ================")
    print(f"3-yönlü GBM blend (Faz 4)  : 86.2748")
    print(f"4-yönlü +lineer blend      : {bw:.4f}  (düz {bp:.4f})")
    print(f"  kazanç vs Faz 4          : {86.2748 - bw:+.4f}")
    print(f"  blend ağırlıkları        : cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f} lin={wn:.2f}")
    print("\nYıl-bazlı 4-yönlü blend OOF (2025-2026 kritik):")
    print(bby.round(3).to_string())

    np.save(os.path.join(exp_dir, 'oof_lin.npy'), oof_lin)
    np.save(os.path.join(exp_dir, 'oof_blend4.npy'), oof_b)
    log = {'phase': 'faz5_linear_4way', 'date': datetime.date.today().isoformat(),
           'ridge_alpha': lalpha, 'individual': indiv,
           'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx, 'lin': wn},
           'blend4': {'plain': bp, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'ref_faz4_3way': 86.2748, 'gain_vs_faz4': 86.2748 - bw}
    with open(os.path.join(exp_dir, 'faz5_linear_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_b})
    fname = f"sub_{datetime.date.today().isoformat()}_blend4_oof{bp:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_lin.npy + oof_blend4.npy + faz5_linear_log.json + {fname}")


if __name__ == '__main__':
    main()
