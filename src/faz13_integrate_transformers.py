"""
Datathon 2026 — Faz 13: TRANSFORMER ÇEŞİTLİLİĞİ entegrasyonu (anti-overfit, KONSERVATİF).

STRATEJİK GEREKÇE (public kanıtı):
- Public'i hareket ettiren TEK şey yapısal metin sinyali: berturk feature 86.32->84.10 (+2.22).
- GBM tuning metrik-overfit oldu (wOOF -0.60 ama public +0.09 KÖTÜ). Tabular doygun.
- Bu yüzden: yeni transformer OOF'larını (electra/xlmr/bert128k) featB'ye FEATURE olarak kat
  (berturk gibi — feature > blend kanıtlandı), AMA:
    * DEFAULT GBM paramları (tuned DEĞİL — tuned overfit etti)
    * SABİT konservatif blend ağırlığı [0.7/0.2/0.1] (wOOF-argmin DEĞİL — o da overfit vektörü)
  wOOF yalnız REFERANS olarak yazılır; KARAR public ile verilir.

Auto-detect: oof_<tag>_train.npy + <tag>_test.npy varsa <tag>_meta olarak eklenir.
Transformer yoksa -> featB'yi aynen üretir (sanity: wOOF ~86.09, public ~84.10 olmalı).

Çeşitlilik raporu: her transformer'ın berturk + GBM-blend OOF ile korelasyonu (düşük=gerçek çeşitlilik).
"""
import os, json, datetime
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

# Aday transformer tag'leri (Kaggle multimodel notebook çıktıları). Var olanlar otomatik girer.
TRANSFORMER_TAGS = ['bert128k', 'electra', 'xlmr']
FIXED_BLEND = (0.7, 0.2, 0.1)   # featB'nin public'te 84.10 veren KANITLI ağırlığı (wOOF-argmin DEĞİL)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)              # sentiment=False
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # --- featB proven meta seti: text + emb + berturk ---
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    meta = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te), 'berturk_meta': (bt_tr, bt_te)}

    # --- YENİ transformerları auto-detect + çeşitlilik raporu ---
    found = []
    for tag in TRANSFORMER_TAGS:
        op = os.path.join(exp_dir, f'oof_{tag}_train.npy'); tp = os.path.join(exp_dir, f'{tag}_test.npy')
        if os.path.exists(op) and os.path.exists(tp):
            o = np.clip(np.load(op).astype('float64'), 0, 100); t = np.clip(np.load(tp).astype('float64'), 0, 100)
            meta[f'{tag}_meta'] = (o, t); found.append((tag, o))
    print(f"[transformer] bulunan: {[f for f,_ in found] or 'YOK (featB reproduce)'}")
    if found:
        print("  çeşitlilik (OOF korelasyon, düşük=iyi):")
        for tag, o in found:
            cb = np.corrcoef(o, bt_tr)[0, 1]
            print(f"    {tag:9s}: berturk ile {cb:.3f}")

    # --- UNTUNED 3-GBM (default paramlar) ---
    def build_X(meta_map):
        cols = num_cols + cat_cols
        Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
        for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
        Xg = train_fe[cols].copy(); Xg_t = test_fe[cols].copy()
        for c in cat_cols:
            cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
            Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
        for m, (a, b) in meta_map.items():
            Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
        feats = cols + list(meta_map); return Xc, Xc_t, Xg, Xg_t, feats

    Xc, Xc_t, Xg, Xg_t, feats = build_X(meta)
    ci = [feats.index(c) for c in cat_cols]
    oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)

    print("\n--- Tekil model (düz | ağırlıklı) ---")
    for k, o in [('cat', oc), ('lgb', ol), ('xgb', ox)]:
        p, wt, _ = metrics(y, o, w, train[YEAR], te_prop); print(f"  {k}: {p:.4f} | {wt:.4f}")
    if found:
        bloof = np.clip(FIXED_BLEND[0]*oc + FIXED_BLEND[1]*ol + FIXED_BLEND[2]*ox, 0, 100)
        for tag, o in found:
            print(f"  çeşitlilik: {tag} vs GBM-blend OOF korelasyon {np.corrcoef(o, bloof)[0,1]:.3f}")

    # --- SABİT konservatif blend (KARAR) + wOOF-argmin (yalnız referans) ---
    a, b, c = FIXED_BLEND
    oof_fx = np.clip(a*oc + b*ol + c*ox, 0, 100); test_fx = np.clip(a*tc + b*tl + c*tx, 0, 100)
    p_fx, w_fx, by_fx = metrics(y, oof_fx, w, train[YEAR], te_prop)
    _, (wa, wb, wc_) = tune_blend([oc, ol, ox], y, w)
    oof_tn = np.clip(wa*oc + wb*ol + wc_*ox, 0, 100)
    _, w_tn, _ = metrics(y, oof_tn, w, train[YEAR], te_prop)

    print(f"\n================ FAZ 13 SONUÇ ================")
    print(f"SABİT blend {FIXED_BLEND} -> wOOF={w_fx:.4f} (düz {p_fx:.4f})   <- SUBMIT EDİLECEK (anti-overfit)")
    print(f"  [referans] wOOF-argmin blend ({wa:.2f},{wb:.2f},{wc_:.2f}) -> {w_tn:.4f}  (KULLANMA: overfit vektörü)")
    print(f"  featB referans wOOF 86.09 | bu wOOF SADECE referans, KARAR public ile")
    print("\nYıl-bazlı (sabit blend):"); print(by_fx.round(3).to_string())

    np.save(os.path.join(exp_dir, 'oof_faz13_blend.npy'), oof_fx)
    log = {'phase': 'faz13_transformer_integrate', 'date': datetime.date.today().isoformat(),
           'transformers_found': [f for f, _ in found], 'fixed_blend': list(FIXED_BLEND),
           'fixed': {'plain': p_fx, 'weighted': w_fx, 'by_year': by_fx.round(4).to_dict()},
           'argmin_ref': {'weights': [wa, wb, wc_], 'weighted': w_tn},
           'diversity_vs_berturk': {tag: float(np.corrcoef(o, bt_tr)[0, 1]) for tag, o in found},
           'note': 'KARAR public ile; wOOF ince ayrimda guvenilmez (faz9 dersi)'}
    with open(os.path.join(exp_dir, 'faz13_integrate_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    suffix = '_'.join([f for f, _ in found]) or 'featB_repro'
    fname = f"sub_{datetime.date.today().isoformat()}_faz13_{suffix}_woof{w_fx:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: test_fx}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_faz13_blend.npy + faz13_integrate_log.json + {fname}")


if __name__ == '__main__':
    main()
