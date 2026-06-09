"""
Datathon 2026 — Faz 7: BERTurk OOF entegrasyonu (2 yöntem, OOF-tune YOK).

A) SABİT-AĞIRLIK BLEND (kullanıcı talebi): final = (1-α)*ensemble + α*berturk, α SABİT (OOF-tune yok).
   α-sweep yalnız MANZARA için gösterilir; karar sabit/konservatif + public doğrulama.
B) FEATURE olarak: berturk_meta'yı GBM feature setine kat (text_meta/emb_meta gibi). Text'in
   sayısalla birleştiği doğal yol; muhtemelen A'dan iyi ama OOF şişebilir (e5 dersi) -> public'te doğrula.

Referans: Faz 5 ensemble (FE+text+emb, 4-yönlü) wOOF 86.24. BERTurk-only wOOF 146.67.
"""
import os, json, datetime
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer


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

    # BERTurk OOF (train) + test
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    bp, bw, bby = metrics(y, bt_tr, w, train[YEAR], te_prop)
    print(f"[BERTurk-only] düz={bp:.4f} ağırlıklı={bw:.4f}")

    # Meta-feature'lar
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)

    # ---------- Mevcut ensemble (Faz 5, berturk'süz) — A için baz ----------
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

    def run3(meta_map):
        Xc, Xc_t, Xg, Xg_t, feats = build_X(meta_map)
        ci = [feats.index(c) for c in cat_cols]
        oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
        ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
        ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
        wm, (a, b, c) = tune_blend([oc, ol, ox], y, w)
        oof = np.clip(a*oc + b*ol + c*ox, 0, 100); tst = np.clip(a*tc + b*tl + c*tx, 0, 100)
        return oof, tst, (a, b, c)

    base_meta = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te)}
    ens_oof, ens_te, ew = run3(base_meta)
    e_p, e_w, e_by = metrics(y, ens_oof, w, train[YEAR], te_prop)
    print(f"[ensemble (berturk'süz)] düz={e_p:.4f} ağırlıklı={e_w:.4f} (Faz 5 ~86.24)")

    # ---------- A) SABİT-AĞIRLIK BLEND ----------
    print("\n================ A) SABİT-AĞIRLIK BLEND (OOF-tune YOK) ================")
    print("α-sweep (yalnız manzara — karar sabit α + public):")
    for a in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        bl = np.clip((1-a)*ens_oof + a*bt_tr, 0, 100)
        _, wa, _ = metrics(y, bl, w, train[YEAR], te_prop)
        print(f"  α={a:.2f}: wOOF={wa:.4f}")
    ALPHA = 0.10  # SABİT konservatif (OOF argmin DEĞİL)
    blA = np.clip((1-ALPHA)*ens_oof + ALPHA*bt_tr, 0, 100)
    blA_te = np.clip((1-ALPHA)*ens_te + ALPHA*bt_te, 0, 100)
    Ap, Aw, Aby = metrics(y, blA, w, train[YEAR], te_prop)
    print(f"SEÇİLEN sabit α={ALPHA}: wOOF={Aw:.4f} (düz {Ap:.4f}) | vs ensemble {e_w-Aw:+.4f}")
    print("  yıl-bazlı:"); print(Aby.round(3).to_string())

    # ---------- B) FEATURE olarak ----------
    print("\n================ B) FEATURE (berturk_meta GBM'e) ================")
    meta_b = dict(base_meta); meta_b['berturk_meta'] = (bt_tr, bt_te)
    feat_oof, feat_te, fw = run3(meta_b)
    Bp, Bw, Bby = metrics(y, feat_oof, w, train[YEAR], te_prop)
    print(f"wOOF={Bw:.4f} (düz {Bp:.4f}) | vs ensemble {e_w-Bw:+.4f}  | blend ağırlık cat/lgb/xgb={fw}")
    print("  yıl-bazlı:"); print(Bby.round(3).to_string())

    # ---------- Kayıt: iki aday submission ----------
    for tag, te_pred, woof in [('blendA', blA_te, Aw), ('featB', feat_te, Bw)]:
        pd.DataFrame({ID: test[ID].values, TARGET: te_pred}).to_csv(
            os.path.join(sub_dir, f"sub_{datetime.date.today().isoformat()}_berturk_{tag}_woof{woof:.2f}.csv"), index=False)
    log = {'phase': 'faz7_berturk_integrate', 'date': datetime.date.today().isoformat(),
           'berturk_only': {'plain': bp, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'ensemble_no_berturk': {'plain': e_p, 'weighted': e_w},
           'A_fixed_blend': {'alpha': ALPHA, 'plain': Ap, 'weighted': Aw, 'by_year': Aby.round(4).to_dict()},
           'B_feature': {'plain': Bp, 'weighted': Bw, 'by_year': Bby.round(4).to_dict(), 'blend_w': fw}}
    with open(os.path.join(exp_dir, 'faz7_berturk_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\n[kayıt] 2 aday submission (blendA, featB) + faz7_berturk_log.json")
    print(f"\nÖZET: ensemble {e_w:.3f} | A(sabit α={ALPHA}) {Aw:.3f} | B(feature) {Bw:.3f}")
    print(f"  köprü -> proj public: A ~{Aw-1.17:.2f} | B ~{Bw-1.17:.2f}  (B feature ise emb-gibi şişebilir, public şart)")


if __name__ == '__main__':
    main()
