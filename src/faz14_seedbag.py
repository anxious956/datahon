"""
Datathon 2026 — Faz 14: featB SEED BAGGING (saf varyans düşürme, optimizasyon YOK).

NEDEN (kritik teşhis):
- Faz 9 GBM tuning wOOF'u 86.09 -> 85.49 indirdi AMA public 84.10 -> 84.19'a ÇIKTI (kötüleşti).
  Optuna test-yıl-ağırlıklı OOF'a (wOOF) overfit etti; köprü KIRILDI, wOOF artık public'i izlemiyor.
  -> SONUÇ: wOOF'u optimize ETME. Bu yol kanıtlı şekilde public'e yansımıyor.
- Faz 10 bagging FAZ 9 TUNED paramları kullanıyordu (zaten overfit'li temel). Burada onun yerine
  EN İYİ PUBLIC konfigürasyonu (featB = faz7: FE + 3-GBM + text_meta + emb_meta + berturk_meta,
  ORİJİNAL SADE paramlar) TEMEL alınır. Hiçbir hiperparametre değiştirilmez/tune edilmez.

YÖNTEM (saf bagging):
- Tüm pipeline'ı (FOLD SPLIT DAHİL) N farklı random seed ile koş. Her seed:
  fold split + 3 GBM (cat/lgb/xgb) random_seed + blend -> kendi test tahminini üretir.
- Sabit (yeniden eğitilmeyen) girdiler: e5 ham embedding (emb_e5_*.npy) ve BERTurk OOF/test
  (oof_berturk_train.npy / berturk_test.npy). Bunlar tüm seed'lerde AYNI kullanılır; yalnızca
  üstlerindeki Ridge-meta fold'a bağlı olduğu için seed başına yeniden türetilir (leakage-free).
- N seed'in TEST tahminlerini ORTALA -> clip(0,100) -> final bagged submission.

wOOF RAPORLANIR ama GÜVENİLMEZ (köprü kırık). Asıl bakılan 3 metrik:
  (a) bagged tahmin <-> faz7(featB) tahmini korelasyonu (çok yüksek olmalı ~0.99).
  (b) seed'ler arası test-tahmini std'sinin ortalaması (ne kadar gürültü vardı).
  (c) bagged'ın faz7'den ortalama mutlak sapması (küçük olmalı).

Beklenti: public 84.10'dan HAFİF iyileşme veya aynı kalma. Mucize değil.

Çıktı: submissions/sub_seedbag7_faz7.csv + experiments/faz14_seedbag_log.json
       + experiments/oof_seedbag.npy / test_seedbag.npy
"""
import os, json, datetime, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR, N_SPLITS
from faz2_text_meta import run_catboost_cv, metrics
import faz2_text_meta as f2
import faz2_2_embed as f22
import faz3_ensemble as f3
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

SEEDS = [int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,99,555,1').split(',')]


def make_folds_seed(train, seed):
    """faz2_text_meta.make_folds ile aynı stratify (yıl × hedef-desil) ama seed parametrik."""
    y = train[TARGET].values
    tbin = pd.qcut(y, 10, labels=False, duplicates='drop')
    strat = train[YEAR].astype(str) + '_' + pd.Series(tbin).astype(str)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    return list(skf.split(train, strat))


def set_seed(seed):
    """Modül-seviyesi seed sabitlerini bu seed'e ayarla (fold + 3 GBM + Ridge-meta hizalama)."""
    f2.CATBOOST_PARAMS['random_seed'] = seed
    f3.LGBM_PARAMS['random_state'] = seed
    f3.XGB_PARAMS['random_state'] = seed
    f2.SEED = seed      # build_text_meta Ridge (kullanılmıyor ama tutarlılık için)
    f22.SEED = seed     # build_charword_meta / build_emb_meta Ridge random_state
    f3.SEED = seed


def build_featB_X(train_fe, test_fe, txt_tr, txt_te, y, num_cols, cat_cols,
                  emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds):
    """featB feature matrislerini kur: FE + sayısal/kategorik + text_meta + emb_meta + berturk_meta.
    Meta-feature'lar bu seed'in fold'larıyla leakage-free türetilir. Döner: Xc, Xc_t, Xg, Xg_t, cat_idx, feats."""
    tm_tr, tm_te = build_charword_meta(txt_tr, txt_te, y, folds)
    em_tr, em_te = build_emb_meta(emb_tr_raw, emb_te_raw, y, folds)
    meta_map = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te),
                'berturk_meta': (bt_tr, bt_te)}

    cols = num_cols + cat_cols
    Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[cols].copy(); Xg_t = test_fe[cols].copy()
    for c in cat_cols:
        cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
    for m, (a, b) in meta_map.items():
        Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
    feats = cols + list(meta_map)
    ci = [feats.index(c) for c in cat_cols]
    return Xc, Xc_t, Xg, Xg_t, ci, feats


def run_featB(train_fe, test_fe, y, w, txt_tr, txt_te, num_cols, cat_cols,
              emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds):
    """faz7 'B yolu' (featB) tek seed için: FE + 3-GBM + text_meta + emb_meta + berturk_meta."""
    Xc, Xc_t, Xg, Xg_t, ci, feats = build_featB_X(
        train_fe, test_fe, txt_tr, txt_te, y, num_cols, cat_cols,
        emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds)
    oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
    _, (a, b, c) = tune_blend([oc, ol, ox], y, w)
    oof = np.clip(a * oc + b * ol + c * ox, 0, 100)
    tst = np.clip(a * tc + b * tl + c * tx, 0, 100)
    return oof, tst, (a, b, c)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols

    tr_prop = train[YEAR].value_counts(normalize=True)
    te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # SABİT girdiler (tüm seed'lerde aynı): e5 ham embedding + BERTurk OOF/test
    emb_tr_raw = np.load(os.path.join(exp_dir, EMB_TRAIN_NPY))
    emb_te_raw = np.load(os.path.join(exp_dir, EMB_TEST_NPY))
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')

    txt_tr = train[TEXT].fillna(''); txt_te = test[TEXT].fillna('')

    print(f"[seedbag] seeds={SEEDS} | feat: {len(num_cols)} sayısal + {len(cat_cols)} kategorik "
          f"+ text_meta/emb_meta/berturk_meta")

    oof_acc = np.zeros(len(y)); test_acc = np.zeros(len(test))
    seed_tests = []; seed_oofs = []; per_seed = []
    for sd in SEEDS:
        t0 = time.time()
        set_seed(sd)
        folds = make_folds_seed(train, sd)
        oof, tst, bw = run_featB(train_fe, test_fe, y, w, txt_tr, txt_te,
                                 num_cols, cat_cols, emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds)
        p, wt, _ = metrics(y, oof, w, train[YEAR], te_prop)
        oof_acc += oof / len(SEEDS); test_acc += tst / len(SEEDS)
        seed_tests.append(tst); seed_oofs.append(oof)
        per_seed.append({'seed': sd, 'plain': p, 'weighted': wt,
                         'blend_w': {'cat': bw[0], 'lgb': bw[1], 'xgb': bw[2]}})
        print(f"  seed {sd}: wOOF={wt:.4f} (düz {p:.4f}) blend cat/lgb/xgb={tuple(round(x,2) for x in bw)} "
              f"({time.time()-t0:.0f}s)")

    bagged_oof = np.clip(oof_acc, 0, 100)
    bagged_test = np.clip(test_acc, 0, 100)
    bp, bwt, bby = metrics(y, bagged_oof, w, train[YEAR], te_prop)

    # --- Asıl metrikler (köprü kırık -> wOOF'a güvenme) ---
    seed_tests = np.array(seed_tests)               # (N, 10000)
    per_row_std = seed_tests.std(axis=0)            # seed'ler arası gürültü, satır başına
    mean_seed_std = float(per_row_std.mean())

    # faz7 (featB) referans test tahmini = mevcut submission CSV
    faz7_csv = os.path.join(sub_dir, 'sub_2026-06-09_berturk_featB_woof86.09.csv')
    corr = mean_abs_dev = None
    if os.path.exists(faz7_csv):
        faz7_pred = pd.read_csv(faz7_csv).set_index(ID).loc[test[ID].values, TARGET].values
        corr = float(np.corrcoef(bagged_test, faz7_pred)[0, 1])
        mean_abs_dev = float(np.mean(np.abs(bagged_test - faz7_pred)))

    print("\n================ FAZ 14 (featB SEED BAGGING) SONUÇ ================")
    print(f"seeds={SEEDS} (N={len(SEEDS)})")
    print(f"bagged wOOF={bwt:.4f} (düz {bp:.4f})  [REFERANS — köprü kırık, GÜVENME]")
    print(f"per-seed wOOF: {[round(s['weighted'],3) for s in per_seed]}")
    print("\n--- Asıl metrikler ---")
    print(f"(a) corr(bagged, faz7-featB)     = {corr if corr is None else round(corr,5)}  (~0.99 beklenir)")
    print(f"(b) seed'ler arası test std (ort) = {mean_seed_std:.4f}  (silinen gürültü ölçüsü)")
    print(f"(c) bagged'ın faz7'den ort. sapması= {mean_abs_dev if mean_abs_dev is None else round(mean_abs_dev,4)}  (küçük olmalı)")
    print("\nYıl-bazlı bagged OOF MSE:"); print(bby.round(3).to_string())

    # --- Kayıtlar ---
    np.save(os.path.join(exp_dir, 'oof_seedbag.npy'), bagged_oof)
    np.save(os.path.join(exp_dir, 'test_seedbag.npy'), bagged_test)
    log = {'phase': 'faz14_featB_seedbag', 'date': datetime.date.today().isoformat(),
           'seeds': SEEDS, 'base_config': 'featB (faz7: FE+3GBM+text+emb+berturk, ORİJİNAL paramlar)',
           'note': 'wOOF REFERANS; köprü kırık (faz9 dersi). Karar public ile.',
           'bagged': {'plain': bp, 'weighted': bwt, 'by_year': bby.round(4).to_dict()},
           'per_seed': per_seed,
           'key_metrics': {'corr_vs_faz7': corr, 'mean_seed_test_std': mean_seed_std,
                           'mean_abs_dev_vs_faz7': mean_abs_dev}}
    with open(os.path.join(exp_dir, 'faz14_seedbag_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = 'sub_seedbag7_faz7.csv'
    pd.DataFrame({ID: test[ID].values, TARGET: bagged_test}).to_csv(
        os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_seedbag.npy + test_seedbag.npy + faz14_seedbag_log.json + {fname}")


if __name__ == '__main__':
    main()
