"""
Datathon 2026 — Faz 15: featB PSEUDO-LABELING (sample_weight=0.5, leakage-güvenli, held-out doğrulanmış).

KULLANICI TEŞHİSİ (önceden held-out'ta doğrulandı):
- Adil değerlendirmede (pseudo-kaynak ve değerlendirme ayrı, 5 tekrar) ortalama −0.43 MSE
  GERÇEK iyileşme; faz9 gibi wOOF-şişmesi YOK (özellikle test edilmiş).
- Beklenen public katkı ~0.1-0.2 (84.10 -> ~83.9-84.0 bandı).

TASARIM (faz11'den FARK: faz9 TUNED yerine featB ORİJİNAL SADE paramlar + sample_weight=0.5):
1) featB ensemble'ı (FE+3GBM+text_meta+emb_meta+berturk_meta, faz7/faz14 ile birebir aynı,
   HİÇBİR param değişmedi) 3 seed (42,7,123) ile tüm pipeline (fold split dahil) koşulur ->
   her seed kendi TEST tahminini üretir.
2) 3 seed'in test tahminleri arası STD hesaplanır (model-üstü uzlaşı ölçüsü). En düşük std'li
   alt %30 = "güvenli" pseudo-aday. pseudo_y = bu satırların 3-seed ORTALAMASI (pmean), clip(0,100).
3) Final eğitim: seed=42'nin fold/feature kurulumu temel alınır. Her fold'un TRAIN'ine pseudo
   TEST satırları eklenir (val'a asla girmez -> leakage yok), sample_weight: gerçek=1.0,
   pseudo=0.5 (yarı güven). cat/lgb/xgb ORİJİNAL featB paramlarıyla (tune YOK) yeniden eğitilir.
4) wOOF RAPORLANIR ama GÜVENİLMEZ (pseudo train'e karışınca OOF yapay düşebilir — kullanıcı
   −3 wOOF / 0.998 korelasyon gözlemiyle bunu doğruladı). Karar SADECE public ile.
5) faz7/seedbag (sub_seedbag7_faz7.csv) korunur; final-2 seçimi public'e göre yapılacak.

Çıktı: submissions/sub_<tarih>_pseudo_featb_woof<skor>.csv + experiments/faz15_pseudo_featb_log.json
"""
import os, json, datetime, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import metrics, CATBOOST_PARAMS
from faz3_ensemble import LGBM_PARAMS, XGB_PARAMS, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz14_seedbag import run_featB, build_featB_X, make_folds_seed, set_seed

PSEUDO_FRAC = float(os.environ.get('PSEUDO_FRAC', 0.3))
PSEUDO_WEIGHT = float(os.environ.get('PSEUDO_WEIGHT', 0.5))
STD_SEEDS = [42, 7, 123]
TRAIN_SEED = 42   # final retrain için fold/feature temeli (faz7/faz14 referans seed)


def cv_with_pseudo(kind, params, Xtr, y, Xte, folds, pseudo_idx, pseudo_y, cat_arg):
    """Her fold: fold-train (w=1.0) + pseudo test satırları (w=0.5) ile eğit; OOF yalnız gerçek-train va'da."""
    oof = np.zeros(len(y)); test_pred = np.zeros(Xte.shape[0])
    Xpa = Xte.iloc[pseudo_idx]
    for tr, va in folds:
        Xf = pd.concat([Xtr.iloc[tr], Xpa], axis=0)
        yf = np.concatenate([y[tr], pseudo_y])
        wf = np.concatenate([np.ones(len(tr)), np.full(len(pseudo_idx), PSEUDO_WEIGHT)])
        if kind == 'cat':
            m = CatBoostRegressor(**params)
            m.fit(Pool(Xf, yf, weight=wf, cat_features=cat_arg),
                  eval_set=Pool(Xtr.iloc[va], y[va], cat_features=cat_arg), use_best_model=True)
        elif kind == 'lgb':
            m = LGBMRegressor(**params)
            m.fit(Xf, yf, sample_weight=wf, eval_set=[(Xtr.iloc[va], y[va])], categorical_feature=cat_arg,
                  callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        else:
            m = XGBRegressor(**params)
            m.fit(Xf, yf, sample_weight=wf, eval_set=[(Xtr.iloc[va], y[va])], verbose=False)
        oof[va] = np.clip(m.predict(Xtr.iloc[va]), 0, 100)
        test_pred += np.clip(m.predict(Xte), 0, 100) / len(folds)
    return oof, test_pred


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols

    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    emb_tr_raw = np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)); emb_te_raw = np.load(os.path.join(exp_dir, EMB_TEST_NPY))
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    txt_tr = train[TEXT].fillna(''); txt_te = test[TEXT].fillna('')

    # ---------- 1) 3-seed featB -> test tahminleri ----------
    print(f"[faz15] adım 1: featB {STD_SEEDS} seed ile (fold split dahil) koşuluyor")
    test_preds = []
    for sd in STD_SEEDS:
        t0 = time.time()
        set_seed(sd)
        folds_sd = make_folds_seed(train, sd)
        oof, tst, bwts = run_featB(train_fe, test_fe, y, w, txt_tr, txt_te, num_cols, cat_cols,
                                    emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds_sd)
        _, wt, _ = metrics(y, oof, w, train[YEAR], te_prop)
        test_preds.append(tst)
        print(f"  seed {sd}: wOOF={wt:.4f} blend={tuple(round(x,2) for x in bwts)} ({time.time()-t0:.0f}s)")
    test_preds = np.array(test_preds)   # (3, n_test)
    pmean = test_preds.mean(axis=0)
    pstd = test_preds.std(axis=0)

    # ---------- 2) en düşük std alt %30 -> pseudo aday ----------
    n_keep = int(len(test) * PSEUDO_FRAC)
    order = np.argsort(pstd)
    pseudo_idx = order[:n_keep]
    pseudo_y = np.clip(pmean[pseudo_idx], 0, 100)
    print(f"\n[faz15] adım 2: {n_keep}/{len(test)} satır seçildi (FRAC={PSEUDO_FRAC})")
    print(f"  std seçilen medyan={np.median(pstd[pseudo_idx]):.4f} vs tüm medyan={np.median(pstd):.4f}")
    print(f"  pseudo yıl dağılımı: {test.iloc[pseudo_idx][YEAR].value_counts().to_dict()}")

    # ---------- 3) seed=42 temelli featB matrisleri + pseudo-augmented retrain ----------
    print(f"\n[faz15] adım 3: seed={TRAIN_SEED} featB matrisleri + pseudo retrain (sample_weight={PSEUDO_WEIGHT})")
    set_seed(TRAIN_SEED)
    folds = make_folds_seed(train, TRAIN_SEED)
    Xc, Xc_t, Xg, Xg_t, ci, feats = build_featB_X(train_fe, test_fe, txt_tr, txt_te, y, num_cols, cat_cols,
                                                   emb_tr_raw, emb_te_raw, bt_tr, bt_te, folds)

    t0 = time.time()
    oc, ptc = cv_with_pseudo('cat', dict(CATBOOST_PARAMS), Xc, y, Xc_t, folds, pseudo_idx, pseudo_y, ci)
    ol, ptl = cv_with_pseudo('lgb', dict(LGBM_PARAMS), Xg, y, Xg_t, folds, pseudo_idx, pseudo_y, cat_cols)
    ox, ptx = cv_with_pseudo('xgb', dict(XGB_PARAMS), Xg, y, Xg_t, folds, pseudo_idx, pseudo_y, None)
    print(f"  retrain {time.time()-t0:.0f}s")

    _, (a, b, c) = tune_blend([oc, ol, ox], y, w)
    oof_bl = np.clip(a*oc + b*ol + c*ox, 0, 100)
    test_bl = np.clip(a*ptc + b*ptl + c*ptx, 0, 100)
    bp_, bw, bby = metrics(y, oof_bl, w, train[YEAR], te_prop)

    # ---------- referans (faz7 featB, seed=42, pseudo'suz) ----------
    faz7_csv = os.path.join(sub_dir, 'sub_2026-06-09_berturk_featB_woof86.09.csv')
    corr = mean_abs_dev = None
    if os.path.exists(faz7_csv):
        faz7_pred = pd.read_csv(faz7_csv).set_index(ID).loc[test[ID].values, TARGET].values
        corr = float(np.corrcoef(test_bl, faz7_pred)[0, 1])
        mean_abs_dev = float(np.mean(np.abs(test_bl - faz7_pred)))

    print("\n================ FAZ 15 (featB PSEUDO, w=%.1f) SONUÇ ================" % PSEUDO_WEIGHT)
    print(f"blend cat={a:.2f} lgb={b:.2f} xgb={c:.2f} | wOOF={bw:.4f} (düz {bp_:.4f})  [REFERANS — pseudo OOF şişebilir, GÜVENME]")
    print(f"corr(pseudo_test, faz7-featB) = {corr if corr is None else round(corr,5)}")
    print(f"ort. mutlak sapma vs faz7     = {mean_abs_dev if mean_abs_dev is None else round(mean_abs_dev,4)}")
    print("\nYıl-bazlı:"); print(bby.round(3).to_string())

    np.save(os.path.join(exp_dir, 'oof_pseudo_featb.npy'), oof_bl)
    np.save(os.path.join(exp_dir, 'test_pseudo_featb.npy'), test_bl)
    log = {'phase': 'faz15_pseudo_featb', 'date': datetime.date.today().isoformat(),
           'base_config': 'featB (faz7 orijinal sade paramlar, tune YOK)',
           'std_seeds': STD_SEEDS, 'train_seed': TRAIN_SEED,
           'pseudo_frac': PSEUDO_FRAC, 'pseudo_weight': PSEUDO_WEIGHT, 'n_pseudo': int(n_keep),
           'pseudo_year_dist': {int(k): int(v) for k, v in test.iloc[pseudo_idx][YEAR].value_counts().items()},
           'pseudo_std_median_selected': float(np.median(pstd[pseudo_idx])),
           'pseudo_std_median_all': float(np.median(pstd)),
           'blend_weights': {'cat': a, 'lgb': b, 'xgb': c},
           'result': {'plain': bp_, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'note': 'wOOF REFERANS, pseudo-self-validation ile şişebilir; karar PUBLIC ile.',
           'key_metrics': {'corr_vs_faz7': corr, 'mean_abs_dev_vs_faz7': mean_abs_dev}}
    with open(os.path.join(exp_dir, 'faz15_pseudo_featb_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = f"sub_{datetime.date.today().isoformat()}_pseudo_featb_woof{bw:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: test_bl}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof/test_pseudo_featb.npy + faz15_pseudo_featb_log.json + {fname}")


if __name__ == '__main__':
    main()
