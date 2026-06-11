"""
Datathon 2026 — Faz 21 (RADİKAL B): adversarial validation weighting.

Train-vs-test classifier (LGBM, tüm feature'lar) -> p(test-like) -> w_adv = p/(1-p).
Yıl-ağırlığından ince: tüm feature uzayında kayma. UYARI: Faz 6'da saf yıl-sample_weight
EĞİTİMDE -0.85 batmıştı; bu daha akıllı versiyonu ama aynı tuzağa düşebilir. Ucuz test:
stack_v3 GBM tabanlarından yalnız CatBoost'u w_adv*0.5+0.5 karışık ağırlıkla yeniden eğit,
fark wOOF'ta BÜYÜKSE devam, değilse reddet (varsayılan beklenti: RED).
"""
import os, json, datetime
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor, Pool

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    folds, _ = make_folds(train)
    trp = train[YEAR].value_counts(normalize=True); tep = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda v: tep.get(v, 0.0) / trp.get(v, np.nan)).values)
    def wm(p): return float(np.sum(w*(y-p)**2)/np.sum(w))

    # ---------- adversarial classifier (ham feature'lar) ----------
    A = pd.concat([train[raw_num + cat_cols], test[raw_num + cat_cols]], axis=0).reset_index(drop=True)
    for c in cat_cols: A[c] = A[c].astype('category')
    lab = np.r_[np.zeros(len(train)), np.ones(len(test))]
    clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED,
                         n_jobs=-1, verbosity=-1)
    # OOF olasılık (5-fold) -> şişmesiz w_adv
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    p = np.zeros(len(A))
    for tr, va in skf.split(A, lab):
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=SEED,
                           n_jobs=-1, verbosity=-1)
        m.fit(A.iloc[tr], lab[tr]); p[va] = m.predict_proba(A.iloc[va])[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(lab, p)
    p_tr = np.clip(p[:len(train)], 0.02, 0.98)
    w_adv = p_tr / (1 - p_tr); w_adv = w_adv / w_adv.mean()
    print(f"[adv] AUC={auc:.4f} (0.5=ayırt edilemiyor, >0.7=belirgin kayma)")
    print(f"[adv] w_adv: min={w_adv.min():.3f} med={np.median(w_adv):.3f} max={w_adv.max():.3f}")
    print(f"[adv] yıl ile korelasyon: {np.corrcoef(w_adv, train[YEAR])[0,1]:.3f} (yüksekse = zaten yıl-ağırlığı)")

    # ---------- ucuz test: CatBoost (5-meta) yumuşak ağırlıkla ----------
    train_fe, test_fe = engineer(train), engineer(test)
    eng = [c for c in train_fe.columns if c not in train.columns]; num_cols = raw_num + eng
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    metas = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te)}
    for tag, of, tf in [('berturk_meta','oof_berturk_train.npy','berturk_test.npy'),
                        ('bert128k_meta','oof_bert128k_train.npy','bert128k_test.npy'),
                        ('electra_meta','oof_electra_train.npy','electra_test.npy')]:
        metas[tag] = (np.clip(np.load(os.path.join(exp_dir,of)).astype('float64'),0,100),
                      np.clip(np.load(os.path.join(exp_dir,tf)).astype('float64'),0,100))
    cols = num_cols + cat_cols
    Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
    for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    for m,(a,b) in metas.items(): Xc[m]=a; Xc_t[m]=b
    feats = cols + list(metas); ci = [feats.index(c) for c in cat_cols]

    # referans (ağırlıksız) — faz20'dan oof_stack üyesi cat'la aynı pipeline; burada tekrar (hızlı)
    oof0, _, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    print(f"[ref] cat (ağırlıksız) wOOF={wm(oof0):.4f}")
    # yumuşak adv ağırlık (0.5 + 0.5*w_adv)
    sw = 0.5 + 0.5*w_adv
    params = dict(loss_function='RMSE', eval_metric='RMSE', iterations=3000, learning_rate=0.03,
                  depth=6, l2_leaf_reg=3.0, random_seed=SEED, od_type='Iter', od_wait=200,
                  verbose=False, allow_writing_files=False)
    oof1 = np.zeros(len(y))
    for tr, va in folds:
        m = CatBoostRegressor(**params)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=ci, weight=sw[tr]),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=ci), use_best_model=True)
        oof1[va] = m.predict(Xc.iloc[va])
    oof1 = np.clip(oof1, 0, 100)
    print(f"[adv] cat (w_adv yumuşak) wOOF={wm(oof1):.4f}  (delta {wm(oof0)-wm(oof1):+.4f})")
    verdict = 'DEVAM' if wm(oof0)-wm(oof1) > 0.3 else 'RED (Faz6 patterni)'
    print(f"VERDICT: {verdict}")
    json.dump({'auc':auc,'ref_woof':wm(oof0),'adv_woof':wm(oof1),'verdict':verdict},
              open(os.path.join(exp_dir,'faz21_adversarial_log.json'),'w'), indent=2)


if __name__ == '__main__':
    main()
