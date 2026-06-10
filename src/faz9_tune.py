"""
Datathon 2026 — Faz 9: GBM HİPERPARAMETRE TUNING (Optuna) + SEED BAGGING.

NEDEN (en bariz açık): cat/lgb/xgb şimdiye dek default-yakını paramlarla koştu, hiç sistematik
tune edilmedi. Klasik +0.2-0.5 kaynağı, köprü GBM tarafında sağlam → "önce güvenli+bedava".

TASARIM (köprü bütünlüğü korunur):
- AYNI fold'lar (make_folds, yıl×hedef-desil, SEED=42). AYNI winner feature seti (Faz 7-B):
  engineer() FE sayısal + kategorik + text_meta(charword) + emb_meta + berturk_meta.
  → tuned paramlar final ensemble'a birebir transfer olur.
- Objektif = test-yıl-ağırlıklı OOF MSE (public proxy), her GBM AYRI tune edilir (TPE).
- iterations/n_estimators TUNE EDİLMEZ: yüksek tavan + early-stopping fold başına en iyi
  iterasyonu zaten bulur (sabit iterasyon tune etmekten daha robust). depth/lr/l2/reg/
  subsample/colsample/min_child/bagging tune edilir.
- Meta-feature'lar (text/emb/berturk) GBM paramından BAĞIMSIZ → BİR KEZ hesaplanır, tüm
  trial'larda yeniden kullanılır (leakage-free OOF; recompute israfı yok).
- SEED BAGGING: tuned her model N seed ile koşulur, OOF+test seed-ortalanır (varyans↓, +0.1-0.3).
  Fold'lar sabit; yalnız model random_seed değişir.

ÇIKTILAR:
- experiments/faz9_tuned_params.json   : her modelin en iyi paramları + tuning skor geçmişi
- experiments/oof_{cat,lgb,xgb}_tuned.npy, oof_blend_tuned.npy
- experiments/test_blend_tuned.npy     : (faz11/faz12 girdisi)
- experiments/faz9_tune_log.json       : tekil + blend wOOF, yıl kırılımı, bagging detayı
- submissions/sub_<tarih>_tuned_bagged_woof<skor>.csv

KOŞMA:
- Lokal: train.csv /tmp/data altında + experiments/ cache'leri (emb_e5_*, *_berturk*) hazır.
- Kaggle: repo'yu dataset olarak ekle, `N_TRIALS=50 python src/faz9_tune.py`.
- Ayar: N_TRIALS (vars. 50), SEEDS env ile değiştirilebilir.
"""
import os, json, datetime
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

import optuna
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = int(os.environ.get('N_TRIALS', 50))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '42,1337,2024,7,99').split(',')]
CAT_ITERS = int(os.environ.get('CAT_ITERS', 5000))
GBM_ESTIMATORS = int(os.environ.get('GBM_ESTIMATORS', 5000))
OD_WAIT = 200


# ----------------------- Parametrize CV runner'lar (params + seed) -----------------------
def cv_catboost(Xc, y, Xc_t, cat_idx, folds, params, seed):
    p = dict(loss_function='RMSE', eval_metric='RMSE', iterations=CAT_ITERS,
             od_type='Iter', od_wait=OD_WAIT, random_seed=seed,
             verbose=False, allow_writing_files=False)
    p.update(params)
    oof = np.zeros(len(Xc)); test = np.zeros(len(Xc_t)); iters = []
    for tr, va in folds:
        m = CatBoostRegressor(**p)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat_idx), use_best_model=True)
        oof[va] = m.predict(Xc.iloc[va]); test += m.predict(Xc_t) / len(folds)
        iters.append(int(m.get_best_iteration()))
    return np.clip(oof, 0, 100), np.clip(test, 0, 100), iters


def cv_lgbm(Xg, y, Xg_t, cat_cols, folds, params, seed):
    p = dict(objective='regression', metric='l2', n_estimators=GBM_ESTIMATORS,
             random_state=seed, n_jobs=-1, verbosity=-1)
    p.update(params)
    oof = np.zeros(len(Xg)); test = np.zeros(len(Xg_t)); iters = []
    for tr, va in folds:
        m = LGBMRegressor(**p)
        m.fit(Xg.iloc[tr], y[tr], eval_set=[(Xg.iloc[va], y[va])],
              categorical_feature=cat_cols,
              callbacks=[early_stopping(OD_WAIT, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(Xg.iloc[va]); test += m.predict(Xg_t) / len(folds)
        iters.append(int(m.best_iteration_ or GBM_ESTIMATORS))
    return np.clip(oof, 0, 100), np.clip(test, 0, 100), iters


def cv_xgb(Xg, y, Xg_t, folds, params, seed):
    p = dict(objective='reg:squarederror', eval_metric='rmse', n_estimators=GBM_ESTIMATORS,
             tree_method='hist', enable_categorical=True, random_state=seed, n_jobs=-1,
             early_stopping_rounds=OD_WAIT)
    p.update(params)
    oof = np.zeros(len(Xg)); test = np.zeros(len(Xg_t)); iters = []
    for tr, va in folds:
        m = XGBRegressor(**p)
        m.fit(Xg.iloc[tr], y[tr], eval_set=[(Xg.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict(Xg.iloc[va]); test += m.predict(Xg_t) / len(folds)
        iters.append(int(m.best_iteration or GBM_ESTIMATORS))
    return np.clip(oof, 0, 100), np.clip(test, 0, 100), iters


# ----------------------- Optuna arama uzayları (depth/lr/l2/reg/subsample/bagging) ----------
def space_cat(trial):
    return dict(
        depth=trial.suggest_int('depth', 4, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
        l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 30.0, log=True),
        random_strength=trial.suggest_float('random_strength', 0.0, 5.0),
        bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.0),
        border_count=trial.suggest_int('border_count', 64, 254),
    )


def space_lgbm(trial):
    return dict(
        num_leaves=trial.suggest_int('num_leaves', 15, 255),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        subsample_freq=trial.suggest_int('subsample_freq', 1, 5),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 30.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
    )


def space_xgb(trial):
    return dict(
        max_depth=trial.suggest_int('max_depth', 3, 10),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
        min_child_weight=trial.suggest_float('min_child_weight', 1.0, 20.0),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 30.0, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        gamma=trial.suggest_float('gamma', 0.0, 5.0),
    )


def tune_model(kind, run_cv, space, y, w, years, te_prop, n_trials):
    """Tek GBM'i wOOF objektifiyle tune et (TPE, sabit SEED fold + tek seed model)."""
    def objective(trial):
        params = space(trial)
        oof, _, _ = run_cv(params, SEED)
        _, wm, _ = metrics(y, oof, w, years, te_prop)
        return wm
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  [{kind}] en iyi wOOF={study.best_value:.4f}  params={study.best_params}")
    return study


def bag(run_cv, best_params, seeds):
    """Tuned modeli N seed ile koş, OOF+test seed-ortala (varyans↓)."""
    oofs, tests, iters_all = [], [], []
    for s in seeds:
        o, t, it = run_cv(best_params, s)
        oofs.append(o); tests.append(t); iters_all.append(it)
    return (np.clip(np.mean(oofs, 0), 0, 100),
            np.clip(np.mean(tests, 0), 0, 100), iters_all)


def prepare():
    """Faz 7 winner ile BİREBİR aynı veri+FE+fold+meta+model-matrislerini kur (faz9/11/12 ortak).

    Dönen sözlük: train/test df, y, w, years, te_prop, folds, cat_cols, cat_idx,
    Xc/Xc_t (CatBoost str-cat), Xg/Xg_t (LGBM/XGB category dtype) — hepsinde meta kolonları DAHİL.
    """
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
    years = train[YEAR]

    # --- Meta-feature'lar (Faz 7 winner ile aynı; BİR KEZ, GBM paramından bağımsız) ---
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    meta = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te), 'berturk_meta': (bt_tr, bt_te)}

    # --- Model-bazlı X (CatBoost str cat / LGBM-XGB category dtype) ---
    base_cols = num_cols + cat_cols
    Xc = train_fe[base_cols].copy(); Xc_t = test_fe[base_cols].copy()
    for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[base_cols].copy(); Xg_t = test_fe[base_cols].copy()
    for c in cat_cols:
        cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
    for m, (a, b) in meta.items():
        Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
    feats_c = base_cols + list(meta); cat_idx = [feats_c.index(c) for c in cat_cols]
    print(f"[feat] {len(num_cols)} sayısal + {len(cat_cols)} kategorik + meta={list(meta)} "
          f"(toplam {len(feats_c)})")
    return dict(train=train, test=test, y=y, w=w, years=years, te_prop=te_prop, folds=folds,
                cat_cols=cat_cols, cat_idx=cat_idx, Xc=Xc, Xc_t=Xc_t, Xg=Xg, Xg_t=Xg_t,
                exp_dir=exp_dir, sub_dir=sub_dir, meta=meta)


def main():
    P = prepare()
    y, w, years, te_prop, folds = P['y'], P['w'], P['years'], P['te_prop'], P['folds']
    cat_cols, cat_idx = P['cat_cols'], P['cat_idx']
    Xc, Xc_t, Xg, Xg_t = P['Xc'], P['Xc_t'], P['Xg'], P['Xg_t']
    exp_dir, sub_dir, test, meta = P['exp_dir'], P['sub_dir'], P['test'], P['meta']

    # CV closure'ları (X sabit; sadece params+seed değişir)
    run_cat = lambda params, seed: cv_catboost(Xc, y, Xc_t, cat_idx, folds, params, seed)
    run_lgb = lambda params, seed: cv_lgbm(Xg, y, Xg_t, cat_cols, folds, params, seed)
    run_xgb = lambda params, seed: cv_xgb(Xg, y, Xg_t, folds, params, seed)

    # ---------------- ADIM 1: tuning (her GBM ayrı) ----------------
    print(f"\n================ ADIM 1: OPTUNA TUNING ({N_TRIALS} trial/model) ================")
    studies = {}
    for kind, run_cv, space in [('cat', run_cat, space_cat),
                                ('lgb', run_lgb, space_lgbm),
                                ('xgb', run_xgb, space_xgb)]:
        studies[kind] = tune_model(kind, run_cv, space, y, w, years, te_prop, N_TRIALS)
    best = {k: s.best_params for k, s in studies.items()}

    # ---------------- ADIM 2: seed bagging (tuned paramlarla) ----------------
    print(f"\n================ ADIM 2: SEED BAGGING (seeds={SEEDS}) ================")
    oof_cat, test_cat, it_cat = bag(run_cat, best['cat'], SEEDS)
    oof_lgb, test_lgb, it_lgb = bag(run_lgb, best['lgb'], SEEDS)
    oof_xgb, test_xgb, it_xgb = bag(run_xgb, best['xgb'], SEEDS)

    models = {'cat': oof_cat, 'lgb': oof_lgb, 'xgb': oof_xgb}
    tests = {'cat': test_cat, 'lgb': test_lgb, 'xgb': test_xgb}
    print("\n--- Tuned+bagged tekil OOF (düz | ağırlıklı) ---")
    indiv = {}
    for k, o in models.items():
        p, wt, by = metrics(y, o, w, years, te_prop)
        indiv[k] = {'plain': p, 'weighted': wt, 'by_year': by.round(4).to_dict()}
        print(f"  {k}: {p:.4f} | {wt:.4f}")

    # ---------------- Blend (OOF ağırlık taraması, public proxy hedefi) ----------------
    _, (wc, wl, wx) = tune_blend([oof_cat, oof_lgb, oof_xgb], y, w)
    oof_blend = np.clip(wc * oof_cat + wl * oof_lgb + wx * oof_xgb, 0, 100)
    test_blend = np.clip(wc * test_cat + wl * test_lgb + wx * test_xgb, 0, 100)
    b_plain, b_weighted, b_by_year = metrics(y, oof_blend, w, years, te_prop)

    REF = 86.09  # Faz 7-B winner wOOF (public 84.10)
    print("\n================ FAZ 9 SONUÇ (tuned+bagged, ağırlıklı OOF) ================")
    print(f"Blend ağırlıkları: cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f}")
    print(f"Blend ağırlıklı OOF : {b_weighted:.4f}  (düz {b_plain:.4f})")
    print(f"  kazanç vs Faz 7-B : {REF - b_weighted:+.4f}  (winner wOOF {REF}, public 84.10)")
    print("\nYıl-bazlı blend OOF MSE (2025-2026 ağır test payı):")
    print(b_by_year.round(3).to_string())

    # ---------------- Kayıtlar ----------------
    for k, o in models.items():
        np.save(os.path.join(exp_dir, f'oof_{k}_tuned.npy'), o)
        np.save(os.path.join(exp_dir, f'test_{k}_tuned.npy'), tests[k])
    np.save(os.path.join(exp_dir, 'oof_blend_tuned.npy'), oof_blend)
    np.save(os.path.join(exp_dir, 'test_blend_tuned.npy'), test_blend)
    with open(os.path.join(exp_dir, 'faz9_tuned_params.json'), 'w') as f:
        json.dump({k: {'best_params': s.best_params, 'best_woof': s.best_value,
                       'n_trials': len(s.trials)} for k, s in studies.items()},
                  f, indent=2, ensure_ascii=False)
    log = {'phase': 'faz9_tune_bagging', 'date': datetime.date.today().isoformat(),
           'n_trials': N_TRIALS, 'seeds': SEEDS, 'meta_cols': list(meta),
           'best_params': best, 'individual': indiv,
           'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
           'blend': {'plain': b_plain, 'weighted': b_weighted, 'by_year': b_by_year.round(4).to_dict()},
           'ref_faz7b_weighted': REF, 'gain_vs_faz7b': REF - b_weighted,
           'bag_iters': {'cat': it_cat, 'lgb': it_lgb, 'xgb': it_xgb}}
    with open(os.path.join(exp_dir, 'faz9_tune_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_blend})
    fname = f"sub_{datetime.date.today().isoformat()}_tuned_bagged_woof{b_weighted:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof/test_*_tuned.npy + faz9_tuned_params.json + faz9_tune_log.json + {fname}")


if __name__ == '__main__':
    main()
