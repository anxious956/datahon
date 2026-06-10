"""
Datathon 2026 — Faz 9: GBM hiperparametre tuning (Optuna, test-yıl-ağırlıklı OOF hedefi).

NEDEN (savaş planı #1): Şimdiye dek 3 GBM default-yakını paramlarla koştu, hiç sistematik
tune edilmedi -> en bariz açık. GBM köprüsü sağlam (wOOF guvenilir) -> burada kazanç GERÇEK.

TASARIM:
- Winner feature seti (Faz 7-B featB): FE(sentiment'siz) + text_meta + emb_meta + berturk_meta.
- Her GBM AYRI tune (cat/lgb/xgb), hedef = test-yıl-ağırlıklı OOF MSE, AYNI fold'lar (SEED=42).
- Fold-bazlı pruning (MedianPruner): kötü trial'lar erken kesilir -> 50-100 trial tractable.
- Tuned paramlar + tuned OOF/test kaydedilir (downstream: seed-bagging, stacker bunları kullanır).
- Blend ağırlıkları tekrar taranır; final wOOF + yıl-bazlı raporlanır.
- Referans: Faz 7-B featB wOOF 86.09 (default-yakını). Buradaki düşüş = saf tuning kazancı.

n_trials env ile: CAT_TRIALS/LGB_TRIALS/XGB_TRIALS (default 60/80/80). Smoke: hepsi=2.
"""
import os, json, datetime, time
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_matrices(train, test, y, folds, exp_dir):
    """Winner feature seti -> CatBoost (str cat) ve LGBM/XGB (category dtype) matrisleri + meta."""
    train_fe, test_fe = engineer(train), engineer(test)              # sentiment=False (reddedildi)
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    cols = raw_num + eng_cols + cat_cols

    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    meta = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te), 'berturk_meta': (bt_tr, bt_te)}

    Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
    for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[cols].copy(); Xg_t = test_fe[cols].copy()
    for c in cat_cols:
        cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
    for m, (a, b) in meta.items():
        Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
    feats = cols + list(meta)
    cat_idx = [feats.index(c) for c in cat_cols]
    return Xc, Xc_t, Xg, Xg_t, cat_cols, cat_idx


def cat_cv(params, Xc, y, Xc_t, cat_idx, folds, w, trial=None, predict_test=False):
    oof = np.zeros(len(y)); test_pred = np.zeros(Xc_t.shape[0]); sw = se = 0.0
    for i, (tr, va) in enumerate(folds):
        m = CatBoostRegressor(**params)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=cat_idx), use_best_model=True)
        p = np.clip(m.predict(Xc.iloc[va]), 0, 100); oof[va] = p
        if predict_test: test_pred += np.clip(m.predict(Xc_t), 0, 100) / len(folds)
        if trial is not None:
            se += np.sum(w[va] * (y[va] - p) ** 2); sw += np.sum(w[va])
            trial.report(se / sw, i)
            if trial.should_prune(): raise optuna.TrialPruned()
    return oof, test_pred


def lgb_cv(params, Xg, y, Xg_t, cat_cols, folds, w, trial=None, predict_test=False):
    oof = np.zeros(len(y)); test_pred = np.zeros(Xg_t.shape[0]); sw = se = 0.0
    for i, (tr, va) in enumerate(folds):
        m = LGBMRegressor(**params)
        m.fit(Xg.iloc[tr], y[tr], eval_set=[(Xg.iloc[va], y[va])], categorical_feature=cat_cols,
              callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        p = np.clip(m.predict(Xg.iloc[va]), 0, 100); oof[va] = p
        if predict_test: test_pred += np.clip(m.predict(Xg_t), 0, 100) / len(folds)
        if trial is not None:
            se += np.sum(w[va] * (y[va] - p) ** 2); sw += np.sum(w[va])
            trial.report(se / sw, i)
            if trial.should_prune(): raise optuna.TrialPruned()
    return oof, test_pred


def xgb_cv(params, Xg, y, Xg_t, folds, w, trial=None, predict_test=False):
    oof = np.zeros(len(y)); test_pred = np.zeros(Xg_t.shape[0]); sw = se = 0.0
    for i, (tr, va) in enumerate(folds):
        m = XGBRegressor(**params)
        m.fit(Xg.iloc[tr], y[tr], eval_set=[(Xg.iloc[va], y[va])], verbose=False)
        p = np.clip(m.predict(Xg.iloc[va]), 0, 100); oof[va] = p
        if predict_test: test_pred += np.clip(m.predict(Xg_t), 0, 100) / len(folds)
        if trial is not None:
            se += np.sum(w[va] * (y[va] - p) ** 2); sw += np.sum(w[va])
            trial.report(se / sw, i)
            if trial.should_prune(): raise optuna.TrialPruned()
    return oof, test_pred


def wmse(y, oof, w): return float(np.sum(w * (y - oof) ** 2) / np.sum(w))


def tune(name, n_trials, objective):
    if n_trials <= 0: return None
    t0 = time.time()
    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    def cb(st, tr):  # canlı ilerleme (buffer'a takılmasın diye her trial yazar)
        tag = 'PRUNE' if tr.state.name == 'PRUNED' else f"{tr.value:.4f}" if tr.value else '-'
        print(f"  [{name}] trial {tr.number+1}/{n_trials} {tag} | best={st.best_value:.4f} | {time.time()-t0:.0f}s", flush=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[cb])
    print(f"[{name}] {n_trials} trial, en iyi wOOF={study.best_value:.4f} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  params: {study.best_params}", flush=True)
    return study


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    Xc, Xc_t, Xg, Xg_t, cat_cols, cat_idx = build_matrices(train, test, y, folds, exp_dir)
    print(f"[feat] cat-matris {Xc.shape}, gbm-matris {Xg.shape}, {len(cat_cols)} kategorik")

    CT = int(os.environ.get('CAT_TRIALS', 60))
    LT = int(os.environ.get('LGB_TRIALS', 80))
    XT = int(os.environ.get('XGB_TRIALS', 80))

    # ---------- CatBoost ----------
    def cat_obj(trial):
        params = dict(loss_function='RMSE', eval_metric='RMSE', iterations=1500,
                      random_seed=SEED, od_type='Iter', od_wait=150, verbose=False,
                      allow_writing_files=False,
                      depth=trial.suggest_int('depth', 4, 8),
                      learning_rate=trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
                      l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True),
                      random_strength=trial.suggest_float('random_strength', 0.0, 3.0),
                      bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 1.5),
                      border_count=trial.suggest_int('border_count', 64, 255))
        oof, _ = cat_cv(params, Xc, y, Xc_t, cat_idx, folds, w, trial=trial)
        return wmse(y, oof, w)
    st_cat = tune('cat', CT, cat_obj)

    # ---------- LightGBM ----------
    def lgb_obj(trial):
        params = dict(objective='regression', metric='l2', n_estimators=2500,
                      random_state=SEED, n_jobs=-1, verbosity=-1, subsample_freq=1,
                      learning_rate=trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
                      num_leaves=trial.suggest_int('num_leaves', 15, 160),
                      min_child_samples=trial.suggest_int('min_child_samples', 5, 120),
                      subsample=trial.suggest_float('subsample', 0.6, 1.0),
                      colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                      reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 30.0, log=True),
                      reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True))
        oof, _ = lgb_cv(params, Xg, y, Xg_t, cat_cols, folds, w, trial=trial)
        return wmse(y, oof, w)
    st_lgb = tune('lgb', LT, lgb_obj)

    # ---------- XGBoost ----------
    def xgb_obj(trial):
        params = dict(objective='reg:squarederror', eval_metric='rmse', n_estimators=2500,
                      tree_method='hist', enable_categorical=True, random_state=SEED, n_jobs=-1,
                      early_stopping_rounds=150,
                      learning_rate=trial.suggest_float('learning_rate', 0.015, 0.12, log=True),
                      max_depth=trial.suggest_int('max_depth', 4, 9),
                      min_child_weight=trial.suggest_float('min_child_weight', 1.0, 15.0),
                      subsample=trial.suggest_float('subsample', 0.6, 1.0),
                      colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                      gamma=trial.suggest_float('gamma', 0.0, 5.0),
                      reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 30.0, log=True),
                      reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True))
        oof, _ = xgb_cv(params, Xg, y, Xg_t, folds, w, trial=trial)
        return wmse(y, oof, w)
    st_xgb = tune('xgb', XT, xgb_obj)

    # ---------- Tuned paramlarla yeniden eğit (OOF + test) ----------
    print("\n[retrain] en iyi paramlarla full CV (test tahmini dahil)...")
    cat_p = dict(loss_function='RMSE', eval_metric='RMSE', iterations=3000, random_seed=SEED,
                 od_type='Iter', od_wait=200, verbose=False, allow_writing_files=False, **st_cat.best_params)
    lgb_p = dict(objective='regression', metric='l2', n_estimators=5000, random_state=SEED,
                 n_jobs=-1, verbosity=-1, subsample_freq=1, **st_lgb.best_params)
    xgb_p = dict(objective='reg:squarederror', eval_metric='rmse', n_estimators=5000, tree_method='hist',
                 enable_categorical=True, random_state=SEED, n_jobs=-1, early_stopping_rounds=200, **st_xgb.best_params)
    oc, tc = cat_cv(cat_p, Xc, y, Xc_t, cat_idx, folds, w, predict_test=True)
    ol, tl = lgb_cv(lgb_p, Xg, y, Xg_t, cat_cols, folds, w, predict_test=True)
    ox, tx = xgb_cv(xgb_p, Xg, y, Xg_t, folds, w, predict_test=True)

    print("\n--- Tuned tekil model (düz | ağırlıklı) | default-yakını referans ---")
    ref = {'cat': 86.89, 'lgb': None, 'xgb': None}
    for k, o in [('cat', oc), ('lgb', ol), ('xgb', ox)]:
        p, wt, _ = metrics(y, o, w, train[YEAR], te_prop); print(f"  {k}: {p:.4f} | {wt:.4f}")

    _, (wc, wl, wx) = tune_blend([oc, ol, ox], y, w)
    oof_bl = np.clip(wc*oc + wl*ol + wx*ox, 0, 100)
    test_bl = np.clip(wc*tc + wl*tl + wx*tx, 0, 100)
    bp, bw, bby = metrics(y, oof_bl, w, train[YEAR], te_prop)
    print(f"\n================ FAZ 9 (TUNED) SONUÇ ================")
    print(f"Blend cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f} | wOOF={bw:.4f} (düz {bp:.4f})")
    print(f"  Faz 7-B featB referans wOOF 86.0920 -> kazanç {86.0920 - bw:+.4f}")
    print("\nYıl-bazlı:"); print(bby.round(3).to_string())

    # ---------- Kayıt (tuned OOF/test -> downstream kullanır) ----------
    np.save(os.path.join(exp_dir, 'oof_tuned_cat.npy'), oc); np.save(os.path.join(exp_dir, 'test_tuned_cat.npy'), tc)
    np.save(os.path.join(exp_dir, 'oof_tuned_lgb.npy'), ol); np.save(os.path.join(exp_dir, 'test_tuned_lgb.npy'), tl)
    np.save(os.path.join(exp_dir, 'oof_tuned_xgb.npy'), ox); np.save(os.path.join(exp_dir, 'test_tuned_xgb.npy'), tx)
    np.save(os.path.join(exp_dir, 'oof_tuned_blend.npy'), oof_bl)
    log = {'phase': 'faz9_gbm_tuning', 'date': datetime.date.today().isoformat(),
           'n_trials': {'cat': CT, 'lgb': LT, 'xgb': XT},
           'best_params': {'cat': st_cat.best_params, 'lgb': st_lgb.best_params, 'xgb': st_xgb.best_params},
           'best_woof_tune': {'cat': st_cat.best_value, 'lgb': st_lgb.best_value, 'xgb': st_xgb.best_value},
           'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
           'blend': {'plain': bp, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'ref_faz7b_woof': 86.0920, 'gain_vs_faz7b': 86.0920 - bw}
    with open(os.path.join(exp_dir, 'faz9_tuning_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = f"sub_{datetime.date.today().isoformat()}_tuned_blend_woof{bw:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: test_bl}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof/test_tuned_{{cat,lgb,xgb}}.npy + oof_tuned_blend.npy + faz9_tuning_log.json + {fname}")


if __name__ == '__main__':
    main()
