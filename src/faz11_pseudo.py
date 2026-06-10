"""
Datathon 2026 — Faz 11: PSEUDO-LABELING (savaş planı #4, RİSKLİ, geç-yılı hedefler).

FİKİR: Ensemble'ın test'te EN KARARLI olduğu örnekler (3 GBM'in std'si düşük = yüksek uzlaşı)
güvenilir pseudo-label'dır. Bunları train'e kat, yeniden eğit. Özellikle 2025-2026 (train'de az,
test'te ağır) için değerli -> geç-yıl sinyalini güçlendirir.

LEAKAGE-GÜVENLİ TASARIM:
- Pseudo-label'lar tuned+bagged ensemble'ın TEST tahmininden (mevcut OOF pipeline çıktısı).
- Seçilen pseudo TEST satırları HER fold'un TRAIN'ine eklenir; HİÇBİR validation fold'una girmez.
- OOF YALNIZ gerçek train satırlarında hesaplanır (her zamanki gibi fold-dışı) -> train holdout dürüst.
- DİKKAT (kullanıcı uyarısı): pseudo-label kendi tahminini doğrulayıp OOF'u YAPAY iyileştirebilir.
  Bu yüzden kazanç OOF'ta görünse bile YARIN PUBLIC'te doğrulanacak; OOF'a tek başına güvenME.

Girdi: faz9 best_params + test_bag_{cat,lgb,xgb}.npy (std için). FRAC env (default 0.3 = en kararlı %30),
LATE_ONLY env ('1' ise yalnız 2025-2026 pseudo). Çıktı: pseudo-augmented OOF/test + log + submission.
"""
import os, json, datetime, time
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, metrics
from faz3_ensemble import tune_blend
from faz9_tuning import build_matrices, wmse
from faz10_bagging import cat_params, lgb_params, xgb_params
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor


def cv_with_pseudo(kind, params, Xtr, y, Xte, folds, pseudo_idx, pseudo_y, cat_arg):
    """Her fold: fold-train + TÜM pseudo-test satırları ile eğit; OOF yalnız gerçek-train va'da."""
    oof = np.zeros(len(y)); test_pred = np.zeros(Xte.shape[0])
    Xpa = Xte.iloc[pseudo_idx]
    for tr, va in folds:
        Xf = pd.concat([Xtr.iloc[tr], Xpa], axis=0)
        yf = np.concatenate([y[tr], pseudo_y])
        if kind == 'cat':
            m = CatBoostRegressor(**params)
            m.fit(Pool(Xf, yf, cat_features=cat_arg), eval_set=Pool(Xtr.iloc[va], y[va], cat_features=cat_arg),
                  use_best_model=True)
        elif kind == 'lgb':
            m = LGBMRegressor(**params)
            m.fit(Xf, yf, eval_set=[(Xtr.iloc[va], y[va])], categorical_feature=cat_arg,
                  callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        else:
            m = XGBRegressor(**params)
            m.fit(Xf, yf, eval_set=[(Xtr.iloc[va], y[va])], verbose=False)
        oof[va] = np.clip(m.predict(Xtr.iloc[va]), 0, 100)
        test_pred += np.clip(m.predict(Xte), 0, 100) / len(folds)
    return oof, test_pred


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)
    bp = json.load(open(os.path.join(exp_dir, 'faz9_tuning_log.json')))['best_params']
    Xc, Xc_t, Xg, Xg_t, cat_cols, cat_idx = build_matrices(train, test, y, folds, exp_dir)

    # --- Pseudo-label = bagged blend test tahmini; güven = 3 GBM std (düşük=uzlaşı) ---
    tc = np.load(os.path.join(exp_dir, 'test_bag_cat.npy')); tl = np.load(os.path.join(exp_dir, 'test_bag_lgb.npy'))
    tx = np.load(os.path.join(exp_dir, 'test_bag_xgb.npy'))
    blw = json.load(open(os.path.join(exp_dir, 'faz10_bagging_log.json')))['blend_weights']
    pseudo_all = np.clip(blw['cat']*tc + blw['lgb']*tl + blw['xgb']*tx, 0, 100)
    std = np.std(np.vstack([tc, tl, tx]), axis=0)        # model-arası anlaşmazlık

    FRAC = float(os.environ.get('FRAC', 0.3)); LATE_ONLY = os.environ.get('LATE_ONLY', '0') == '1'
    elig = np.ones(len(test), bool)
    if LATE_ONLY: elig = test[YEAR].isin([2025, 2026]).values
    order = np.argsort(std[elig]); cand = np.where(elig)[0][order]
    n_keep = int(len(cand) * FRAC)
    pseudo_idx = cand[:n_keep]; pseudo_y = pseudo_all[pseudo_idx]
    print(f"[pseudo] {n_keep}/{len(test)} test satırı (FRAC={FRAC}, LATE_ONLY={LATE_ONLY}), "
          f"std medyan seçilen={np.median(std[pseudo_idx]):.3f} vs tüm={np.median(std):.3f}")
    print(f"  pseudo yıl dağılımı: {test.iloc[pseudo_idx][YEAR].value_counts().to_dict()}")

    t0 = time.time()
    oc, ptc = cv_with_pseudo('cat', cat_params(bp['cat'], 42), Xc, y, Xc_t, folds, pseudo_idx, pseudo_y, cat_idx)
    ol, ptl = cv_with_pseudo('lgb', lgb_params(bp['lgb'], 42), Xg, y, Xg_t, folds, pseudo_idx, pseudo_y, cat_cols)
    ox, ptx = cv_with_pseudo('xgb', xgb_params(bp['xgb'], 42), Xg, y, Xg_t, folds, pseudo_idx, pseudo_y, None)
    print(f"[retrain w/ pseudo] {time.time()-t0:.0f}s | cat {wmse(y,oc,w):.3f} lgb {wmse(y,ol,w):.3f} xgb {wmse(y,ox,w):.3f}")

    _, (wc, wl, wx) = tune_blend([oc, ol, ox], y, w)
    oof_bl = np.clip(wc*oc + wl*ol + wx*ox, 0, 100); test_bl = np.clip(wc*ptc + wl*ptl + wx*ptx, 0, 100)
    bp_, bw, bby = metrics(y, oof_bl, w, train[YEAR], te_prop)
    ref = json.load(open(os.path.join(exp_dir, 'faz10_bagging_log.json')))['blend']['weighted']
    print(f"\n================ FAZ 11 (PSEUDO) SONUÇ ================")
    print(f"Blend cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f} | wOOF={bw:.4f} (düz {bp_:.4f})")
    print(f"  Faz 10 bagged wOOF {ref:.4f} -> pseudo Δ {ref - bw:+.4f}  (⚠️ OOF şişebilir, PUBLIC şart)")
    print("\nYıl-bazlı (2025-2026 kritik):"); print(bby.round(3).to_string())

    np.save(os.path.join(exp_dir, 'oof_pseudo_blend.npy'), oof_bl)
    log = {'phase': 'faz11_pseudo_labeling', 'date': datetime.date.today().isoformat(),
           'frac': FRAC, 'late_only': LATE_ONLY, 'n_pseudo': int(n_keep),
           'pseudo_year_dist': {int(k): int(v) for k, v in test.iloc[pseudo_idx][YEAR].value_counts().items()},
           'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
           'blend': {'plain': bp_, 'weighted': bw, 'by_year': bby.round(4).to_dict()},
           'ref_faz10_bagged': ref, 'gain_vs_faz10': ref - bw,
           'WARNING': 'pseudo OOF self-validation ile sisebilir; public dogrulamasi sart'}
    with open(os.path.join(exp_dir, 'faz11_pseudo_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    fname = f"sub_{datetime.date.today().isoformat()}_pseudo_woof{bw:.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: test_bl}).to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_pseudo_blend.npy + faz11_pseudo_log.json + {fname}")


if __name__ == '__main__':
    main()
