"""
Datathon 2026 — Faz 1: Saf sayısal+kategorik CatBoost baseline (temporal-aware CV).

NEDEN bu tasarım:
- Faz 0'da train/test ayrımının ZAMANSAL olduğunu tespit ettik: test geç yıllara
  (2024-2026 ≈ %62) ağırlıklı, geç kohort daha düşük mean + daha yüksek varyans.
  Bu yüzden düz random K-fold OOF MSE public LB'den sistematik olarak İYİMSER çıkar.
- Çözüm: (a) fold'ları (yıl × hedef-bin) ile stratify et -> her fold yıl ve hedef
  dağılımı bakımından temsili; (b) public'i izleyen asıl metrik olarak test-yıl-ağırlıklı
  OOF MSE raporla (importance weighting: her train satırını test/train yıl-oran'ına göre ağırlıkla).
- METİN YOK: bu kasıtlı. Saf sayısal+kategorik taban, sonra metnin marjinal katkısını
  temiz ölçmek için referans. (mentor_feedback_text Faz 2'de eklenecek.)

CatBoost neden: kategorikleri native işler (encoding yok), sayısal NaN'ı native yönetir,
MSE için RMSE loss doğrudan optimize eder.

Çıktılar:
- experiments/oof_catboost.npy        : OOF tahminleri (train sırasıyla)
- experiments/faz1_catboost_log.json  : 3 metrik + yıl kırılımı + feature importance
- submissions/sub_<tarih>_catboost_oof<skor>.csv : test tahmini (clip 0-100)
"""
import os, glob, json, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostRegressor, Pool

SEED = 42
N_SPLITS = 5
ID, TARGET, TEXT = 'student_id', 'career_success_score', 'mentor_feedback_text'
YEAR = 'application_year'   # temporal kayma sürücüsü; ağırlık & stratify burada


def resolve_paths():
    """Kaggle (/kaggle/input) veya lokal (/tmp/data) otomatik bul; çıktı dizinlerini ayarla."""
    cands = glob.glob('/kaggle/input/*')
    if cands and os.path.exists(f'{cands[0]}/train.csv'):
        data_dir = cands[0]
    elif os.path.exists('/tmp/data/train.csv'):
        data_dir = '/tmp/data'
    else:
        raise FileNotFoundError("train.csv bulunamadı (/kaggle/input/* veya /tmp/data)")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(repo, 'experiments')
    sub_dir = '/kaggle/working' if os.path.isdir('/kaggle/working') else os.path.join(repo, 'submissions')
    os.makedirs(exp_dir, exist_ok=True); os.makedirs(sub_dir, exist_ok=True)
    return data_dir, exp_dir, sub_dir


def is_str_col(s):
    # pandas 2.x metni 'object', 3.x 'str' (StringDtype) işaretler.
    return (s.dtype == 'object' or pd.api.types.is_string_dtype(s)) and not pd.api.types.is_numeric_dtype(s)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    print(f"[data] {data_dir}")
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test_x.csv')

    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    features = num_cols + cat_cols                      # METİN YOK (kasıtlı)
    print(f"[feat] {len(num_cols)} sayısal + {len(cat_cols)} kategorik = {len(features)} (metin hariç)")

    X = train[features].copy()
    y = train[TARGET].values
    X_test = test[features].copy()
    # CatBoost kategorikleri string ister + NaN olmamalı (burada kategoriklerde NaN yok).
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    cat_idx = [features.index(c) for c in cat_cols]

    # --- Temporal-aware stratify anahtarı: yıl × hedef-desil ---
    # Hem yıl (kayma ekseni) hem hedef dağılımı her fold'da temsili olsun.
    tbin = pd.qcut(y, 10, labels=False, duplicates='drop')
    strat = train[YEAR].astype(str) + '_' + pd.Series(tbin).astype(str)
    vc = strat.value_counts()
    print(f"[cv] {strat.nunique()} stratum, en küçük={vc.min()} (>= {N_SPLITS} olmalı)")

    # --- Test-yıl-ağırlıkları (public'i izleyen metrik için importance weighting) ---
    tr_prop = train[YEAR].value_counts(normalize=True)
    te_prop = test[YEAR].value_counts(normalize=True)
    w = train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values
    w = np.nan_to_num(w, nan=0.0)

    # --- CV ---
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances = np.zeros(len(features))
    fold_iters = []

    params = dict(
        loss_function='RMSE', eval_metric='RMSE',   # RMSE optimize = MSE optimize
        iterations=3000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, random_seed=SEED,
        od_type='Iter', od_wait=200,                 # early stopping
        verbose=False, allow_writing_files=False,
    )

    for fold, (tr, va) in enumerate(skf.split(X, strat)):
        model = CatBoostRegressor(**params)
        model.fit(
            Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
            eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx),
            use_best_model=True,
        )
        oof[va] = model.predict(X.iloc[va])
        test_pred += model.predict(X_test) / N_SPLITS
        importances += model.get_feature_importance() / N_SPLITS
        fold_iters.append(int(model.get_best_iteration()))
        fold_mse = float(np.mean((y[va] - oof[va]) ** 2))
        print(f"  fold {fold}: best_iter={fold_iters[-1]:4d}  OOF MSE={fold_mse:.4f}")

    # Clip (CONTEXT zorunlu adım: 0-100 dışı MSE'yi kötüleştirir)
    oof = np.clip(oof, 0, 100)
    test_pred = np.clip(test_pred, 0, 100)

    # --- 3 METRİK ---
    plain_mse = float(np.mean((y - oof) ** 2))
    weighted_mse = float(np.sum(w * (y - oof) ** 2) / np.sum(w))   # test-yıl-ağırlıklı (public proxy)

    by_year = (pd.DataFrame({'year': train[YEAR], 'err2': (y - oof) ** 2})
               .groupby('year')['err2'].agg(['mean', 'count'])
               .assign(test_share=lambda d: d.index.map(lambda yr: float(te_prop.get(yr, 0.0))))
               .round(4))

    print("\n================ FAZ 1 SONUÇ ================")
    print(f"1) Düz OOF MSE              : {plain_mse:.4f}")
    print(f"2) Test-yıl-ağırlıklı OOF MSE: {weighted_mse:.4f}   <- public LB'yi izleyen sayı")
    print(f"   (Faz 0 sabit baseline: düz=230.6, public=274.7)")
    print(f"   ort. best_iter: {int(np.mean(fold_iters))}")
    print("\n3) Yıl-bazlı OOF MSE kırılımı:")
    print(by_year.to_string())

    imp = (pd.Series(importances, index=features).sort_values(ascending=False))
    print("\nFeature importance (top 20):")
    print(imp.head(20).round(3).to_string())

    # --- Kayıtlar ---
    np.save(os.path.join(exp_dir, 'oof_catboost.npy'), oof)
    log = {
        'phase': 'faz1_catboost_baseline_no_text',
        'date': datetime.date.today().isoformat(),
        'seed': SEED, 'n_splits': N_SPLITS,
        'n_features': len(features), 'num_cols': num_cols, 'cat_cols': cat_cols,
        'plain_oof_mse': plain_mse,
        'test_year_weighted_oof_mse': weighted_mse,
        'fold_best_iters': fold_iters,
        'by_year_oof_mse': by_year['mean'].to_dict(),
        'by_year_count': by_year['count'].astype(int).to_dict(),
        'feature_importance': imp.round(4).to_dict(),
        'baseline_phase0': {'plain_var': 230.61, 'public_mse': 274.72},
        'catboost_params': {k: v for k, v in params.items() if k != 'verbose'},
    }
    with open(os.path.join(exp_dir, 'faz1_catboost_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_pred})
    assert len(sub) == len(test)
    fname = f"sub_{datetime.date.today().isoformat()}_catboost_oof{plain_mse:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_catboost.npy | faz1_catboost_log.json | {fname}")


if __name__ == '__main__':
    main()
