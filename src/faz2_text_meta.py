"""
Datathon 2026 — Faz 2 İSKELET: metin meta-feature (TF-IDF (1,2)-gram + Ridge OOF) -> CatBoost.

⚠️ HENÜZ ÇALIŞTIRMA. Önce Faz 1 submission'ı ile OOF↔public köprüsü kalibre edilecek.
   (Köprü doğrulanınca bu Faz 2 lokalde, submission harcamadan ilerletilecek.)

TASARIM (neden böyle):
- METNİ DOĞRUDAN CatBoost'a vermek yerine, ucuz ve güçlü bir aracı kullanıyoruz:
  TF-IDF (1,2)-gram -> Ridge regresyon -> tek bir "text_meta" tahmini. Bu, metindeki
  doğrusal sinyali tek kolona sıkıştırır; CatBoost onu sayısal feature gibi kullanır.
- LEAKAGE-FREE: text_meta, Faz 1 ile BİREBİR AYNI fold'larda OOF olarak üretilir
  (aynı SEED, aynı yıl×hedef-desil stratify). Her fold'da TF-IDF idf + Ridge yalnız
  fold-train metninde öğrenilir; validation ve test fold-dışı tahmin alır.
- APPLES-TO-APPLES: CatBoost params Faz 1 ile birebir aynı; tek fark eklenen text_meta
  kolonu. Böylece OOF farkı = metnin SAF marjinal katkısı.
- ÖLÇÜM: metnin katkısını GENEL + YIL-BAZLI raporla; özellikle 2025-2026 (en zor +
  en ağır test payı) ayrı. Asıl kazanç orada olmalı.

ÇIKTILAR (çalıştırılınca):
- experiments/oof_text_ridge.npy        : metin-only Ridge OOF (meta-feature)
- experiments/oof_catboost_text.npy      : CatBoost+metin OOF
- experiments/faz2_text_meta_log.json    : metin-only + birleşik metrikler, yıl-bazlı delta
- submissions/sub_<tarih>_catboost_text_oof<skor>.csv
"""
import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool

# Faz 1 ile ortak sabit ve yardımcılar (aynı sonuç/fold için kritik)
from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, N_SPLITS, ID, TARGET, TEXT, YEAR

# --- TF-IDF / Ridge ayarları (Faz 2.1: ucuz doğrusal metin sinyali) ---
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),        # kullanıcı isteği: 1- ve 2-gram
    min_df=3, max_df=0.9,      # çok nadir / çok sık terimleri ele
    max_features=50000,
    sublinear_tf=True,         # tf -> 1+log(tf), uzun metinlerde daha stabil
    lowercase=True,
    strip_accents=None,        # Türkçe karakterleri KORU (ç/ğ/ı/ş/ö/ü anlam taşır)
)
RIDGE_ALPHA = 3.0              # köprü sonrası küçük bir alpha taraması yapılabilir

# Küçük TR stopword listesi (TF-IDF gürültüsünü azaltır; min_df zaten çoğunu eler).
TR_STOPWORDS = [
    've', 'ile', 'bir', 'bu', 'da', 'de', 'için', 'çok', 'daha', 'gibi', 'ama',
    'ancak', 'olarak', 'ya', 'ise', 'en', 'her', 'o', 'ki', 'mi', 'ne', 'fazla',
    'olan', 'üzerinde', 'konusunda', 'birlikte', 'sağlayabilir', 'olabilir',
]

# CatBoost params — Faz 1 ile BİREBİR AYNI (apples-to-apples marjinal ölçüm için)
CATBOOST_PARAMS = dict(
    loss_function='RMSE', eval_metric='RMSE',
    iterations=3000, learning_rate=0.03, depth=6,
    l2_leaf_reg=3.0, random_seed=SEED,
    od_type='Iter', od_wait=200,
    verbose=False, allow_writing_files=False,
)


def make_folds(train):
    """Faz 1 ile BİREBİR AYNI fold'ları üret (yıl × hedef-desil stratify, aynı SEED)."""
    y = train[TARGET].values
    tbin = pd.qcut(y, 10, labels=False, duplicates='drop')
    strat = train[YEAR].astype(str) + '_' + pd.Series(tbin).astype(str)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(train, strat)), strat


def build_text_meta(text_tr, text_te, y, folds):
    """
    TF-IDF (1,2)-gram + Ridge ile LEAKAGE-FREE metin meta-feature üret.
    Dönen: oof_text (train, fold-dışı tahmin), test_text (fold ortalaması).
    Her fold: vectorizer+Ridge yalnız fold-train metninde fit edilir.
    """
    oof_text = np.zeros(len(text_tr))
    test_text = np.zeros(len(text_te))
    for tr, va in folds:
        vec = TfidfVectorizer(stop_words=TR_STOPWORDS, **TFIDF_PARAMS)
        Xtr = vec.fit_transform(text_tr.iloc[tr])
        Xva = vec.transform(text_tr.iloc[va])
        Xte = vec.transform(text_te)
        ridge = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
        ridge.fit(Xtr, y[tr])
        oof_text[va] = ridge.predict(Xva)
        test_text += ridge.predict(Xte) / len(folds)
    return oof_text, test_text


def run_catboost_cv(X, y, X_test, cat_idx, folds):
    """Faz 1 ile aynı CatBoost CV; OOF + test tahmini + ortalama importance döner."""
    oof = np.zeros(len(X)); test_pred = np.zeros(len(X_test))
    importances = np.zeros(X.shape[1]); best_iters = []
    for tr, va in folds:
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(Pool(X.iloc[tr], y[tr], cat_features=cat_idx),
              eval_set=Pool(X.iloc[va], y[va], cat_features=cat_idx),
              use_best_model=True)
        oof[va] = m.predict(X.iloc[va])
        test_pred += m.predict(X_test) / len(folds)
        importances += m.get_feature_importance() / len(folds)
        best_iters.append(int(m.get_best_iteration()))
    return np.clip(oof, 0, 100), np.clip(test_pred, 0, 100), importances, best_iters


def metrics(y, oof, w, years, te_prop):
    """Düz MSE, test-yıl-ağırlıklı MSE, yıl-bazlı MSE kırılımı (Faz 1 ile aynı formül)."""
    plain = float(np.mean((y - oof) ** 2))
    weighted = float(np.sum(w * (y - oof) ** 2) / np.sum(w))
    by_year = (pd.DataFrame({'year': years, 'err2': (y - oof) ** 2})
               .groupby('year')['err2'].mean())
    return plain, weighted, by_year


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values

    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]

    folds, _ = make_folds(train)

    # Test-yıl-ağırlıkları (Faz 1 ile aynı: public proxy)
    tr_prop = train[YEAR].value_counts(normalize=True)
    te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # --- 1) Metin-only meta-feature (TF-IDF + Ridge OOF) ---
    oof_text, test_text = build_text_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    t_plain, t_weighted, t_by_year = metrics(y, np.clip(oof_text, 0, 100), w, train[YEAR], te_prop)
    print("[metin-only Ridge]  düz OOF MSE=%.4f | ağırlıklı=%.4f" % (t_plain, t_weighted))
    print("  yıl-bazlı:\n", t_by_year.round(3).to_string())

    # --- 2) CatBoost + text_meta (Faz 1 feature seti + tek metin kolonu) ---
    features = num_cols + cat_cols + ['text_meta']
    X = train[num_cols + cat_cols].copy(); X_test = test[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)
    X['text_meta'] = oof_text; X_test['text_meta'] = test_text   # OOF train / fold-ort test
    cat_idx = [features.index(c) for c in cat_cols]

    oof, test_pred, importances, best_iters = run_catboost_cv(X, y, X_test, cat_idx, folds)
    c_plain, c_weighted, c_by_year = metrics(y, oof, w, train[YEAR], te_prop)

    # --- 3) Marjinal katkı vs Faz 1 (yıl-bazlı, özellikle 2025-2026) ---
    base = json.load(open(os.path.join(exp_dir, 'faz1_catboost_log.json')))
    base_plain = base['plain_oof_mse']; base_weighted = base['test_year_weighted_oof_mse']
    base_by_year = {int(k): v for k, v in base['by_year_oof_mse'].items()}
    delta_by_year = {yr: round(base_by_year.get(yr, np.nan) - c_by_year.get(yr, np.nan), 4)
                     for yr in sorted(c_by_year.index)}

    print("\n================ FAZ 2 SONUÇ ================")
    print("                         düz OOF      ağırlıklı OOF")
    print("Faz 1 (metinsiz)      : %9.4f   %9.4f" % (base_plain, base_weighted))
    print("Faz 2 (+metin)        : %9.4f   %9.4f" % (c_plain, c_weighted))
    print("Marjinal kazanç (Δ↓)  : %9.4f   %9.4f" % (base_plain - c_plain, base_weighted - c_weighted))
    print("\nYıl-bazlı marjinal kazanç (Faz1 - Faz2, pozitif=iyileşme):")
    for yr, d in delta_by_year.items():
        flag = "  <- en zor+ağır test payı" if yr in (2025, 2026) else ""
        print("  %d: %+.4f%s" % (yr, d, flag))

    # --- Kayıtlar ---
    np.save(os.path.join(exp_dir, 'oof_text_ridge.npy'), np.clip(oof_text, 0, 100))
    np.save(os.path.join(exp_dir, 'oof_catboost_text.npy'), oof)
    log = {
        'phase': 'faz2_text_meta_tfidf_ridge',
        'date': datetime.date.today().isoformat(), 'seed': SEED,
        'text_only': {'plain_oof_mse': t_plain, 'weighted_oof_mse': t_weighted,
                      'by_year': t_by_year.round(4).to_dict()},
        'catboost_text': {'plain_oof_mse': c_plain, 'weighted_oof_mse': c_weighted,
                          'by_year': c_by_year.round(4).to_dict(), 'best_iters': best_iters},
        'baseline_faz1': {'plain': base_plain, 'weighted': base_weighted},
        'marginal_gain': {'plain': base_plain - c_plain, 'weighted': base_weighted - c_weighted,
                          'by_year': delta_by_year},
        'tfidf_params': {k: str(v) for k, v in TFIDF_PARAMS.items()}, 'ridge_alpha': RIDGE_ALPHA,
        'text_meta_importance': float(pd.Series(importances, index=features)['text_meta']),
    }
    with open(os.path.join(exp_dir, 'faz2_text_meta_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_pred})
    fname = f"sub_{datetime.date.today().isoformat()}_catboost_text_oof{c_plain:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_text_ridge.npy | oof_catboost_text.npy | faz2_text_meta_log.json | {fname}")


if __name__ == '__main__':
    # ⚠️ Köprü kalibrasyonu beklenirken çalıştırma. Hazır olunca: python3 src/faz2_text_meta.py
    main()
