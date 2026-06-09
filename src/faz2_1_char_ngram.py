"""
Datathon 2026 — Faz 2.1: kelime + char n-gram TF-IDF + Ridge alpha taraması -> CatBoost.

NEDEN char n-gram:
- Türkçe agglutinative (sondan eklemeli): "gelişim/gelişme/geliştirme/geliştirilmesi" aynı kökten
  türer. Kelime TF-IDF bunları AYRI token sayar; char_wb (3,5)-gram kök/ek örüntüsünü yakalar
  -> morfolojik sinyali sıkıştırır. Türkçe NLP'de kelime+char birleşimi sık kazandırır.
- char_wb (char değil): n-gram'lar kelime sınırı içinde kalır (boşlukla pad'lenir), gürültü az.

TASARIM:
- Faz 2.0 ile BİREBİR AYNI leakage-free fold'lar (yıl×hedef-desil, SEED=42).
- Her fold'da word(1,2) + char_wb(3,5) TF-IDF yalnız fold-train'de fit -> hstack.
- TF-IDF matrisleri fold başına BİR KEZ kurulur; Ridge alpha taraması bu matrisler üzerinde
  ucuzca döner (idf/transform pahalı, Ridge.fit ucuz).
- En iyi alpha AĞIRLIKLI OOF (public proxy) ile seçilir. Sonra tek CatBoost+text_meta koşusu.

ÇIKTILAR:
- experiments/oof_text_charword.npy
- experiments/oof_catboost_charword.npy
- experiments/faz2_1_charword_log.json
- submissions/sub_<tarih>_catboost_charword_oof<skor>.csv  (submit EDİLMEZ, aday)
"""
import os, json, datetime
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics, TR_STOPWORDS

WORD_PARAMS = dict(analyzer='word', ngram_range=(1, 2), min_df=3, max_df=0.9,
                   max_features=50000, sublinear_tf=True, lowercase=True)
CHAR_PARAMS = dict(analyzer='char_wb', ngram_range=(3, 5), min_df=3,
                   max_features=100000, sublinear_tf=True, lowercase=True)
ALPHA_GRID = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]


def build_text_meta_sweep(text_tr, text_te, y, folds, w):
    """
    word+char TF-IDF; her fold'da matrisi bir kez kur, tüm alpha'lar için Ridge OOF üret.
    Dönen: dict[alpha] -> (oof, test), ve alpha-bazlı (düz, ağırlıklı) metrikler.
    """
    n_tr, n_te = len(text_tr), len(text_te)
    oof = {a: np.zeros(n_tr) for a in ALPHA_GRID}
    test = {a: np.zeros(n_te) for a in ALPHA_GRID}
    for tr, va in folds:
        wv = TfidfVectorizer(stop_words=TR_STOPWORDS, **WORD_PARAMS)
        cv = TfidfVectorizer(**CHAR_PARAMS)   # char'da stopword anlamsız
        Xtr = hstack([wv.fit_transform(text_tr.iloc[tr]), cv.fit_transform(text_tr.iloc[tr])]).tocsr()
        Xva = hstack([wv.transform(text_tr.iloc[va]), cv.transform(text_tr.iloc[va])]).tocsr()
        Xte = hstack([wv.transform(text_te), cv.transform(text_te)]).tocsr()
        for a in ALPHA_GRID:
            r = Ridge(alpha=a, random_state=SEED).fit(Xtr, y[tr])
            oof[a][va] = r.predict(Xva)
            test[a] += r.predict(Xte) / len(folds)
    rows = []
    for a in ALPHA_GRID:
        p = float(np.mean((y - np.clip(oof[a], 0, 100)) ** 2))
        wt = float(np.sum(w * (y - np.clip(oof[a], 0, 100)) ** 2) / np.sum(w))
        rows.append((a, p, wt))
    sweep = pd.DataFrame(rows, columns=['alpha', 'plain', 'weighted']).set_index('alpha')
    return oof, test, sweep


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv')
    test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    folds, _ = make_folds(train)

    tr_prop = train[YEAR].value_counts(normalize=True)
    te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # --- 1) word+char text-only + alpha taraması ---
    oof_d, test_d, sweep = build_text_meta_sweep(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds, w)
    best_alpha = float(sweep['weighted'].idxmin())   # public proxy ile seç
    oof_text = np.clip(oof_d[best_alpha], 0, 100)
    test_text = np.clip(test_d[best_alpha], 0, 100)
    t_plain, t_weighted, t_by_year = metrics(y, oof_text, w, train[YEAR], te_prop)

    print("[alpha taraması] (word(1,2)+char_wb(3,5) text-only OOF):")
    print(sweep.round(4).to_string())
    print(f"En iyi alpha (ağırlıklı OOF): {best_alpha}")
    print(f"[text-only best] düz={t_plain:.4f} ağırlıklı={t_weighted:.4f}")
    print("  yıl-bazlı:\n", t_by_year.round(3).to_string())

    # --- 2) CatBoost + text_meta ---
    features = num_cols + cat_cols + ['text_meta']
    X = train[num_cols + cat_cols].copy(); X_test = test[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)
    X['text_meta'] = oof_text; X_test['text_meta'] = test_text
    cat_idx = [features.index(c) for c in cat_cols]
    oof, test_pred, importances, best_iters = run_catboost_cv(X, y, X_test, cat_idx, folds)
    c_plain, c_weighted, c_by_year = metrics(y, oof, w, train[YEAR], te_prop)

    # --- 3) Delta: Faz 1 (metinsiz) ve Faz 2.0 (word-only) referansları ---
    f1 = json.load(open(os.path.join(exp_dir, 'faz1_catboost_log.json')))
    f2 = json.load(open(os.path.join(exp_dir, 'faz2_text_meta_log.json')))
    f1_w, f2_w = f1['test_year_weighted_oof_mse'], f2['catboost_text']['weighted_oof_mse']
    f2_by_year = {int(k): v for k, v in f2['catboost_text']['by_year'].items()}
    delta_vs_f2 = {yr: round(f2_by_year.get(yr, np.nan) - c_by_year.get(yr, np.nan), 4)
                   for yr in sorted(c_by_year.index)}

    print("\n================ FAZ 2.1 SONUÇ (ağırlıklı OOF) ================")
    print(f"Faz 1 (metinsiz)        : {f1_w:.4f}")
    print(f"Faz 2.0 (word TF-IDF)   : {f2_w:.4f}")
    print(f"Faz 2.1 (word+char)     : {c_weighted:.4f}")
    print(f"  kazanç vs Faz 2.0     : {f2_w - c_weighted:+.4f}")
    print(f"  kazanç vs Faz 1       : {f1_w - c_weighted:+.4f}")
    print(f"  (düz OOF: {c_plain:.4f})")
    print("\nYıl-bazlı kazanç vs Faz 2.0 (pozitif=char ek iyileşme):")
    for yr, d in delta_vs_f2.items():
        flag = "  <- ağır test payı" if yr in (2025, 2026) else ""
        print(f"  {yr}: {d:+.4f}{flag}")

    # --- Kayıtlar ---
    np.save(os.path.join(exp_dir, 'oof_text_charword.npy'), oof_text)
    np.save(os.path.join(exp_dir, 'oof_catboost_charword.npy'), oof)
    log = {
        'phase': 'faz2_1_word_plus_char_ngram', 'date': datetime.date.today().isoformat(),
        'best_alpha': best_alpha, 'alpha_sweep': sweep.round(4).to_dict(),
        'text_only': {'plain': t_plain, 'weighted': t_weighted, 'by_year': t_by_year.round(4).to_dict()},
        'catboost_text': {'plain': c_plain, 'weighted': c_weighted,
                          'by_year': c_by_year.round(4).to_dict(), 'best_iters': best_iters},
        'ref': {'faz1_weighted': f1_w, 'faz2_0_weighted': f2_w},
        'gain': {'vs_faz2_0': f2_w - c_weighted, 'vs_faz1': f1_w - c_weighted, 'by_year_vs_faz2_0': delta_vs_f2},
        'text_meta_importance': float(pd.Series(importances, index=features)['text_meta']),
    }
    with open(os.path.join(exp_dir, 'faz2_1_charword_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_pred})
    fname = f"sub_{datetime.date.today().isoformat()}_catboost_charword_oof{c_plain:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_text_charword.npy | oof_catboost_charword.npy | faz2_1_charword_log.json | {fname}")


if __name__ == '__main__':
    main()
