"""
Datathon 2026 — Faz 2.2: multilingual-e5-base embedding meta-feature -> CatBoost.

⚠️ BU SCRIPT KAGGLE'DA ÇALIŞTIRILIR (GPU + model). Bu container HF'yi blokluyor (allowlist
   proxy: PyPI/GitHub açık, huggingface.co kapalı) -> e5 ağırlıkları lokalde indirilemez.

MODEL YÜKLEME (iki senaryo):
- Internet AÇIK Kaggle notebook : hiçbir şey gerekmez, HF'den iner.
- Internet KAPALI               : modeli Kaggle Dataset olarak önceden yükle, sonra
                                   E5_PATH=/kaggle/input/<dataset>/multilingual-e5-base ver.

TASARIM (kullanıcı kuralları):
- e5-base (large DEĞİL — marjinal kazanç için ağır). Beklenti +0.5..+1.5 ağırlıklı OOF.
- Embedding'i TF-IDF'in YERİNE değil, AYRI meta-feature olarak ekle:
    final feature seti = sayısal + kategorik + text_meta(word+char) + emb_meta
- NET-BIRAK KURALI: ağırlıklı OOF kazancı (vs Faz 2.1 = 87.83) < +0.5 ise embedding BIRAKILIR.
  Script sonunda KEEP/DROP verdict'i basar.
- Leakage yok: embedding'ler ön-eğitimli/donuk model ile ETİKETSİZ üretilir (tüm veride bir kez
  encode). Meta-feature'a çevirmek için Ridge-OOF aynı leakage-free fold'larda eğitilir.
- Yıl-bazlı kırılım, ÖZELLİKLE 2026: embedding mantığı az-örnekli kohortta transfer learning;
  kazancı orada bekliyoruz.

ÇIKTILAR:
- experiments/emb_e5_train.npy / emb_e5_test.npy  (cache; yeniden encode etmemek için)
- experiments/oof_emb_ridge.npy, oof_catboost_emb.npy
- experiments/faz2_2_embed_log.json
- submissions/sub_<tarih>_catboost_emb_oof<skor>.csv  (aday; karar netleşince submit)
"""
import os, json, datetime
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics, TR_STOPWORDS
from faz2_1_char_ngram import WORD_PARAMS, CHAR_PARAMS

E5_MODEL = os.environ.get('E5_PATH', 'intfloat/multilingual-e5-base')
CHARWORD_ALPHA = 5.0    # Faz 2.1'de seçilen en iyi alpha
EMB_RIDGE_ALPHA = 8.0   # 768-boyut yoğun embedding için biraz daha yüksek regülarizasyon
KEEP_THRESHOLD = 0.5    # net-bırak kuralı (ağırlıklı OOF kazancı)


def build_charword_meta(text_tr, text_te, y, folds, alpha=CHARWORD_ALPHA):
    """Faz 2.1 ile aynı word+char text_meta (sabit alpha), leakage-free OOF + test."""
    oof = np.zeros(len(text_tr)); test = np.zeros(len(text_te))
    for tr, va in folds:
        wv = TfidfVectorizer(stop_words=TR_STOPWORDS, **WORD_PARAMS)
        cv = TfidfVectorizer(**CHAR_PARAMS)
        Xtr = hstack([wv.fit_transform(text_tr.iloc[tr]), cv.fit_transform(text_tr.iloc[tr])]).tocsr()
        Xva = hstack([wv.transform(text_tr.iloc[va]), cv.transform(text_tr.iloc[va])]).tocsr()
        Xte = hstack([wv.transform(text_te), cv.transform(text_te)]).tocsr()
        r = Ridge(alpha=alpha, random_state=SEED).fit(Xtr, y[tr])
        oof[va] = r.predict(Xva); test += r.predict(Xte) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def encode_e5(texts, exp_dir, tag):
    """e5-base ile encode (cache'li). e5 'query: ' prefiksi ister; normalize edilmiş döner."""
    cache = os.path.join(exp_dir, f'emb_e5_{tag}.npy')
    if os.path.exists(cache):
        print(f"[emb] cache: {cache}")
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    print(f"[emb] model yükleniyor: {E5_MODEL}")
    model = SentenceTransformer(E5_MODEL)
    emb = model.encode(['query: ' + t for t in texts], batch_size=64,
                       show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


def build_emb_meta(emb_tr, emb_te, y, folds, alpha=EMB_RIDGE_ALPHA):
    """768-boyut embedding'i tek 'emb_meta' kolonuna indir (Ridge-OOF, leakage-free)."""
    oof = np.zeros(len(emb_tr)); test = np.zeros(len(emb_te))
    for tr, va in folds:
        r = Ridge(alpha=alpha, random_state=SEED).fit(emb_tr[tr], y[tr])
        oof[va] = r.predict(emb_tr[va]); test += r.predict(emb_te) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    text_tr, text_te = train[TEXT].fillna(''), test[TEXT].fillna('')

    # --- word+char text_meta (Faz 2.1 ile aynı) ---
    tm_tr, tm_te = build_charword_meta(text_tr, text_te, y, folds)

    # --- e5 embedding -> emb_meta ---
    emb_tr = encode_e5(text_tr.tolist(), exp_dir, 'train')
    emb_te = encode_e5(text_te.tolist(), exp_dir, 'test')
    em_tr, em_te = build_emb_meta(emb_tr, emb_te, y, folds)
    e_plain, e_weighted, e_by_year = metrics(y, em_tr, w, train[YEAR], te_prop)
    print(f"[embedding-only Ridge] düz={e_plain:.4f} ağırlıklı={e_weighted:.4f}")
    print("  yıl-bazlı:\n", e_by_year.round(3).to_string())

    # --- CatBoost: sayısal+kategorik + text_meta(word+char) + emb_meta ---
    features = num_cols + cat_cols + ['text_meta', 'emb_meta']
    X = train[num_cols + cat_cols].copy(); X_test = test[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str); X_test[c] = X_test[c].astype(str)
    X['text_meta'] = tm_tr; X_test['text_meta'] = tm_te
    X['emb_meta'] = em_tr;  X_test['emb_meta'] = em_te
    cat_idx = [features.index(c) for c in cat_cols]
    oof, test_pred, importances, best_iters = run_catboost_cv(X, y, X_test, cat_idx, folds)
    c_plain, c_weighted, c_by_year = metrics(y, oof, w, train[YEAR], te_prop)

    # --- Referanslar + yıl-bazlı delta (özellikle 2026) ---
    f21 = json.load(open(os.path.join(exp_dir, 'faz2_1_charword_log.json')))
    ref_w = f21['catboost_text']['weighted']               # 87.83
    ref_by_year = {int(k): v for k, v in f21['catboost_text']['by_year'].items()}
    gain = ref_w - c_weighted
    delta_by_year = {yr: round(ref_by_year.get(yr, np.nan) - c_by_year.get(yr, np.nan), 4)
                     for yr in sorted(c_by_year.index)}
    verdict = 'KEEP' if gain >= KEEP_THRESHOLD else 'DROP'

    print("\n================ FAZ 2.2 SONUÇ (ağırlıklı OOF) ================")
    print(f"Faz 2.1 (word+char)        : {ref_w:.4f}")
    print(f"Faz 2.2 (+e5 emb_meta)     : {c_weighted:.4f}")
    print(f"  kazanç vs Faz 2.1        : {gain:+.4f}  (eşik {KEEP_THRESHOLD})")
    print(f"  emb_meta importance      : {float(pd.Series(importances, index=features)['emb_meta']):.3f}")
    print(f"\n>>> NET-BIRAK VERDICT: {verdict} "
          f"({'embedding tutulur' if verdict=='KEEP' else 'embedding bırakılır, TF-IDF+char ile devam'})")
    print("\nYıl-bazlı kazanç vs Faz 2.1 (pozitif=embedding ek iyileşme):")
    for yr, d in delta_by_year.items():
        flag = "  <- 2026: transfer learning beklentisi" if yr == 2026 else (
               "  <- ağır test payı" if yr == 2025 else "")
        print(f"  {yr}: {d:+.4f}{flag}")

    # --- Kayıtlar ---
    np.save(os.path.join(exp_dir, 'oof_emb_ridge.npy'), em_tr)
    np.save(os.path.join(exp_dir, 'oof_catboost_emb.npy'), oof)
    log = {
        'phase': 'faz2_2_e5_embedding', 'date': datetime.date.today().isoformat(),
        'model': E5_MODEL, 'emb_ridge_alpha': EMB_RIDGE_ALPHA,
        'embedding_only': {'plain': e_plain, 'weighted': e_weighted, 'by_year': e_by_year.round(4).to_dict()},
        'catboost_full': {'plain': c_plain, 'weighted': c_weighted,
                          'by_year': c_by_year.round(4).to_dict(), 'best_iters': best_iters},
        'ref_faz2_1_weighted': ref_w, 'gain_vs_faz2_1': gain,
        'keep_threshold': KEEP_THRESHOLD, 'verdict': verdict,
        'gain_by_year_vs_faz2_1': delta_by_year,
        'emb_meta_importance': float(pd.Series(importances, index=features)['emb_meta']),
    }
    with open(os.path.join(exp_dir, 'faz2_2_embed_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_pred})
    fname = f"sub_{datetime.date.today().isoformat()}_catboost_emb_oof{c_plain:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] emb cache + oof_emb_ridge.npy + oof_catboost_emb.npy + log + {fname}")


if __name__ == '__main__':
    main()
