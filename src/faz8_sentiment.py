"""
Datathon 2026 — Faz 8: explicit sentiment feature etkisi (lokalde, hızlı).

Bağımsız formül-avının tek somut bulgusu: mentor metninde olumlu-eksi-olumsuz kelime
farkı (net) hedefle 0.383 korele. TF-IDF/embedding meta'sı bu YÖN bilgisini dolaylı
taşıyor ama explicit count GBM'e bedava kestirme verebilir.

Bu script winner'ı (Faz 7-B featB = FE+text_meta+emb_meta+berturk_meta üzerine 3-GBM blend)
sentiment'li ve sentiment'siz koşturup wOOF farkını ölçer. Karar: yalnız wOOF düşerse al.
Hiçbir OOF-tune yok; aynı fold/ağırlık/meta — tek değişken sentiment feature'ları.
"""
import os, json, datetime
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer


def run_featB(train, test, y, folds, w, te_prop, meta_map, sentiment):
    """featB ensemble (FE[+sentiment] + meta'lar üzerine cat/lgb/xgb blend) -> oof, test, wOOF."""
    train_fe, test_fe = engineer(train, sentiment=sentiment), engineer(test, sentiment=sentiment)
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols
    cols = num_cols + cat_cols

    Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
    for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[cols].copy(); Xg_t = test_fe[cols].copy()
    for c in cat_cols:
        cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
    for m, (a, b) in meta_map.items():
        Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
    feats = cols + list(meta_map)
    ci = [feats.index(c) for c in cat_cols]

    oc, tc, imp, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
    _, (a, b, c) = tune_blend([oc, ol, ox], y, w)
    oof = np.clip(a*oc + b*ol + c*ox, 0, 100); tst = np.clip(a*tc + b*tl + c*tx, 0, 100)
    p, wt, by = metrics(y, oof, w, train[YEAR], te_prop)
    return oof, tst, (a, b, c), p, wt, by, dict(zip(feats, imp))


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # Meta-feature'lar (sentiment'ten bağımsız, bir kez kur)
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    meta_map = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te), 'berturk_meta': (bt_tr, bt_te)}

    print("================ Faz 8: SENTIMENT etkisi (featB, tek değişken) ================")
    print("\n[A] sentiment'siz (Faz 7-B referans, ~86.09 beklenir)")
    _, te0, w0, p0, wt0, _, _ = run_featB(train, test, y, folds, w, te_prop, meta_map, sentiment=False)
    print(f"  wOOF={wt0:.4f} (düz {p0:.4f}) | blend {w0}")

    print("\n[B] sentiment'li (+net/pos/neg)")
    oof1, te1, w1, p1, wt1, by1, imp1 = run_featB(train, test, y, folds, w, te_prop, meta_map, sentiment=True)
    print(f"  wOOF={wt1:.4f} (düz {p1:.4f}) | blend {w1}")

    delta = wt0 - wt1
    print(f"\n>>> SENTIMENT etkisi: wOOF {wt0:.4f} -> {wt1:.4f}  ({delta:+.4f}, {'İYİLEŞME' if delta>0 else 'KÖTÜLEŞME'})")
    sent_imp = {k: round(v, 3) for k, v in imp1.items() if k.startswith('sentiment_')}
    print(f"  sentiment feature importance (CatBoost): {sent_imp}")

    log = {'phase': 'faz8_sentiment', 'date': datetime.date.today().isoformat(),
           'no_sentiment': {'plain': p0, 'weighted': wt0, 'blend_w': list(w0)},
           'with_sentiment': {'plain': p1, 'weighted': wt1, 'blend_w': list(w1),
                              'by_year': by1.round(4).to_dict(), 'sentiment_importance': sent_imp},
           'delta_weighted': delta}
    with open(os.path.join(exp_dir, 'faz8_sentiment_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    if delta > 0:
        fname = f"sub_{datetime.date.today().isoformat()}_featB_sentiment_woof{wt1:.2f}.csv"
        pd.DataFrame({ID: test[ID].values, TARGET: te1}).to_csv(os.path.join(sub_dir, fname), index=False)
        print(f"\n[kayıt] İyileşme -> {fname} + faz8_sentiment_log.json")
    else:
        print(f"\n[kayıt] İyileşme yok, submission yazılmadı (yalnız log). faz8_sentiment_log.json")


if __name__ == '__main__':
    main()
