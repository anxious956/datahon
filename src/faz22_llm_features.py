"""
Datathon 2026 — Faz 22: LLM yapılandırılmış feature'ları GBM tabanına ekle.

GEREKÇE: Qwen2.5-7B-Instruct her mentor metninden ayrık semantik feature çıkardı
(ton/gelisim/somut_basari + 28 beceri bayrağı). Tekil sinyal güçlü: ton corr_y=0.49.
SORU: berturk/text metaları zaten metni okuyor -> bu ayrık form EK katkı mı, yoksa redundant mı?

KANITLI MEKANİZMA: yeni sinyal GBM FEATURE'ı olarak girer (berturk +2.22, bert128k/electra +0.37).
LLM feat'leri HAM feature (OOF değil) -> doğrudan num_cols'a eklenir.

KARŞILAŞTIRMA: aynı 5-meta base'i (text+emb+berturk+bert128k+electra) iki kez kur:
  (A) LLM'siz  -> faz13 referans (wOOF ~85.85)
  (B) LLM'li   -> delta. wOOF düşüşü BÜYÜKSE (>0.3) sinyal gerçek; flat ise redundant.
Anti-overfit: DEFAULT GBM paramları, SABİT blend [0.7/0.2/0.1]. KARAR public ile.

Çıktı: oof_faz22_blend.npy + test_faz22_blend.npy (+llm) ve submission.
"""
import os, json, datetime
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

TRANSFORMER_TAGS = ['bert128k', 'electra']
FIXED_BLEND = (0.7, 0.2, 0.1)


def build_gbm_oof(train_fe, test_fe, num_cols, cat_cols, meta, y, folds):
    """faz13 ile aynı: 3 GBM (default), sabit blend. num_cols LLM'li/llmsiz değişir."""
    cols = num_cols + cat_cols
    Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
    for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[cols].copy(); Xg_t = test_fe[cols].copy()
    for c in cat_cols:
        cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
    for m, (a, b) in meta.items():
        Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
    feats = cols + list(meta); ci = [feats.index(c) for c in cat_cols]
    oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
    a, b, c = FIXED_BLEND
    oof = np.clip(a*oc + b*ol + c*ox, 0, 100); tst = np.clip(a*tc + b*tl + c*tx, 0, 100)
    return oof, tst, (oc, ol, ox)


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols
    folds, _ = make_folds(train)
    tr_p = train[YEAR].value_counts(normalize=True); te_p = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_p.get(yr, 0.0) / tr_p.get(yr, np.nan)).values)
    def wm(p): return float(np.sum(w*(y-p)**2)/np.sum(w))

    # --- 5-meta base (faz13/kombo ile aynı) ---
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    meta = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te), 'berturk_meta': (bt_tr, bt_te)}
    for tag in TRANSFORMER_TAGS:
        op = os.path.join(exp_dir, f'oof_{tag}_train.npy'); tp = os.path.join(exp_dir, f'{tag}_test.npy')
        if os.path.exists(op) and os.path.exists(tp):
            meta[f'{tag}_meta'] = (np.clip(np.load(op).astype('float64'),0,100),
                                   np.clip(np.load(tp).astype('float64'),0,100))
    print(f"[faz22] metalar: {list(meta)}")

    # --- LLM feature'larını id ile hizala ---
    lf_tr = pd.read_csv(os.path.join(exp_dir, 'llm_feats_train.csv'))
    lf_te = pd.read_csv(os.path.join(exp_dir, 'llm_feats_test.csv'))
    assert (train[ID].values == lf_tr[ID].values).all(), "train id eşleşmiyor!"
    assert (test[ID].values == lf_te[ID].values).all(), "test id eşleşmiyor!"
    llm_cols = [c for c in lf_tr.columns if c not in (ID, 'parse_ok')]
    # sabit (varyansı 0) kolonları at
    llm_cols = [c for c in llm_cols if lf_tr[c].std() > 0]
    print(f"[faz22] LLM feature: {len(llm_cols)} kolon ({llm_cols[:6]}...)")
    train_fe = train_fe.copy(); test_fe = test_fe.copy()
    for c in llm_cols:
        train_fe[f'llm_{c}'] = lf_tr[c].values.astype('float64')
        test_fe[f'llm_{c}'] = lf_te[c].values.astype('float64')
    llm_num = [f'llm_{c}' for c in llm_cols]

    # --- (A) LLM'SİZ base (referans) ---
    print("\n[A] LLM'siz base kuruluyor...")
    oof_a, tst_a, _ = build_gbm_oof(train_fe, test_fe, num_cols, cat_cols, meta, y, folds)
    wA = wm(oof_a)
    print(f"[A] LLM'siz wOOF={wA:.4f}  (faz13/kombo base referansı)")

    # --- (B) LLM'Lİ ---
    print("\n[B] LLM'li base kuruluyor...")
    oof_b, tst_b, _ = build_gbm_oof(train_fe, test_fe, num_cols + llm_num, cat_cols, meta, y, folds)
    wB = wm(oof_b)
    print(f"[B] LLM'li   wOOF={wB:.4f}")

    print(f"\n================ FAZ 22 SONUÇ ================")
    print(f"[A] LLM'siz : {wA:.4f}")
    print(f"[B] LLM'li  : {wB:.4f}   (delta {wA-wB:+.4f})")
    verdict = ('GÜÇLÜ sinyal' if wA-wB > 0.3 else
               'zayıf/redundant' if wA-wB < 0.1 else 'orta')
    print(f"VERDICT (wOOF): {verdict}  [wOOF ince ayrımda güvenilmez; KARAR public ile]")

    # --- kombo: f13 yerine faz22(+llm) ile yeniden kur (kanıtlı tarif) ---
    sv2_o = np.load(os.path.join(exp_dir,'oof_stacker_v2.npy')).astype('float64')
    sv2_t = np.load(os.path.join(exp_dir,'test_stacker_v2.npy')).astype('float64')
    tab_o = np.clip(np.load(os.path.join(exp_dir,'oof_tabpfn_train.npy')).astype('float64'),0,100)
    tab_t = np.clip(np.load(os.path.join(exp_dir,'tabpfn_test.npy')).astype('float64'),0,100)
    f13_o = np.load(os.path.join(exp_dir,'oof_faz13_blend.npy')).astype('float64')
    # eski kombo referans (public 82.78 kalibrasyon)
    old_k = np.clip(0.375*sv2_o + 0.375*f13_o + 0.25*tab_o, 0, 100)
    # yeni kombo: f13 -> faz22(+llm)
    new_k_o = np.clip(0.375*sv2_o + 0.375*oof_b + 0.25*tab_o, 0, 100)
    new_k_t = np.clip(0.375*sv2_t + 0.375*tst_b + 0.25*tab_t, 0, 100)
    print(f"\nESKİ kombo wOOF = {wm(old_k):.4f}  (= public 82.78 kalibrasyon)")
    print(f"YENİ kombo(+llm) = {wm(new_k_o):.4f}  (delta {wm(old_k)-wm(new_k_o):+.4f})")

    np.save(os.path.join(exp_dir, 'oof_faz22_blend.npy'), oof_b)
    np.save(os.path.join(exp_dir, 'test_faz22_blend.npy'), tst_b)
    today = datetime.date.today().isoformat()
    # iki aday: saf faz22(+llm) base ve kombo(+llm)
    pd.DataFrame({ID: test[ID].values, TARGET: tst_b}).to_csv(
        os.path.join(sub_dir, f"sub_{today}_faz22_llm_base_woof{wB:.2f}.csv"), index=False)
    pd.DataFrame({ID: test[ID].values, TARGET: new_k_t}).to_csv(
        os.path.join(sub_dir, f"sub_{today}_kombo_llm_woof{wm(new_k_o):.2f}.csv"), index=False)
    log = {'phase':'faz22_llm_features','date':today,'metas':list(meta),'llm_cols':llm_cols,
           'woof_no_llm':wA,'woof_llm':wB,'delta':wA-wB,'verdict':verdict,
           'old_kombo_woof':wm(old_k),'new_kombo_woof':wm(new_k_o)}
    json.dump(log, open(os.path.join(exp_dir,'faz22_llm_log.json'),'w'), indent=2, ensure_ascii=False)
    print(f"\n[kayıt] oof/test_faz22_blend.npy + faz22_llm_log.json + 2 submission")


if __name__ == '__main__':
    main()
