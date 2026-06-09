"""
Datathon 2026 — Faz 4: Feature Engineering -> mevcut ensemble (text_meta+emb_meta).

NEDEN: Plato ~86.2. GBM köprüsü güvenilir, FE ucuz + jüri-dostu + sızıntısız.
Hedef: ham 39 sayısala anlamlı türev feature'lar ekleyip wOOF'u düşürmek.

FE grupları (hepsi train/test'e BİREBİR aynı, hedef KULLANILMADAN):
- Agregasyon: teknik/yumuşak/mülakat skor setlerinin mean/std/min/max
- 100-tavan proxy: kaç skill ~maxed (>=95) -> tavan etkisi sinyali
- Oran/etkileşim: interview_rate, award_rate, avg_internship_len, github_impact, total_projects, cgpa×quality
- Yıl-türevi: years_to_grad, exp_years (app-grad), grad_age_proxy  (temporal bağlam)
- NaN bayrakları: 7 eksik kolon için "_isna" (NaN = 'yapmadı/yok' anlamı; staj süresi vb.)

Karşılaştırma: Faz 3 (+emb) blend wOOF 86.62 / cat 86.89 (FE'siz). FE bunu ne kadar düşürür?
"""
import os, json, datetime
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, SEED, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, tune_blend, EMB_TRAIN_NPY, EMB_TEST_NPY

SKILL = ['coding_score','problem_solving_score','data_structures_score','sql_score',
         'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']
SOFT = ['communication_score','teamwork_score','leadership_score','presentation_score']
INTERVIEW = ['technical_interview_score','hr_interview_score']
NA_FLAG_COLS = ['internship_duration_months','english_exam_score','github_avg_stars',
                'open_source_contribution_count','hr_interview_score','linkedin_profile_score','portfolio_score']

# Faz 8: explicit sentiment leksikonu (kök-bazlı, suffix'e dayanıklı substring).
# Bağımsız formül-avının tek bulgusu: net = pos - neg, hedefle 0.383 korele (bedava kestirme).
# GBM'in TF-IDF/embedding meta'sından ÇIKARMAKTA zorlandığı yön bilgisini explicit veriyoruz.
SENT_POS = ['mükemmel','güçlü','etkileyici','başarılı','yüksek','dikkat çekici','uzmanlık',
            'olağanüstü','üstün','yetkin','sağlam','parlak','övgü','etkili']
SENT_NEG = ['ancak','eksik','zayıf','geliştir','yetersiz','sınırlı','daha fazla çalışma',
            'gelişim göster','düşük','kaygı','endişe','sorun','rağmen','ne yazık']
TEXT_COL = 'mentor_feedback_text'


def add_sentiment(df):
    """mentor metninden olumlu/olumsuz kelime sayıları (satır-içi, sızıntısız)."""
    s = df[TEXT_COL].fillna('').str.lower()
    pos = s.apply(lambda t: sum(t.count(w) for w in SENT_POS))
    neg = s.apply(lambda t: sum(t.count(w) for w in SENT_NEG))
    df['sentiment_pos'] = pos.values
    df['sentiment_neg'] = neg.values
    df['sentiment_net'] = (pos - neg).values   # asıl sinyal (corr 0.383)
    return df


def engineer(df, sentiment=False):  # Faz 8: sentiment REDDEDİLDİ (wOOF -0.31, text zaten içeriyor)
    """
    Sızıntısız türev feature'lar (yalnız satır-içi, hedef yok). KÜRATÖRLÜ set.

    NOT: Önce 26-feature'lık geniş set denendi -> wOOF +0.6 KÖTÜLEŞTİ (blend 86.62->87.22).
    Suçlular: (a) yıl-türevleri (years_to_grad/exp_years/grad_age) temporal drift'i kodluyor,
    geç-yıl ağırlıklı metrikte zarar; (b) bölme-oranları (interview_rate vb.) NaN gürültüsü.
    Bunlar ÇIKARILDI. Kalan agregasyon + NaN bayrakları (11 feat) -> wOOF -0.35 (86.62->86.27).
    GBM oranları/etkileşimleri zaten kendi split'leriyle yakalıyor; sadece özet+eksik-sinyali katma.
    """
    df = df.copy()
    # Agregasyon (skor setlerinin özeti — GBM'in tek-kolon split'iyle yakalayamadığı global sinyal)
    df['skill_mean'] = df[SKILL].mean(axis=1)
    df['soft_mean'] = df[SOFT].mean(axis=1)
    df['interview_mean'] = df[INTERVIEW].mean(axis=1)
    df['maxed_skills'] = (df[SKILL] >= 95).sum(axis=1)   # 100-tavan proxy
    # NaN bayrakları ('yapmadı/yok' anlamı: staj süresi NaN = staj yok, vb.)
    for c in NA_FLAG_COLS:
        df[f'{c}_isna'] = df[c].isna().astype(int)
    if sentiment and TEXT_COL in df.columns:
        df = add_sentiment(df)
    return df


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]

    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols
    print(f"[FE] +{len(eng_cols)} türev feature -> toplam {len(num_cols)} sayısal: {eng_cols}")

    folds, _ = make_folds(train)
    tr_prop = train[YEAR].value_counts(normalize=True); te_prop = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda yr: te_prop.get(yr, 0.0) / tr_prop.get(yr, np.nan)).values)

    # Meta-feature'lar (Faz 2.1/2.2 ile aynı)
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    meta_cols = ['text_meta']; meta_tr = {'text_meta': tm_tr}; meta_te = {'text_meta': tm_te}
    emb_tr_p = os.path.join(exp_dir, EMB_TRAIN_NPY); emb_te_p = os.path.join(exp_dir, EMB_TEST_NPY)
    if os.path.exists(emb_tr_p) and os.path.exists(emb_te_p):
        em_tr, em_te = build_emb_meta(np.load(emb_tr_p), np.load(emb_te_p), y, folds)
        meta_cols.append('emb_meta'); meta_tr['emb_meta'] = em_tr; meta_te['emb_meta'] = em_te
    print(f"[meta] {meta_cols}")

    base_cols = num_cols + cat_cols
    Xc = train_fe[base_cols].copy(); Xc_t = test_fe[base_cols].copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[base_cols].copy(); Xg_t = test_fe[base_cols].copy()
    for c in cat_cols:
        cats = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cats); Xg_t[c] = Xg_t[c].astype(cats)
    for mc in meta_cols:
        Xc[mc] = meta_tr[mc]; Xc_t[mc] = meta_te[mc]; Xg[mc] = meta_tr[mc]; Xg_t[mc] = meta_te[mc]
    feats_c = base_cols + meta_cols
    cat_idx = [feats_c.index(c) for c in cat_cols]

    oof_cat, test_cat, importances, _ = run_catboost_cv(Xc, y, Xc_t, cat_idx, folds)
    oof_lgb, test_lgb, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    oof_xgb, test_xgb, _ = run_xgb_cv(Xg, y, Xg_t, folds)

    print("\n--- Tekil model OOF (düz | ağırlıklı) [FE'li] ---")
    for k, o in [('cat', oof_cat), ('lgb', oof_lgb), ('xgb', oof_xgb)]:
        p, wt, _ = metrics(y, o, w, train[YEAR], te_prop); print(f"  {k}: {p:.4f} | {wt:.4f}")

    wm, (wc, wl, wx) = tune_blend([oof_cat, oof_lgb, oof_xgb], y, w)
    oof_blend = np.clip(wc*oof_cat + wl*oof_lgb + wx*oof_xgb, 0, 100)
    test_blend = np.clip(wc*test_cat + wl*test_lgb + wx*test_xgb, 0, 100)
    b_plain, b_weighted, b_by_year = metrics(y, oof_blend, w, train[YEAR], te_prop)

    print("\n================ FAZ 4 (FE) SONUÇ — ağırlıklı OOF ================")
    print(f"Blend ağırlıkları: cat={wc:.2f} lgb={wl:.2f} xgb={wx:.2f}")
    print(f"Blend ağırlıklı OOF : {b_weighted:.4f}  (düz {b_plain:.4f})")
    print(f"  FE'siz referans     : cat 86.89 / blend 86.62 (Faz 3 +emb)")
    print(f"  kazanç vs Faz 3 blend: {86.6224 - b_weighted:+.4f}")
    print("\nYıl-bazlı blend OOF MSE (FE'li):")
    print(b_by_year.round(3).to_string())
    imp = pd.Series(importances, index=feats_c).sort_values(ascending=False)
    print("\nFeature importance top 15 (CatBoost):")
    print(imp.head(15).round(3).to_string())
    eng_in_top = [c for c in imp.head(20).index if c in eng_cols]
    print(f"\nİlk 20'de türev feature: {eng_in_top}")

    np.save(os.path.join(exp_dir, 'oof_blend_fe.npy'), oof_blend)
    log = {'phase': 'faz4_feature_engineering', 'date': datetime.date.today().isoformat(),
           'eng_cols': eng_cols, 'meta_cols': meta_cols,
           'blend_weights': {'cat': wc, 'lgb': wl, 'xgb': wx},
           'blend': {'plain': b_plain, 'weighted': b_weighted, 'by_year': b_by_year.round(4).to_dict()},
           'ref_faz3_blend_weighted': 86.6224, 'gain_vs_faz3': 86.6224 - b_weighted,
           'top_importance': imp.head(25).round(4).to_dict()}
    with open(os.path.join(exp_dir, 'faz4_fe_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: test_blend})
    fname = f"sub_{datetime.date.today().isoformat()}_blend_fe_oof{b_plain:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_blend_fe.npy + faz4_fe_log.json + {fname}")


if __name__ == '__main__':
    main()
