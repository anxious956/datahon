"""
Faz 25: AGRESİF PSEUDO-LABELING (P0). Kombo (faz24-B, public 82.65) test tahminlerini
soft-label yap, GBM bazlarını train+pseudo-test ile YENİDEN eğit.

KRİTİK temizlik: pseudo satırlar her fold'un TRAIN'ine girer; VALIDATION her zaman sadece
orijinal train satırları -> wOOF sızıntısız (ama pseudo confirmation-bias'ı wOOF'u yapay
iyileştirebilir; wOOF = sanity-check, gerçek karar PUBLIC ile).

Yapı: sv2 (7-üye nested-ridge, GBM bazlar pseudo'lu) + f13 (5-meta 3-GBM, pseudo'lu) +
tabpfn (DEĞİŞMEZ). kombo = 0.375/0.375/0.25.
Varyantlar: (A) TÜM 10k pseudo, (B) üye-uyuşma filtreli (en güvenli %65). pw=0.5.
"""
import os, sys, json, datetime, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, CATBOOST_PARAMS
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, LGBM_PARAMS, XGB_PARAMS, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz5_linear import run_linear_cv
from faz14_seedbag import build_featB_X
from faz16_stacker import nested_ridge_stack, RIDGE_ALPHAS
from faz22_llm_features import build_gbm_oof

LLM_FEATS = ['ton', 'gelisim', 'somut_basari']
PW = float(os.environ.get('PW', '0.5'))           # pseudo ağırlığı
ROUND = os.environ.get('ROUND', '1')


def aug_cat(Xc, y, Xc_t, ci, folds, p_te, pw, mask):
    oof = np.zeros(len(Xc)); test = np.zeros(len(Xc_t))
    Xp = Xc_t.iloc[mask]; yp = p_te[mask]
    for tr, va in folds:
        Xtr = pd.concat([Xc.iloc[tr], Xp], ignore_index=True)
        ytr = np.concatenate([y[tr], yp])
        wtr = np.concatenate([np.ones(len(tr)), np.full(len(yp), pw)])
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(Pool(Xtr, ytr, cat_features=ci, weight=wtr),
              eval_set=Pool(Xc.iloc[va], y[va], cat_features=ci), use_best_model=True)
        oof[va] = m.predict(Xc.iloc[va]); test += m.predict(Xc_t) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def aug_lgb(Xg, y, Xg_t, cat_cols, folds, p_te, pw, mask):
    oof = np.zeros(len(Xg)); test = np.zeros(len(Xg_t))
    Xp = Xg_t.iloc[mask]; yp = p_te[mask]
    for tr, va in folds:
        Xtr = pd.concat([Xg.iloc[tr], Xp], ignore_index=True)
        ytr = np.concatenate([y[tr], yp]); wtr = np.concatenate([np.ones(len(tr)), np.full(len(yp), pw)])
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xg.iloc[va], y[va])],
              categorical_feature=cat_cols, callbacks=[early_stopping(200, verbose=False), log_evaluation(0)])
        oof[va] = m.predict(Xg.iloc[va]); test += m.predict(Xg_t) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def aug_xgb(Xg, y, Xg_t, folds, p_te, pw, mask):
    oof = np.zeros(len(Xg)); test = np.zeros(len(Xg_t))
    Xp = Xg_t.iloc[mask]; yp = p_te[mask]
    for tr, va in folds:
        Xtr = pd.concat([Xg.iloc[tr], Xp], ignore_index=True)
        ytr = np.concatenate([y[tr], yp]); wtr = np.concatenate([np.ones(len(tr)), np.full(len(yp), pw)])
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xg.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict(Xg.iloc[va]); test += m.predict(Xg_t) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def main():
    dd, ed, sd = resolve_paths()
    train = pd.read_csv(f'{dd}/train.csv'); test = pd.read_csv(f'{dd}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    tfe, vfe = engineer(train), engineer(test)
    eng = [c for c in tfe.columns if c not in train.columns]; num_cols = raw_num + eng
    folds, _ = make_folds(train)
    trp = train[YEAR].value_counts(normalize=True); tep = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda v: tep.get(v, 0.0)/trp.get(v, np.nan)).values)
    wm = lambda p: float(np.sum(w*(y-p)**2)/np.sum(w))

    lf_tr = pd.read_csv(f'{ed}/llm_feats_train.csv'); lf_te = pd.read_csv(f'{ed}/llm_feats_test.csv')
    lp_tr = pd.read_csv(f'{ed}/llm_pred_train.csv'); lp_te = pd.read_csv(f'{ed}/llm_pred_test.csv')
    for c in LLM_FEATS:
        tfe['llm_'+c] = lf_tr[c].values.astype('float64'); vfe['llm_'+c] = lf_te[c].values.astype('float64')
    tfe['llm_pred'] = lp_tr['pred'].values.astype('float64'); vfe['llm_pred'] = lp_te['pred'].values.astype('float64')
    num4 = num_cols + [f'llm_{c}' for c in LLM_FEATS] + ['llm_pred']
    # few-shot LLM (varsa) 5. feature olarak otomatik ekle
    if os.path.exists(f'{ed}/llm_pred2_train.csv') and os.environ.get('USE_FS','0')=='1':
        f2t=pd.read_csv(f'{ed}/llm_pred2_train.csv'); f2v=pd.read_csv(f'{ed}/llm_pred2_test.csv')
        tfe['llm_pred2']=f2t['pred'].values.astype('float64'); vfe['llm_pred2']=f2v['pred'].values.astype('float64')
        num4 = num4 + ['llm_pred2']
        print(f"[faz25] few-shot llm_pred2 EKLENDİ -> num4={len(num4)} | corr_y={np.corrcoef(f2t['pred'],y)[0,1]:.4f}")

    emb_tr = np.load(f'{ed}/{EMB_TRAIN_NPY}'); emb_te = np.load(f'{ed}/{EMB_TEST_NPY}')
    bt_tr = np.load(f'{ed}/oof_berturk_train.npy').astype('float64'); bt_te = np.load(f'{ed}/berturk_test.npy').astype('float64')
    txt_tr = train[TEXT].fillna(''); txt_te = test[TEXT].fillna('')
    tab_o = np.clip(np.load(f'{ed}/oof_tabpfn_train.npy').astype('float64'), 0, 100)
    tab_t = np.clip(np.load(f'{ed}/tabpfn_test.npy').astype('float64'), 0, 100)

    # pseudo etiket = kombo (faz24-B) test tahmini
    PLABEL = os.environ.get('PLABEL','test_faz24_B')
    p_te = np.clip(np.load(f'{ed}/{PLABEL}.npy').astype('float64'), 0, 100)
    print(f'[faz25] pseudo-etiket kaynağı: {PLABEL}')

    # üye-uyuşma maskesi (B): mevcut üye test tahminlerinin std'si düşük = güvenli
    members_t = np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'), np.load(f'{ed}/test_stacker_v2.npy'), tab_t])
    disagree = members_t.std(1)
    thr = np.quantile(disagree, 0.65)
    mask_all = np.arange(len(p_te))
    mask_agree = np.where(disagree <= thr)[0]
    print(f"[faz25] pseudo R{ROUND} pw={PW} | A=tüm {len(mask_all)} | B=uyuşma {len(mask_agree)} (thr std={thr:.2f})")

    # --- ortak meta matrisleri (build_featB_X: text/emb/berturk metalı Xc/Xg) ---
    Xc, Xc_t, Xg, Xg_t, ci, feats = build_featB_X(tfe, vfe, txt_tr, txt_te, y, num4, cat_cols,
                                                  emb_tr, emb_te, bt_tr, bt_te, folds)
    tm_o, tm_e = Xc['text_meta'].values, Xc_t['text_meta'].values
    em_o, em_e = Xc['emb_meta'].values, Xc_t['emb_meta'].values
    # f13 5-meta matrisleri (num4 + cat + 5 meta)
    b128_o = np.clip(np.load(f'{ed}/oof_bert128k_train.npy').astype('float64'),0,100); b128_e = np.clip(np.load(f'{ed}/bert128k_test.npy').astype('float64'),0,100)
    el_o = np.clip(np.load(f'{ed}/oof_electra_train.npy').astype('float64'),0,100); el_e = np.clip(np.load(f'{ed}/electra_test.npy').astype('float64'),0,100)
    metacols = {'text_meta':(tm_o,tm_e),'emb_meta':(em_o,em_e),'berturk_meta':(bt_tr,bt_te),
                'bert128k_meta':(b128_o,b128_e),'electra_meta':(el_o,el_e)}
    Xf = tfe[num4+cat_cols].copy(); Xf_t = vfe[num4+cat_cols].copy()
    for c in cat_cols:
        Xf[c]=Xf[c].astype(str); Xf_t[c]=Xf_t[c].astype(str)
    Xfg = tfe[num4+cat_cols].copy(); Xfg_t = vfe[num4+cat_cols].copy()
    for c in cat_cols:
        cc=pd.CategoricalDtype(categories=sorted(set(Xfg[c].dropna())|set(Xfg_t[c].dropna())))
        Xfg[c]=Xfg[c].astype(cc); Xfg_t[c]=Xfg_t[c].astype(cc)
    for m,(a,b) in metacols.items():
        Xf[m]=a; Xf_t[m]=b; Xfg[m]=a; Xfg_t[m]=b
    fci = [list(Xf.columns).index(c) for c in cat_cols]

    def build(mask, tag):
        t0=time.time()
        # sv2: 3 aug-GBM + lineer(pseudosuz) + nested ridge
        oc,tc = aug_cat(Xc,y,Xc_t,ci,folds,p_te,PW,mask)
        ol,tl = aug_lgb(Xg,y,Xg_t,cat_cols,folds,p_te,PW,mask)
        ox,tx = aug_xgb(Xg,y,Xg_t,folds,p_te,PW,mask)
        lin_num = num4+['text_meta','emb_meta','berturk_meta']
        Xl=tfe[num4+cat_cols].copy(); Xl_t=vfe[num4+cat_cols].copy()
        Xl['text_meta'],Xl['emb_meta'],Xl['berturk_meta']=tm_o,em_o,bt_tr
        Xl_t['text_meta'],Xl_t['emb_meta'],Xl_t['berturk_meta']=tm_e,em_e,bt_te
        _,_,olin,tlin = run_linear_cv(Xl,Xl_t,y,lin_num,cat_cols,folds,w)
        Mo=np.column_stack([oc,ol,ox,olin,tm_o,em_o,bt_tr]); Mt=np.column_stack([tc,tl,tx,tlin,tm_e,em_e,bt_te])
        sv2_o,sv2_t,sv2_w,sa = nested_ridge_stack(Mo,Mt,y,w,folds,RIDGE_ALPHAS)
        # f13: 3 aug-GBM, sabit blend 0.7/0.2/0.1
        fc_o,fc_t = aug_cat(Xf,y,Xf_t,fci,folds,p_te,PW,mask)
        fl_o,fl_t = aug_lgb(Xfg,y,Xfg_t,cat_cols,folds,p_te,PW,mask)
        fx_o,fx_t = aug_xgb(Xfg,y,Xfg_t,folds,p_te,PW,mask)
        f13_o=np.clip(0.7*fc_o+0.2*fl_o+0.1*fx_o,0,100); f13_t=np.clip(0.7*fc_t+0.2*fl_t+0.1*fx_t,0,100)
        k_o=np.clip(0.375*sv2_o+0.375*f13_o+0.25*tab_o,0,100); k_t=np.clip(0.375*sv2_t+0.375*f13_t+0.25*tab_t,0,100)
        print(f"[{tag}] sv2={sv2_w:.4f} f13={wm(f13_o):.4f} KOMBO={wm(k_o):.4f}  {time.time()-t0:.0f}s")
        return k_o,k_t

    print(f"\n[referans] faz24-B kombo wOOF=83.8959 (public 82.65); eski(llmsiz)=84.07")
    res={}
    for tag,mask in [('A_all',mask_all),('B_agree',mask_agree)]:
        ko,kt = build(mask,tag); res[tag]=(ko,kt)
        today=datetime.date.today().isoformat()
        fn=f"sub_{today}_pseudoR{ROUND}_{tag}_pw{PW}_woof{wm(ko):.2f}.csv"
        pd.DataFrame({ID:test[ID].values,TARGET:kt}).to_csv(f'{sd}/{fn}',index=False)
        np.save(f'{ed}/test_pseudoR{ROUND}_{tag}.npy',kt); np.save(f'{ed}/oof_pseudoR{ROUND}_{tag}.npy',ko)
        print(f"   -> {fn}")
    best=min(res,key=lambda k:wm(res[k][0]))
    print(f"\nEN İYİ: {best} wOOF={wm(res[best][0]):.4f} (faz24-B 83.8959'a delta {83.8959-wm(res[best][0]):+.4f})")
    json.dump({'phase':'faz25_pseudo','round':ROUND,'pw':PW,
               'A_woof':wm(res['A_all'][0]),'B_woof':wm(res['B_agree'][0]),'best':best},
              open(f'{ed}/faz25_R{ROUND}_log.json','w'),indent=2)


if __name__ == '__main__':
    main()
