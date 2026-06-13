"""
Faz 26: ÇEŞİTLİLİK EKSENİ. Pseudo-B (public 82.424) üzerine FARKLI MODEL AİLELERİ ekle
(ExtraTrees + HistGBM + MLP), hepsi pseudo-augmented. Ensemble decorrelation -> kazanç umudu.
Tüm ağaç-bazlıydık; bu yeni eksen. wOOF + en iyi blend CSV.
"""
import os, json, datetime, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds
from faz3_ensemble import EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer

LLM_FEATS = ['ton','gelisim','somut_basari']

def main():
    dd, ed, sd = resolve_paths()
    train = pd.read_csv(f'{dd}/train.csv'); test = pd.read_csv(f'{dd}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID,TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID,TARGET,TEXT,*cat_cols)]
    tfe, vfe = engineer(train), engineer(test)
    eng = [c for c in tfe.columns if c not in train.columns]; num_cols = raw_num + eng
    folds,_ = make_folds(train)
    trp=train[YEAR].value_counts(normalize=True); tep=test[YEAR].value_counts(normalize=True)
    w=np.nan_to_num(train[YEAR].map(lambda v:tep.get(v,0.0)/trp.get(v,np.nan)).values)
    wm=lambda p:float(np.sum(w*(y-p)**2)/np.sum(w))

    # LLM feature'lar
    lf_tr=pd.read_csv(f'{ed}/llm_feats_train.csv'); lf_te=pd.read_csv(f'{ed}/llm_feats_test.csv')
    lp_tr=pd.read_csv(f'{ed}/llm_pred_train.csv'); lp_te=pd.read_csv(f'{ed}/llm_pred_test.csv')
    for c in LLM_FEATS:
        tfe['llm_'+c]=lf_tr[c].values; vfe['llm_'+c]=lf_te[c].values
    tfe['llm_pred']=lp_tr['pred'].values; vfe['llm_pred']=lp_te['pred'].values
    num4=num_cols+[f'llm_{c}' for c in LLM_FEATS]+['llm_pred']

    # metalar (hepsi sayısal feature olarak)
    def L(n): return np.clip(np.load(f'{ed}/{n}').astype('float64'),0,100)
    metas={'text': None,'emb':None}
    # text/emb metaları yeniden türetmek yerine berturk/bert128k/electra OOF kullan + e5 ham yok; basit tut
    bt_o=np.load(f'{ed}/oof_berturk_train.npy'); bt_e=np.load(f'{ed}/berturk_test.npy')
    b128_o=L('oof_bert128k_train.npy'); b128_e=L('bert128k_test.npy')
    el_o=L('oof_electra_train.npy'); el_e=L('electra_test.npy')
    tab_o=L('oof_tabpfn_train.npy'); tab_e=L('tabpfn_test.npy')
    for nm,a,b in [('berturk_meta',bt_o,bt_e),('bert128k_meta',b128_o,b128_e),
                   ('electra_meta',el_o,el_e),('tabpfn_meta',tab_o,tab_e)]:
        tfe[nm]=a; vfe[nm]=b
    feat = num4 + ['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta']

    # sayısal matris (kategorikler kod olarak)
    X=tfe[feat].copy(); Xt=vfe[feat].copy()
    for c in cat_cols:
        X[c]=tfe[c].astype('category').cat.codes; Xt[c]=vfe[c].astype('category').cat.codes
    X=X.fillna(-999.0); Xt=Xt.fillna(-999.0)
    Xs=StandardScaler().fit(pd.concat([X,Xt])); Xz=Xs.transform(X); Xtz=Xs.transform(Xt)

    # pseudo etiket = pseudo-B (en iyi)
    p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy').astype('float64'),0,100)
    members_t=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),tab_e])
    mask=np.where(members_t.std(1)<=np.quantile(members_t.std(1),0.65))[0]
    print(f"[faz26] pseudo mask {len(mask)}/10000")

    def aug(model_fn, Xtr_full, Xte_full, scaled=False):
        oof=np.zeros(len(y)); test_p=np.zeros(len(Xte_full))
        Xp=Xte_full[mask]; yp=p_te[mask]
        for tr,va in folds:
            Xtr=np.vstack([Xtr_full[tr],Xp]); ytr=np.concatenate([y[tr],yp])
            m=model_fn().fit(Xtr,ytr)
            oof[va]=m.predict(Xtr_full[va]); test_p+=m.predict(Xte_full)/len(folds)
        return np.clip(oof,0,100), np.clip(test_p,0,100)

    Xv=X.values; Xtv=Xt.values
    models={
      'extratrees': (lambda: ExtraTreesRegressor(n_estimators=400,min_samples_leaf=5,n_jobs=-1,random_state=42), Xv, Xtv),
      'histgbm':    (lambda: HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42), Xv, Xtv),
      'mlp':        (lambda: MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=42), Xz, Xtz),
    }
    pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')  # ~pseudo-B oof
    print(f"[faz26] pseudo-B(referans) wOOF={wm(pb_o):.4f}  (public 82.424)")
    new_oofs={}; new_tests={}
    for nm,(fn,Xa,Xta) in models.items():
        t0=time.time(); o,t=aug(fn,Xa,Xta)
        new_oofs[nm]=o; new_tests[nm]=t
        print(f"  {nm}: wOOF={wm(o):.4f}  corr(pseudoB)={np.corrcoef(o,pb_o)[0,1]:.4f}  {time.time()-t0:.0f}s")

    # blend taraması: pseudo-B'ye yeni aileleri kat
    pb_t=p_te
    best=(wm(pb_o), 'pseudoB', pb_t)
    for nm in models:
        for bw in [0.1,0.15,0.2,0.25]:
            o=np.clip((1-bw)*pb_o+bw*new_oofs[nm],0,100)
            if wm(o)<best[0]:
                best=(wm(o), f'pseudoB+{bw}*{nm}', np.clip((1-bw)*pb_t+bw*new_tests[nm],0,100))
    # üçlü ortalama da dene
    avg_o=np.mean([new_oofs[m] for m in models],0); avg_t=np.mean([new_tests[m] for m in models],0)
    for bw in [0.1,0.15,0.2,0.25,0.3]:
        o=np.clip((1-bw)*pb_o+bw*avg_o,0,100)
        if wm(o)<best[0]:
            best=(wm(o), f'pseudoB+{bw}*divavg', np.clip((1-bw)*pb_t+bw*avg_t,0,100))
    print(f"\n[faz26] EN İYİ: {best[1]} wOOF={best[0]:.4f}  (pseudoB {wm(pb_o):.4f}, delta {wm(pb_o)-best[0]:+.4f})")

    if best[0] < wm(pb_o)-0.005:
        today=datetime.date.today().isoformat()
        fn=f"sub_{today}_faz26_{best[1].replace('*','x').replace('+','_').replace('.','')}_woof{best[0]:.2f}.csv"
        pd.DataFrame({ID:test[ID].values,TARGET:best[2]}).to_csv(f'{sd}/{fn}',index=False)
        print(f"[kayıt] {fn}")
    else:
        print("[faz26] çeşitlilik anlamlı kazanç vermedi (<0.005) -> CSV yok")

if __name__=='__main__':
    main()
