"""
Faz 31: SÜPER-ÇEŞİTLİ ensemble (floor'u düşürme denemesi). Diversity floor 81.979'du (trio).
Daha fazla decorrelation kaynağı: FARKLI feature görünümleri (sadece-meta / sadece-num /
sadece-text) + yeni algoritmalar (PLS, SVR, farklı MLP). Decorrelation arttıkça blend floor'u düşer.
Hepsi pseudo-aug. wOOF + public-trend ile optimal bw -> en iyi CSV.
"""
import os, datetime, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds
from faz4_features import engineer
LF=['ton','gelisim','somut_basari']
def main():
    dd,ed,sd=resolve_paths()
    train=pd.read_csv(f'{dd}/train.csv'); test=pd.read_csv(f'{dd}/test_x.csv'); y=train[TARGET].values
    cc=[c for c in train.columns if c not in (ID,TEXT) and is_str_col(train[c])]
    rn=[c for c in train.columns if c not in (ID,TARGET,TEXT,*cc)]
    tfe,vfe=engineer(train),engineer(test); eng=[c for c in tfe.columns if c not in train.columns]; num=rn+eng
    folds,_=make_folds(train)
    trp=train[YEAR].value_counts(normalize=True); tep=test[YEAR].value_counts(normalize=True)
    w=np.nan_to_num(train[YEAR].map(lambda v:tep.get(v,0.0)/trp.get(v,np.nan)).values); wm=lambda p:float(np.sum(w*(y-p)**2)/np.sum(w))
    lf=pd.read_csv(f'{ed}/llm_feats_train.csv');lfe=pd.read_csv(f'{ed}/llm_feats_test.csv')
    lp=pd.read_csv(f'{ed}/llm_pred_train.csv');lpe=pd.read_csv(f'{ed}/llm_pred_test.csv')
    for c in LF: tfe['llm_'+c]=lf[c].values; vfe['llm_'+c]=lfe[c].values
    tfe['llm_pred']=lp['pred'].values; vfe['llm_pred']=lpe['pred'].values
    num4=num+[f'llm_{c}' for c in LF]+['llm_pred']
    def Lf(n): return np.clip(np.load(f'{ed}/{n}').astype('float64'),0,100)
    metacols=['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta','llm_pred']
    for nm,a,b in [('berturk_meta',np.load(f'{ed}/oof_berturk_train.npy'),np.load(f'{ed}/berturk_test.npy')),('bert128k_meta',Lf('oof_bert128k_train.npy'),Lf('bert128k_test.npy')),('electra_meta',Lf('oof_electra_train.npy'),Lf('electra_test.npy')),('tabpfn_meta',Lf('oof_tabpfn_train.npy'),Lf('tabpfn_test.npy'))]:
        tfe[nm]=a; vfe[nm]=b
    allfeat=num4+['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta']
    numonly=[c for c in num4 if c not in ['llm_pred']]   # sadece sayısal+llm-feat
    def mat(cols):
        Xa=tfe[cols].copy(); Xb=vfe[cols].copy()
        for c in cc:
            if c in cols: Xa[c]=tfe[c].astype('category').cat.codes; Xb[c]=vfe[c].astype('category').cat.codes
        return Xa.fillna(-999.0).values, Xb.fillna(-999.0).values
    Xall,Xtall=mat(allfeat); Xnum,Xtnum=mat(numonly)
    Xmeta=tfe[metacols].values; Xtmeta=vfe[metacols].values
    sc=StandardScaler().fit(np.vstack([Xall,Xtall])); Xz,Xtz=sc.transform(Xall),sc.transform(Xtall)
    scn=StandardScaler().fit(np.vstack([Xnum,Xtnum])); Xnz,Xtnz=scn.transform(Xnum),scn.transform(Xtnum)
    p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy').astype('float64'),0,100)
    tab_e=Lf('tabpfn_test.npy')
    members_t=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),tab_e])
    mask=np.where(members_t.std(1)<=np.quantile(members_t.std(1),0.65))[0]
    pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
    print(f"[faz31] pseudo-B wOOF={wm(pb_o):.4f} (public 81.979@bw45 referans)")
    def aug(fn,Xa,Xta):
        oof=np.zeros(len(y)); tp=np.zeros(len(Xta)); Xp=Xta[mask]; yp=p_te[mask]
        for tr,va in folds:
            m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp])); oof[va]=m.predict(Xa[va]).ravel(); tp+=m.predict(Xta).ravel()/len(folds)
        return np.clip(oof,0,100),np.clip(tp,0,100)
    F=[ # (isim, fn, X_oof, X_test) -- FARKLI feature görünümleri = decorrelation
      ('et_all',lambda:ExtraTreesRegressor(500,min_samples_leaf=4,n_jobs=-1,random_state=1),Xall,Xtall),
      ('hgb_all',lambda:HistGradientBoostingRegressor(max_iter=700,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=3),Xall,Xtall),
      ('mlp_all',lambda:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=400,early_stopping=True,random_state=5),Xz,Xtz),
      ('et_num',lambda:ExtraTreesRegressor(400,min_samples_leaf=5,n_jobs=-1,random_state=11),Xnum,Xtnum),   # sadece num görünüm
      ('mlp_num',lambda:MLPRegressor(hidden_layer_sizes=(96,),alpha=3e-3,max_iter=400,early_stopping=True,random_state=12),Xnz,Xtnz),
      ('hgb_meta',lambda:HistGradientBoostingRegressor(max_iter=400,learning_rate=0.05,max_depth=3,random_state=13),Xmeta,Xtmeta),  # sadece meta görünüm
      ('ridge_meta',lambda:Ridge(alpha=5.0),Xmeta,Xtmeta),
      ('pls',lambda:PLSRegression(n_components=12),Xz,Xtz),       # yeni algoritma
      ('lsvr',lambda:LinearSVR(C=0.5,max_iter=3000),Xz,Xtz),      # yeni algoritma
      ('mlp_deep',lambda:MLPRegressor(hidden_layer_sizes=(256,128,64),alpha=5e-3,max_iter=400,early_stopping=True,random_state=21),Xz,Xtz),
    ]
    O={};T={}
    for nm,fn,Xa,Xta in F:
        t0=time.time(); o,t=aug(fn,Xa,Xta); O[nm]=o;T[nm]=t
        print(f"  {nm}: wOOF={wm(o):.2f} corr(pB)={np.corrcoef(o,pb_o)[0,1]:.3f} {time.time()-t0:.0f}s")
    # decorrelation-ağırlıklı ortalama (pB ile en az korele olanlara biraz daha ağırlık)
    names=list(O); A_o=np.mean([O[n] for n in names],0); A_t=np.mean([T[n] for n in names],0)
    print(f"\n[faz31] super-div avg wOOF={wm(A_o):.3f} corr(pB)={np.corrcoef(A_o,pb_o)[0,1]:.3f}")
    today=datetime.date.today().isoformat(); best=None
    for bw in [0.30,0.40,0.45,0.50,0.55,0.60]:
        o=np.clip((1-bw)*pb_o+bw*A_o,0,100); t=np.clip((1-bw)*p_te+bw*A_t,0,100)
        print(f"  bw={bw}: CVwOOF={wm(o):.4f}")
        if best is None or wm(o)<best[0]: best=(wm(o),bw,t)
    bw=best[1]; t=best[2]
    fn=f"sub_{today}_faz31_superdiv_bw{int(bw*100)}_woof{best[0]:.2f}.csv"
    pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/{fn}',index=False)
    np.save(f'{ed}/oof_superdiv.npy',A_o); np.save(f'{ed}/test_superdiv.npy',A_t)
    print(f"\n>>> EN İYİ CV bw={bw} wOOF={best[0]:.4f} -> {fn}")
    print(f"   (trio floor wOOF~82.10/public81.979 -> super-div wOOF {best[0]:.3f})")
if __name__=='__main__': main()
