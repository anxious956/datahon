"""
Faz 27: ÇEŞİTLİLİK v2 (faz26 public'te wOOF'un 3.5 katı çevirdi -> bu eksende DERİNLEŞ).
Çok daha fazla model ailesi (ET/RF/HistGBM/GBM/2xMLP/kNN/Ridge/ElasticNet), hepsi pseudo-aug.
Ortalamalarını pseudo-B'ye DAHA YÜKSEK ağırlıkla blend et (wOOF kötümser -> public daha iyi).
Birden çok ağırlık CSV'si üret (0.20/0.30/0.40) -> public'te test.
"""
import os, datetime, time
import numpy as np, pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds
from faz4_features import engineer

LLM_FEATS=['ton','gelisim','somut_basari']

def main():
    dd,ed,sd=resolve_paths()
    train=pd.read_csv(f'{dd}/train.csv'); test=pd.read_csv(f'{dd}/test_x.csv'); y=train[TARGET].values
    cat_cols=[c for c in train.columns if c not in (ID,TEXT) and is_str_col(train[c])]
    raw_num=[c for c in train.columns if c not in (ID,TARGET,TEXT,*cat_cols)]
    tfe,vfe=engineer(train),engineer(test)
    eng=[c for c in tfe.columns if c not in train.columns]; num_cols=raw_num+eng
    folds,_=make_folds(train)
    trp=train[YEAR].value_counts(normalize=True); tep=test[YEAR].value_counts(normalize=True)
    w=np.nan_to_num(train[YEAR].map(lambda v:tep.get(v,0.0)/trp.get(v,np.nan)).values)
    wm=lambda p:float(np.sum(w*(y-p)**2)/np.sum(w))

    lf_tr=pd.read_csv(f'{ed}/llm_feats_train.csv'); lf_te=pd.read_csv(f'{ed}/llm_feats_test.csv')
    lp_tr=pd.read_csv(f'{ed}/llm_pred_train.csv'); lp_te=pd.read_csv(f'{ed}/llm_pred_test.csv')
    for c in LLM_FEATS: tfe['llm_'+c]=lf_tr[c].values; vfe['llm_'+c]=lf_te[c].values
    tfe['llm_pred']=lp_tr['pred'].values; vfe['llm_pred']=lp_te['pred'].values
    num4=num_cols+[f'llm_{c}' for c in LLM_FEATS]+['llm_pred']
    def L(n): return np.clip(np.load(f'{ed}/{n}').astype('float64'),0,100)
    for nm,a,b in [('berturk_meta',np.load(f'{ed}/oof_berturk_train.npy'),np.load(f'{ed}/berturk_test.npy')),
                   ('bert128k_meta',L('oof_bert128k_train.npy'),L('bert128k_test.npy')),
                   ('electra_meta',L('oof_electra_train.npy'),L('electra_test.npy')),
                   ('tabpfn_meta',L('oof_tabpfn_train.npy'),L('tabpfn_test.npy'))]:
        tfe[nm]=a; vfe[nm]=b
    feat=num4+['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta']
    X=tfe[feat].copy(); Xt=vfe[feat].copy()
    for c in cat_cols: X[c]=tfe[c].astype('category').cat.codes; Xt[c]=vfe[c].astype('category').cat.codes
    X=X.fillna(-999.0).values; Xt=Xt.fillna(-999.0).values
    sc=StandardScaler().fit(np.vstack([X,Xt])); Xz=sc.transform(X); Xtz=sc.transform(Xt)

    p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy').astype('float64'),0,100)
    tab_e=L('tabpfn_test.npy')
    members_t=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),tab_e])
    mask=np.where(members_t.std(1)<=np.quantile(members_t.std(1),0.65))[0]
    pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
    print(f"[faz27] pseudo-B wOOF={wm(pb_o):.4f} (public 82.183 referans), mask {len(mask)}")

    def aug(fn,Xa,Xta):
        oof=np.zeros(len(y)); tp=np.zeros(len(Xta)); Xp=Xta[mask]; yp=p_te[mask]
        for tr,va in folds:
            m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp]))
            oof[va]=m.predict(Xa[va]); tp+=m.predict(Xta)/len(folds)
        return np.clip(oof,0,100),np.clip(tp,0,100)

    fams={
      'et':(lambda:ExtraTreesRegressor(500,min_samples_leaf=4,n_jobs=-1,random_state=1),X,Xt),
      'rf':(lambda:RandomForestRegressor(400,min_samples_leaf=4,n_jobs=-1,random_state=2),X,Xt),
      'hgb':(lambda:HistGradientBoostingRegressor(max_iter=700,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=3),X,Xt),
      'mlp1':(lambda:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=5),Xz,Xtz),
      'mlp2':(lambda:MLPRegressor(hidden_layer_sizes=(256,),alpha=3e-3,max_iter=300,early_stopping=True,random_state=6),Xz,Xtz),
      'knn':(lambda:KNeighborsRegressor(n_neighbors=30,weights='distance'),Xz,Xtz),
      'ridge':(lambda:Ridge(alpha=10.0),Xz,Xtz),
      'enet':(lambda:ElasticNet(alpha=0.01,l1_ratio=0.3,max_iter=2000),Xz,Xtz),
    }
    oofs={}; tests={}
    for nm,(fn,Xa,Xta) in fams.items():
        t0=time.time(); o,t=aug(fn,Xa,Xta); oofs[nm]=o; tests[nm]=t
        print(f"  {nm}: wOOF={wm(o):.3f} corr(pB)={np.corrcoef(o,pb_o)[0,1]:.3f} {time.time()-t0:.0f}s")

    avg_o=np.mean(list(oofs.values()),0); avg_t=np.mean(list(tests.values()),0)
    today=datetime.date.today().isoformat()
    print(f"\n[faz27] divavg(9 aile) wOOF={wm(avg_o):.3f} corr(pB)={np.corrcoef(avg_o,pb_o)[0,1]:.3f}")
    for bw in [0.15,0.20,0.25,0.30,0.40]:
        o=np.clip((1-bw)*pb_o+bw*avg_o,0,100); t=np.clip((1-bw)*p_te+bw*avg_t,0,100)
        print(f"  blend bw={bw}: wOOF={wm(o):.4f}")
        fn=f"sub_{today}_faz27_div9_bw{int(bw*100)}_woof{wm(o):.2f}.csv"
        pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/{fn}',index=False)
    np.save(f'{ed}/oof_div9avg.npy',avg_o); np.save(f'{ed}/test_div9avg.npy',avg_t)
    print(f"[faz27] 5 CSV (bw 0.15-0.40) hazır + div9avg npy")

if __name__=='__main__':
    main()
