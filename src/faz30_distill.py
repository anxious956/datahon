"""
Faz 30: GÜÇLÜ DISTILLATION-DENOISE. faz26'da diverse modeller 82.42 öğretmeninden distill
edilip BLEND ile 81.98 verdi (öğretmeni geçti -> denoise kazancı). Şimdi DAHA İYİ öğretmenden
(trio_bw45=81.98) + GERÇEK train etiketleriyle distill -> blend belki <81.98.
Çok model ailesi, TÜM 10k test pseudo (teacher 81.98), agresif blend. Son atış için.
"""
import os, datetime, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
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
    for nm,a,b in [('berturk_meta',np.load(f'{ed}/oof_berturk_train.npy'),np.load(f'{ed}/berturk_test.npy')),('bert128k_meta',Lf('oof_bert128k_train.npy'),Lf('bert128k_test.npy')),('electra_meta',Lf('oof_electra_train.npy'),Lf('electra_test.npy')),('tabpfn_meta',Lf('oof_tabpfn_train.npy'),Lf('tabpfn_test.npy'))]:
        tfe[nm]=a; vfe[nm]=b
    feat=num4+['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta']
    X=tfe[feat].copy(); Xt=vfe[feat].copy()
    for c in cc: X[c]=tfe[c].astype('category').cat.codes; Xt[c]=vfe[c].astype('category').cat.codes
    X=X.fillna(-999.0).values; Xt=Xt.fillna(-999.0).values
    sc=StandardScaler().fit(np.vstack([X,Xt])); Xz=sc.transform(X); Xtz=sc.transform(Xt)
    # ÖĞRETMEN = trio_bw45 (81.98), TÜM 10k test pseudo (mask yok = max)
    teacher=np.clip(np.load(f'{ed}/test_teacher_t1.npy').astype('float64'),0,100)
    mask=np.arange(len(teacher))   # TÜM test
    pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
    print(f"[faz30] öğretmen=trio_bw45(81.98), TÜM {len(mask)} test pseudo")
    def aug(fn,Xa,Xta):
        oof=np.zeros(len(y)); tp=np.zeros(len(Xta)); Xp=Xta[mask]; yp=teacher[mask]
        for tr,va in folds:
            m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp])); oof[va]=m.predict(Xa[va]); tp+=m.predict(Xta)/len(folds)
        return np.clip(oof,0,100),np.clip(tp,0,100)
    F={'et':(lambda:ExtraTreesRegressor(500,min_samples_leaf=4,n_jobs=-1,random_state=42),X,Xt),
       'rf':(lambda:RandomForestRegressor(400,min_samples_leaf=4,n_jobs=-1,random_state=42),X,Xt),
       'hgb':(lambda:HistGradientBoostingRegressor(max_iter=700,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42),X,Xt),
       'mlp1':(lambda:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=400,early_stopping=True,random_state=42),Xz,Xtz),
       'mlp2':(lambda:MLPRegressor(hidden_layer_sizes=(256,128),alpha=3e-3,max_iter=400,early_stopping=True,random_state=7),Xz,Xtz),
       'ridge':(lambda:Ridge(alpha=10.0),Xz,Xtz)}
    O={};T={}
    for nm,(fn,Xa,Xta) in F.items():
        o,t=aug(fn,Xa,Xta); O[nm]=o; T[nm]=t
        print(f"  {nm}: wOOF={wm(o):.2f} corr(teacher_oof? )={np.corrcoef(o,pb_o)[0,1]:.3f}")
    today=datetime.date.today().isoformat()
    # distill-avg (tüm 6) ve trio(et,hgb,mlp1)
    for cname,ms in [('d6',list(F)),('trio',['et','hgb','mlp1'])]:
        ao=np.mean([O[m] for m in ms],0); at=np.mean([T[m] for m in ms],0)
        print(f"\n[{cname}] distill-avg wOOF={wm(ao):.3f} (öğretmen-oof {wm(pb_o):.3f})")
        # blend teacher_test ile (teacher=81.98)
        for bw in [0.3,0.45,0.6,0.75,1.0]:
            t=np.clip((1-bw)*teacher+bw*at,0,100)
            o=np.clip((1-bw)*pb_o+bw*ao,0,100)
            fn=f"sub_{today}_faz30_{cname}_bw{int(bw*100)}_woof{wm(o):.2f}.csv"
            pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/{fn}',index=False)
            print(f"  {cname} bw={bw}: blendwOOF={wm(o):.4f} -> {fn}")
    print("[faz30] CSV'ler hazır")
if __name__=='__main__': main()
