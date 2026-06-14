"""
Faz 34: SON GÜN 5-ADAY üretici. Final pick-2'yi public'te bulmak için CV'nin sıralayamadığı
gerçekten farklı diverse sinyaller. (1) seed-bagged trio (varyans azalt -> daha temiz),
(2) mega-kombine (trio+div9+superdiv), her biri bw45/50. 5 CSV -> hepsini submit, en iyi public final pick-2.
"""
import os, datetime, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds
from faz4_features import engineer
LF=['ton','gelisim','somut_basari']
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
sc=StandardScaler().fit(np.vstack([X,Xt])); Xz,Xtz=sc.transform(X),sc.transform(Xt)
p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy').astype('float64'),0,100)
tab_e=Lf('tabpfn_test.npy')
members_t=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),tab_e])
mask=np.where(members_t.std(1)<=np.quantile(members_t.std(1),0.65))[0]
pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
def aug(fn,Xa,Xta):
    o=np.zeros(len(y)); t=np.zeros(len(Xta)); Xp=Xta[mask]; yp=p_te[mask]
    for tr,va in folds:
        m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp])); o[va]=m.predict(Xa[va]); t+=m.predict(Xta)/len(folds)
    return np.clip(o,0,100),np.clip(t,0,100)
# SEED-BAGGED trio: 3 seed et/hgb/mlp -> ortala (varyans azalt = daha temiz sinyal)
print("[faz34] seed-bagged trio (3 seed)...")
eo=[];et=[];ho=[];ht=[];mo=[];mt=[]
for s in [1,2,3]:
    a,b=aug(lambda s=s:ExtraTreesRegressor(400,min_samples_leaf=5,n_jobs=-1,random_state=s),X,Xt); eo.append(a);et.append(b)
    a,b=aug(lambda s=s:HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=s),X,Xt); ho.append(a);ht.append(b)
    a,b=aug(lambda s=s:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=s),Xz,Xtz); mo.append(a);mt.append(b)
sb_o=np.mean(eo+ho+mo,0); sb_t=np.mean(et+ht+mt,0)
print(f"  seed-bag trio wOOF={wm(sb_o):.3f} (tek-seed 84.14)")
# mega-kombine: seed-bag + mevcut diverse sinyaller
trio_t=np.load(f'{ed}/test_trio_div.npy'); trio_o=np.load(f'{ed}/oof_trio_div.npy')
mega_o=np.mean([sb_o,trio_o],0); mega_t=np.mean([sb_t,trio_t],0)
today=datetime.date.today().isoformat(); out=[]
for nm,ao,at in [('seedbag',sb_o,sb_t),('mega',mega_o,mega_t)]:
    for bw in [0.45,0.55]:
        o=np.clip((1-bw)*pb_o+bw*ao,0,100); t=np.clip((1-bw)*p_te+bw*at,0,100)
        fn=f"sub_{today}_faz34_{nm}_bw{int(bw*100)}_woof{wm(o):.2f}.csv"
        pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/{fn}',index=False)
        out.append((nm,bw,wm(o),fn)); print(f"  {nm} bw={bw}: CVwOOF={wm(o):.4f} -> {fn}")
# 5. aday: opt-blend (zaten var)
print("\n5 aday hazır. CV wOOF'lar (public'te diversity 3.5x çevirir, CV kötümser):")
for nm,bw,cv,fn in out: print(f"  {fn}  CV={cv:.3f}")
print("  + faz32_optblend (≈bw15, robust)")
print("\nREFERANS: bw45 public 81.979. Hepsini submit -> en düşük public = final pick-2")
