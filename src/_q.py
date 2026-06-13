import numpy as np, pandas as pd, datetime
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
tfe,vfe=engineer(train),engineer(test)
eng=[c for c in tfe.columns if c not in train.columns]; num=rn+eng
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
p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy').astype('float64'),0,100)
te=Lf('tabpfn_test.npy')
mt=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),te])
mask=np.where(mt.std(1)<=np.quantile(mt.std(1),0.65))[0]
pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
def aug(fn,Xa,Xta):
    o=np.zeros(len(y)); t=np.zeros(len(Xta)); Xp=Xta[mask]; yp=p_te[mask]
    for tr,va in folds:
        m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp])); o[va]=m.predict(Xa[va]); t+=m.predict(Xta)/len(folds)
    return np.clip(o,0,100),np.clip(t,0,100)
eo,et2=aug(lambda:ExtraTreesRegressor(400,min_samples_leaf=5,n_jobs=-1,random_state=42),X,Xt)
ho,ht=aug(lambda:HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42),X,Xt)
mo,mtt=aug(lambda:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=42),Xz,Xtz)
ao=np.mean([eo,ho,mo],0); at=np.mean([et2,ht,mtt],0)
np.save(f'{ed}/test_trio_div.npy',at); np.save(f'{ed}/oof_trio_div.npy',ao)
today=datetime.date.today().isoformat()
print("bw  wOOF   (public: bw15=82.183 bw25=82.074)")
for bw in [0.35,0.40,0.45,0.50,0.55]:
    o=np.clip((1-bw)*pb_o+bw*ao,0,100); t=np.clip((1-bw)*p_te+bw*at,0,100)
    fn=f"sub_{today}_faz28_trio_bw{int(bw*100)}_woof{wm(o):.2f}.csv"
    pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/{fn}',index=False)
    print(f"{bw:.2f} {wm(o):.4f}  -> {fn}")
