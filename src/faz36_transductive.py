"""
Faz 36: TRANSDUCTIVE feature'lar (lider hipotezi-2). Test PUBLIC -> kural-içi.
Temporal-shift'te transductive güçlü olabilir: (1) train+test ortak PCA, (2) test-istatistik
feature'ları (her satırın test-dağılımındaki konumu), (3) train+test KMeans cluster.
Bunları num4'e ekle -> diversity ensemble yeniden -> floor (81.979/CV82.36) düşer mi?
"""
import os, datetime, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
basef=num4+['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta']
# sayısal matris
def mat(df,cols):
    X=df[cols].copy()
    for c in cc:
        if c in cols: X[c]=df[c].astype('category').cat.codes
    return X.fillna(-999.0).values
Xtr_b=mat(tfe,basef); Xte_b=mat(vfe,basef)
# === TRANSDUCTIVE feature'lar (train+test BİRLİKTE) ===
allX=np.vstack([Xtr_b,Xte_b]); sc=StandardScaler().fit(allX); allZ=sc.transform(allX)
ntr=len(Xtr_b)
# (1) train+test ortak PCA
pca=PCA(n_components=8,random_state=42).fit(allZ); P=pca.transform(allZ)
# (2) train+test KMeans cluster -> mesafe feature
km=KMeans(n_clusters=8,random_state=42,n_init=3).fit(allZ); D=km.transform(allZ)  # her cluster'a mesafe
# (3) test-istatistik: her satırın TEST dağılımındaki z-skoru (test mean/std ile)
te_mean=Xte_b.mean(0); te_std=Xte_b.std(0)+1e-9
Ztest=(allX-te_mean)/te_std  # test-dağılımına göre konum
trans=np.hstack([P,D])  # PCA8 + cluster-dist8 (hafif)
print(f'[faz36] transductive feature: PCA8+clusterdist8 = {trans.shape[1]}',flush=True)
# diversity ensemble: base vs base+transductive
p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy'),0,100); tab_e=Lf('tabpfn_test.npy')
members_t=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),tab_e])
mask=np.where(members_t.std(1)<=np.quantile(members_t.std(1),0.65))[0]
pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
def trio(Xtr,Xte):
    sc2=StandardScaler().fit(np.vstack([Xtr,Xte])); Xz,Xtz=sc2.transform(Xtr),sc2.transform(Xte)
    def aug(fn,Xa,Xta):
        o=np.zeros(len(y)); t=np.zeros(len(Xta)); Xp=Xta[mask]; yp=p_te[mask]
        for tr,va in folds:
            m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp])); o[va]=m.predict(Xa[va]); t+=m.predict(Xta)/len(folds)
        return np.clip(o,0,100),np.clip(t,0,100)
    eo,et=aug(lambda:ExtraTreesRegressor(300,min_samples_leaf=5,n_jobs=-1,random_state=42),Xtr,Xte)
    ho,ht=aug(lambda:HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42),Xtr,Xte)
    mo,mt=aug(lambda:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=42),Xz,Xtz)
    return np.mean([eo,ho,mo],0),np.mean([et,ht,mt],0)
Ptr,Pte=trans[:ntr],trans[ntr:]
a0=np.load(f"{ed}/oof_trio_div.npy")  # BASE bilinen (bw45 82.36)
print("[+TRANS] transductive'li..."); a1,t1=trio(np.hstack([Xtr_b,Ptr]),np.hstack([Xte_b,Pte]))
print(f"\n{'':>10}{'trio wOOF':>12}{'bw15':>10}{'bw45':>10}")
for nm,ao in [('BASE',a0),('+TRANS',a1)]:
    print(f"  {nm:<8}{wm(ao):>12.3f}{wm(np.clip(0.85*pb_o+0.15*ao,0,100)):>10.4f}{wm(np.clip(0.55*pb_o+0.45*ao,0,100)):>10.4f}")
today=datetime.date.today().isoformat()
for bw in [0.25,0.45]:
    t=np.clip((1-bw)*p_te+bw*t1,0,100); o=np.clip((1-bw)*pb_o+bw*a1,0,100)
    pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/sub_{today}_faz36_trans_bw{int(bw*100)}_woof{wm(o):.2f}.csv',index=False)
print("(referans: bw45 CV82.36/public81.979) -- +TRANS bw45 düşükse transductive çalışıyor")
