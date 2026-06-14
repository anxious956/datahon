"""
Faz 35: ÇOK-TURLU AGRESİF PSEUDO (lider hipotezi). bw45 öğretmeniyle başla, 8 tur:
her tur 3-GBM (cat/lgb/xgb) train+TÜM-test-pseudo(pw=0.8) ile eğit -> yeni OOF+test ->
yeni öğretmen = test tahmini -> tekrar. Tur-tur wOOF eğrisi: 2'de mi doygun, devam mı?
DÜRÜST: pseudo wOOF confirmation-bias'la düşebilir -> public ile doğrula. Her tur test kaydedilir.
"""
import os, datetime, time
import numpy as np, pandas as pd
from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds
from faz3_ensemble import EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz14_seedbag import build_featB_X
from faz25_pseudo import aug_cat, aug_lgb, aug_xgb
LF=['ton','gelisim','somut_basari']; PW=0.8; ROUNDS=8
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
emb_tr=np.load(f'{ed}/{EMB_TRAIN_NPY}'); emb_te=np.load(f'{ed}/{EMB_TEST_NPY}')
bt_tr=np.load(f'{ed}/oof_berturk_train.npy').astype('float64'); bt_te=np.load(f'{ed}/berturk_test.npy').astype('float64')
txt_tr=train[TEXT].fillna(''); txt_te=test[TEXT].fillna('')
# featB matrisleri (cat/lgb/xgb için meta'lı)
Xc,Xc_t,Xg,Xg_t,ci,feats=build_featB_X(tfe,vfe,txt_tr,txt_te,y,num4,cc,emb_tr,emb_te,bt_tr,bt_te,folds)
tab_o=np.clip(np.load(f'{ed}/oof_tabpfn_train.npy'),0,100); tab_t=np.clip(np.load(f'{ed}/tabpfn_test.npy'),0,100)
trio_t=np.load(f'{ed}/test_trio_div.npy')
# başlangıç öğretmen = bw45 (en iyi)
pt=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy'),0,100)
teacher=np.clip(0.55*pt+0.45*trio_t,0,100)  # bw45 public 81.979
mask=np.arange(len(teacher))  # TÜM test (agresif)
today=datetime.date.today().isoformat()
print(f"[faz35] çok-turlu pseudo, pw={PW}, TÜM test, başlangıç öğretmen=bw45(81.979)")
print(f"{'tur':>4}{'3GBM-blend wOOF':>18}{'+div bw45 wOOF':>16}")
hist=[]
for r in range(1,ROUNDS+1):
    oc,tc=aug_cat(Xc,y,Xc_t,ci,folds,teacher,PW,mask)
    ol,tl=aug_lgb(Xg,y,Xg_t,cc,folds,teacher,PW,mask)
    ox,tx=aug_xgb(Xg,y,Xg_t,folds,teacher,PW,mask)
    blend_o=np.clip(0.7*oc+0.2*ol+0.1*ox,0,100); blend_t=np.clip(0.7*tc+0.2*tl+0.1*tx,0,100)
    # kombo benzeri: blend + tabpfn (sabit) + diversity
    kombo_o=np.clip(0.75*blend_o+0.25*tab_o,0,100); kombo_t=np.clip(0.75*blend_t+0.25*tab_t,0,100)
    div_o=np.clip(0.55*kombo_o+0.45*np.load(f'{ed}/oof_trio_div.npy'),0,100)
    div_t=np.clip(0.55*kombo_t+0.45*trio_t,0,100)
    print(f"{r:>4}{wm(kombo_o):>18.4f}{wm(div_o):>16.4f}",flush=True)
    hist.append((r,wm(kombo_o),wm(div_o)))
    pd.DataFrame({ID:test[ID].values,TARGET:div_t}).to_csv(f'{sd}/sub_{today}_multiR{r}_div45_woof{wm(div_o):.2f}.csv',index=False)
    teacher=kombo_t  # yeni öğretmen = bu turun kombo test tahmini
print("\n=== TUR-TUR wOOF (doygunluk analizi) ===")
for r,k,d in hist: print(f"  tur{r}: kombo={k:.4f}  +div={d:.4f}")
ks=[k for _,k,_ in hist]
print(f"\ndüşüş: tur1->2 {ks[0]-ks[1]:+.3f}, tur2->4 {ks[1]-ks[3]:+.3f}, tur4->8 {ks[3]-ks[-1]:+.3f}")
print("DEVAM ediyorsa (tur4->8 hala düşüyor) -> lider yolu olabilir, public'te doğrula")
