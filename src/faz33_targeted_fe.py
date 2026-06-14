"""
Faz 33: HEDEFLİ FE testi (diversity pipeline'da). Geniş FE base'de zarar verdi (yıl-drift).
Ama HEDEFLİ yüksek-sinyal interaction'lar (en güçlü pair-R² çiftleri) diversity ensemble'da
yarayabilir. Birkaç güvenli interaction/agregasyon ekle -> trio yeniden kur -> floor düşer mi?
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
def main():
    dd,ed,sd=resolve_paths()
    train=pd.read_csv(f'{dd}/train.csv'); test=pd.read_csv(f'{dd}/test_x.csv'); y=train[TARGET].values
    cc=[c for c in train.columns if c not in (ID,TEXT) and is_str_col(train[c])]
    rn=[c for c in train.columns if c not in (ID,TARGET,TEXT,*cc)]
    tfe,vfe=engineer(train),engineer(test); eng=[c for c in tfe.columns if c not in train.columns]; num=rn+eng
    folds,_=make_folds(train)
    trp=train[YEAR].value_counts(normalize=True); tep=test[YEAR].value_counts(normalize=True)
    w=np.nan_to_num(train[YEAR].map(lambda v:tep.get(v,0.0)/trp.get(v,np.nan)).values); wm=lambda p:float(np.sum(w*(y-p)**2)/np.sum(w))
    # LLM + meta feature'lar
    lf=pd.read_csv(f'{ed}/llm_feats_train.csv');lfe=pd.read_csv(f'{ed}/llm_feats_test.csv')
    lp=pd.read_csv(f'{ed}/llm_pred_train.csv');lpe=pd.read_csv(f'{ed}/llm_pred_test.csv')
    for c in LF: tfe['llm_'+c]=lf[c].values; vfe['llm_'+c]=lfe[c].values
    tfe['llm_pred']=lp['pred'].values; vfe['llm_pred']=lpe['pred'].values
    num4=num+[f'llm_{c}' for c in LF]+['llm_pred']
    def Lf(n): return np.clip(np.load(f'{ed}/{n}').astype('float64'),0,100)
    for nm,a,b in [('berturk_meta',np.load(f'{ed}/oof_berturk_train.npy'),np.load(f'{ed}/berturk_test.npy')),('bert128k_meta',Lf('oof_bert128k_train.npy'),Lf('bert128k_test.npy')),('electra_meta',Lf('oof_electra_train.npy'),Lf('electra_test.npy')),('tabpfn_meta',Lf('oof_tabpfn_train.npy'),Lf('tabpfn_test.npy'))]:
        tfe[nm]=a; vfe[nm]=b
    base_feat=num4+['berturk_meta','bert128k_meta','electra_meta','tabpfn_meta']
    # --- HEDEFLİ yeni feature'lar (en güçlü sinyaller, güvenli) ---
    NEWF=[]
    for d in (tfe,vfe):
        d['f_pq_x_ti']=d['project_quality_score']*d['technical_interview_score']/100   # pair-R2 0.37
        d['f_pq_x_ps']=d['project_quality_score']*d['problem_solving_score']/100
        d['f_pq_x_cod']=d['project_quality_score']*d['coding_score']/100
        d['f_tech_mean']=d[['coding_score','problem_solving_score','data_structures_score','sql_score','machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']].mean(1)
        d['f_tech_x_pq']=d['f_tech_mean']*d['project_quality_score']/100
        d['f_interview_mean']=d[['technical_interview_score','hr_interview_score']].mean(1)
        d['f_llmpred_x_pq']=d['llm_pred']*d['project_quality_score']/100   # llm × en güçlü num
        d['f_eng_x_tech']=d['english_exam_score']*d['f_tech_mean']/100
    NEWF=['f_pq_x_ti','f_pq_x_ps','f_pq_x_cod','f_tech_mean','f_tech_x_pq','f_interview_mean','f_llmpred_x_pq','f_eng_x_tech']
    print(f"[faz33] {len(NEWF)} hedefli feature eklendi: {NEWF}")
    def mat(cols):
        Xa=tfe[cols].copy(); Xb=vfe[cols].copy()
        for c in cc:
            if c in cols: Xa[c]=tfe[c].astype('category').cat.codes; Xb[c]=vfe[c].astype('category').cat.codes
        return Xa.fillna(-999.0).values, Xb.fillna(-999.0).values
    p_te=np.clip(np.load(f'{ed}/test_pseudoR1_B_agree_pw05.npy').astype('float64'),0,100)
    tab_e=Lf('tabpfn_test.npy')
    members_t=np.column_stack([np.load(f'{ed}/test_sv2_llm.npy'),np.load(f'{ed}/test_stacker_v2.npy'),tab_e])
    mask=np.where(members_t.std(1)<=np.quantile(members_t.std(1),0.65))[0]
    pb_o=np.load(f'{ed}/oof_pseudoR1_B_agree.npy').astype('float64')
    def build_trio(feat):
        Xall,Xtall=mat(feat); sc=StandardScaler().fit(np.vstack([Xall,Xtall])); Xz,Xtz=sc.transform(Xall),sc.transform(Xtall)
        def aug(fn,Xa,Xta):
            o=np.zeros(len(y)); t=np.zeros(len(Xta)); Xp=Xta[mask]; yp=p_te[mask]
            for tr,va in folds:
                m=fn().fit(np.vstack([Xa[tr],Xp]),np.concatenate([y[tr],yp])); o[va]=m.predict(Xa[va]); t+=m.predict(Xta)/len(folds)
            return np.clip(o,0,100),np.clip(t,0,100)
        eo,et=aug(lambda:ExtraTreesRegressor(400,min_samples_leaf=5,n_jobs=-1,random_state=42),Xall,Xtall)
        ho,ht=aug(lambda:HistGradientBoostingRegressor(max_iter=600,learning_rate=0.05,max_depth=4,l2_regularization=1.0,random_state=42),Xall,Xtall)
        mo,mt=aug(lambda:MLPRegressor(hidden_layer_sizes=(128,64),alpha=1e-3,max_iter=300,early_stopping=True,random_state=42),Xz,Xtz)
        return np.mean([eo,ho,mo],0),np.mean([et,ht,mt],0)
    print("\n[BASE] mevcut feature seti...")
    a0,_=build_trio(base_feat)
    print("[+FE ] hedefli feature'lı...")
    a1,t1=build_trio(base_feat+NEWF)
    pt=p_te
    print(f"\n{'':>14}{'trio_div wOOF':>14}{'blend bw15':>12}{'bw45':>10}")
    for nm,ao in [('BASE',a0),('+FE',a1)]:
        b15=wm(np.clip(0.85*pb_o+0.15*ao,0,100)); b45=wm(np.clip(0.55*pb_o+0.45*ao,0,100))
        print(f"  {nm:<12}{wm(ao):>14.3f}{b15:>12.4f}{b45:>10.4f}")
    # +FE iyiyse CSV
    today=datetime.date.today().isoformat()
    for bw in [0.25,0.45]:
        t=np.clip((1-bw)*pt+bw*t1,0,100)
        o=np.clip((1-bw)*pb_o+bw*a1,0,100)
        pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/sub_{today}_faz33_FE_bw{int(bw*100)}_woof{wm(o):.2f}.csv',index=False)
    print(f"\n(referans: FE'siz bw15 CV=82.10, bw45 CV=82.36/public81.979)")
if __name__=='__main__': main()
