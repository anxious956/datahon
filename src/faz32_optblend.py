"""
Faz 32: RIGOROUS OPTIMAL BLEND. Tüm güçlü sinyalleri nested-CV non-negatif ağırlıkla birleştir.
El-ayarı 2'li blend yerine VERİDEN öğrenilen ağırlık; nested-CV overfit'i engeller (private-sağlam).
Ağırlıklar train-OOF'ta optimize -> diversity'nin test-fit avantajını GÖRMEZ -> doğal olarak
KONSERVATİF/robust blend seçer = private için ideal. Çıktı: en iyi nested-wOOF blend CSV.
"""
import os, datetime
import numpy as np, pandas as pd
from scipy.optimize import nnls
from faz1_catboost_baseline import resolve_paths, ID, TARGET, YEAR
from faz2_text_meta import make_folds

dd,ed,sd=resolve_paths()
train=pd.read_csv(f'{dd}/train.csv'); test=pd.read_csv(f'{dd}/test_x.csv'); y=train[TARGET].values
folds,_=make_folds(train)
trp=train[YEAR].value_counts(normalize=True); tep=test[YEAR].value_counts(normalize=True)
w=np.nan_to_num(train[YEAR].map(lambda v:tep.get(v,0.0)/trp.get(v,np.nan)).values)
wm=lambda p:float(np.sum(w*(y-p)**2)/np.sum(w))

# (isim, oof_dosya, test_dosya) -- her ikisi de olan güçlü sinyaller
S=[('pseudoB','oof_pseudoR1_B_agree.npy','test_pseudoR1_B_agree_pw05.npy'),
   ('sv2','oof_stacker_v2.npy','test_stacker_v2.npy'),
   ('sv2_llm','oof_sv2_llm.npy','test_sv2_llm.npy'),
   ('faz24B','oof_faz24_B.npy','test_faz24_B.npy'),
   ('tabpfn','oof_tabpfn_train.npy','tabpfn_test.npy'),
   ('trio_div','oof_trio_div.npy','test_trio_div.npy'),
   ('superdiv','oof_superdiv.npy','test_superdiv.npy'),
   ('berturk','oof_berturk_train.npy','berturk_test.npy'),
   ('electra','oof_electra_train.npy','electra_test.npy'),
   ('bert128k','oof_bert128k_train.npy','bert128k_test.npy')]
O=[];T=[];names=[]
for nm,of,tf in S:
    op=f'{ed}/{of}'; tp=f'{ed}/{tf}'
    if os.path.exists(op) and os.path.exists(tp):
        O.append(np.clip(np.load(op).astype('float64'),0,100))
        T.append(np.clip(np.load(tp).astype('float64'),0,100)); names.append(nm)
O=np.column_stack(O); T=np.column_stack(T)
print(f"[faz32] {len(names)} sinyal: {names}")
print("tekil wOOF:", {names[i]:round(wm(O[:,i]),2) for i in range(len(names))})

# ağırlıklı NNLS: sqrt(w) ile ölçekle -> weighted LS
sw=np.sqrt(w)
def fit_nnls(Oo,yy,ww):
    A=Oo*ww[:,None]; b=yy*ww
    c,_=nnls(A,b);
    return c
# nested-CV: her fold ağırlık diğer foldlarda fit, held-out tahmin
nested=np.zeros(len(y))
for tr,va in folds:
    c=fit_nnls(O[tr],y[tr],sw[tr])
    nested[va]=O[va]@c
nested=np.clip(nested,0,100)
print(f"\n[faz32] NNLS nested-CV wOOF = {wm(nested):.4f}")
print(f"   (referans: trio_bw45 CV 82.36 / public 81.979, bw15 CV 82.10)")
# final ağırlık (tüm OOF) + test
c=fit_nnls(O,y,sw); c=c/c.sum() if c.sum()>0 else c
print("optimal ağırlıklar:", {names[i]:round(c[i],3) for i in range(len(names)) if c[i]>0.01})
test_blend=np.clip(T@c,0,100)
# nested wOOF iyiyse kaydet
today=datetime.date.today().isoformat()
fn=f"sub_{today}_faz32_optblend_woof{wm(nested):.2f}.csv"
pd.DataFrame({ID:test[ID].values,TARGET:test_blend}).to_csv(f'{sd}/{fn}',index=False)
np.save(f'{ed}/oof_optblend.npy',nested); np.save(f'{ed}/test_optblend.npy',test_blend)
print(f"\n[kayıt] {fn}")
# ayrıca: bu opt-blend ile trio_div'i hafif blend (test-fit upside) — birkaç bw
ao=np.load(f'{ed}/oof_trio_div.npy'); at=np.load(f'{ed}/test_trio_div.npy')
print("\nopt-blend + trio_div (test-fit upside):")
for bw in [0.0,0.1,0.2,0.3]:
    o=np.clip((1-bw)*nested+bw*ao,0,100); t=np.clip((1-bw)*test_blend+bw*at,0,100)
    print(f"  bw={bw}: nestedwOOF={wm(o):.4f}")
    if bw>0:
        pd.DataFrame({ID:test[ID].values,TARGET:t}).to_csv(f'{sd}/sub_{today}_faz32_optblend_div{int(bw*100)}_woof{wm(o):.2f}.csv',index=False)
