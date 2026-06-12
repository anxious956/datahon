"""Keşif 2: BAND-SINIFLANDIRMA. Metin (e5 emb + tfidf) hedef bandını ne kadar biliyor?
Yüksek accuracy -> metin hedefi yapılandırılmış kodluyor. Olasılık×merkez -> stacker üyesi, wOOF."""
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from faz1_catboost_baseline import resolve_paths, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds

dd, ed, sd = resolve_paths()
train = pd.read_csv(f'{dd}/train.csv'); y = train[TARGET].values
folds, _ = make_folds(train)
tep = pd.read_csv(f'{dd}/test_x.csv')[YEAR].value_counts(normalize=True)
trp = train[YEAR].value_counts(normalize=True)
w = np.nan_to_num(train[YEAR].map(lambda v: tep.get(v,0.0)/trp.get(v,np.nan)).values)
wm = lambda p: float(np.sum(w*(y-p)**2)/np.sum(w))

band = np.clip((y//10).astype(int), 0, 9)
nb = 10; centers = np.array([5+10*i for i in range(nb)], dtype='float64')
print("band dağılımı:", np.bincount(band, minlength=nb))
base_acc = np.bincount(band).max()/len(band)
print(f"baseline (çoğunluk) acc = {base_acc:.4f}")

# metin temsili: e5 emb + char/word tfidf
emb = np.load(f'{ed}/emb_e5_train.npy').astype('float32')
txt = train[TEXT].fillna('').values
vw = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=5, max_features=20000)
vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=5, max_features=20000)
Xw = vw.fit_transform(txt); Xc = vc.fit_transform(txt)
X = hstack([csr_matrix(emb), Xw, Xc]).tocsr()
print(f"feature boyut: emb{emb.shape[1]} + word{Xw.shape[1]} + char{Xc.shape[1]} = {X.shape[1]}")

oof_proba = np.zeros((len(y), nb)); oof_pred_band = np.zeros(len(y), dtype=int)
for tr,va in folds:
    clf = LGBMClassifier(objective='multiclass', num_class=nb, n_estimators=300, learning_rate=0.05,
                         num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
                         reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1)
    clf.fit(X[tr], band[tr])
    oof_proba[va] = clf.predict_proba(X[va]); oof_pred_band[va] = oof_proba[va].argmax(1)
acc = (oof_pred_band==band).mean()
within1 = (np.abs(oof_pred_band-band)<=1).mean()
print(f"\nband-accuracy = {acc:.4f}  (baseline {base_acc:.4f}, lift {acc-base_acc:+.4f})")
print(f"within-1-band = {within1:.4f}")
cm = pd.crosstab(band, oof_pred_band, rownames=['gerçek'], colnames=['tahmin'])
print("confusion:\n", cm.to_string())

# olasılık × merkez -> regresyon tahmini
band_pred = np.clip(oof_proba @ centers, 0, 100)
print(f"\nband-tahmin corr_y = {np.corrcoef(band_pred,y)[0,1]:.4f}  wOOF(saf)={wm(band_pred):.4f}")
kombo = np.load(f'{ed}/oof_faz24_B.npy').astype('float64')
print(f"corr(band_pred, kombo) = {np.corrcoef(band_pred,kombo)[0,1]:.4f}  (düşükse ortogonal)")
for bw in [0.1,0.2,0.3]:
    bl = np.clip((1-bw)*kombo + bw*band_pred, 0, 100)
    print(f"  blend kombo+{bw}*band: wOOF {wm(kombo):.4f} -> {wm(bl):.4f}  (delta {wm(kombo)-wm(bl):+.4f})")
np.save(f'{ed}/oof_bandpred_train.npy', band_pred)
