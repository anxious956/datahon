"""Keşif 1: RESIDUAL-AĞACI. Kombo (faz24-B, public 82.65) OOF residual'ında koşullu yapı avı.
In-sample ağaç -> kural keşfi (yorum). CV-ağaç -> dürüst wOOF etkisi (overfit-korumalı)."""
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds
from faz4_features import engineer

dd, ed, sd = resolve_paths()
train = pd.read_csv(f'{dd}/train.csv'); y = train[TARGET].values
oof = np.load(f'{ed}/oof_faz24_B.npy').astype('float64'); resid = y - oof
folds, _ = make_folds(train)
tep = pd.read_csv(f'{dd}/test_x.csv')[YEAR].value_counts(normalize=True)
trp = train[YEAR].value_counts(normalize=True)
w = np.nan_to_num(train[YEAR].map(lambda v: tep.get(v,0.0)/trp.get(v,np.nan)).values)
wm = lambda p: float(np.sum(w*(y-p)**2)/np.sum(w))

cat_cols = [c for c in train.columns if c not in (ID,TEXT) and is_str_col(train[c])]
tfe = engineer(train)
lf = pd.read_csv(f'{ed}/llm_feats_train.csv'); lp = pd.read_csv(f'{ed}/llm_pred_train.csv')
for c in ['ton','gelisim','somut_basari']: tfe['llm_'+c] = lf[c].values
tfe['llm_pred'] = lp['pred'].values
num = [c for c in tfe.columns if c not in (ID,TARGET,TEXT,*cat_cols)]
X = tfe[num].copy()
for c in cat_cols: X[c] = tfe[c].astype('category').cat.codes
X = X.fillna(-999.0); feats = list(X.columns)

print(f"residual: mean={resid.mean():.3f} std={resid.std():.3f}  baseline wOOF={wm(oof):.4f}")
# --- in-sample ağaç: KURAL keşfi ---
t = DecisionTreeRegressor(max_depth=4, min_samples_leaf=150, random_state=42).fit(X, resid)
print(f"\nin-sample ağaç R2(residual)={t.score(X,resid):.4f}  (>~0.03 ise sinyal olabilir)")
imp = sorted(zip(feats, t.feature_importances_), key=lambda z:-z[1])[:8]
print("top importance:", [(f,round(i,3)) for f,i in imp if i>0])
print(export_text(t, feature_names=feats, max_depth=3, decimals=2)[:1800])

# --- CV-ağaç: DÜRÜST wOOF etkisi (her fold diğer foldların residual'ıyla fit) ---
for depth in [3,4,5]:
    corr = np.zeros(len(y))
    for tr,va in folds:
        tt = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=150, random_state=42).fit(X.iloc[tr], resid[tr])
        corr[va] = tt.predict(X.iloc[va])
    for shr in [1.0, 0.5]:
        cof = np.clip(oof + shr*corr, 0, 100)
        print(f"  CV-correction depth={depth} shrink={shr}: wOOF {wm(oof):.4f} -> {wm(cof):.4f}  (delta {wm(oof)-wm(cof):+.4f})")
