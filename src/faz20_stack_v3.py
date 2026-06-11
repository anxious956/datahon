"""
Datathon 2026 — Faz 20: STACKER v3 (5-meta) + TabPFN sabit karışım = kombo_v2.

KANIT ZİNCİRİ: featB+yeni metalar (faz13v2) -> kombo public 82.78 (+0.37). Ama karışımdaki
sv2 hâlâ ESKİ metalarla (text/emb/berturk) kurulu. Bu script sv2'nin 5-üye nested-Ridge
yapısını YENİ metalarla (bert128k_meta + electra_meta dahil) yeniden kurar -> stack_v3.
Sonra kanıtlı tarif: 0.75*stack_v3 + 0.25*tabpfn = kombo_v2 adayı.

Anti-overfit: nested ridge (faz16 ile aynı, sv2 translate etmişti), TabPFN ağırlığı SABİT 0.25
(öğrenilmiş değil — faz17 dersi), karar public ile.
"""
import os, json, datetime, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz2_2_embed import build_charword_meta, build_emb_meta
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz5_linear import run_linear_cv

RIDGE_ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
TAB_W = 0.25   # kanıtlı sabit TabPFN ağırlığı (public çanağı dibi)


def nested_ridge(M_oof, M_test, y, w, folds, alphas):
    best = None
    for a in alphas:
        no = np.zeros(len(y))
        for tr, va in folds:
            no[va] = Ridge(alpha=a, random_state=42).fit(M_oof[tr], y[tr]).predict(M_oof[va])
        no = np.clip(no, 0, 100)
        wt = float(np.sum(w*(y-no)**2)/np.sum(w))
        if best is None or wt < best[0]: best = (wt, a, no)
    wt, a, no = best
    te = np.clip(Ridge(alpha=a, random_state=42).fit(M_oof, y).predict(M_test), 0, 100)
    return no, te, wt, a


def main():
    data_dir, exp_dir, sub_dir = resolve_paths()
    train = pd.read_csv(f'{data_dir}/train.csv'); test = pd.read_csv(f'{data_dir}/test_x.csv')
    y = train[TARGET].values
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    raw_num = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    train_fe, test_fe = engineer(train), engineer(test)
    eng_cols = [c for c in train_fe.columns if c not in train.columns]
    num_cols = raw_num + eng_cols
    folds, _ = make_folds(train)
    trp = train[YEAR].value_counts(normalize=True); tep = test[YEAR].value_counts(normalize=True)
    w = np.nan_to_num(train[YEAR].map(lambda v: tep.get(v, 0.0) / trp.get(v, np.nan)).values)
    def wm(p): return float(np.sum(w*(y-p)**2)/np.sum(w))

    # ---------- 5 meta (text/emb/berturk + YENİ bert128k/electra) ----------
    tm_tr, tm_te = build_charword_meta(train[TEXT].fillna(''), test[TEXT].fillna(''), y, folds)
    em_tr, em_te = build_emb_meta(np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)),
                                  np.load(os.path.join(exp_dir, EMB_TEST_NPY)), y, folds)
    metas = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te)}
    for tag, of, tf in [('berturk_meta','oof_berturk_train.npy','berturk_test.npy'),
                        ('bert128k_meta','oof_bert128k_train.npy','bert128k_test.npy'),
                        ('electra_meta','oof_electra_train.npy','electra_test.npy')]:
        metas[tag] = (np.clip(np.load(os.path.join(exp_dir, of)).astype('float64'),0,100),
                      np.clip(np.load(os.path.join(exp_dir, tf)).astype('float64'),0,100))
    print(f"[v3] metalar: {list(metas)}")

    # ---------- GBM tabanlar (5-meta feature seti, default paramlar) ----------
    cols = num_cols + cat_cols
    Xc = train_fe[cols].copy(); Xc_t = test_fe[cols].copy()
    for c in cat_cols: Xc[c] = Xc[c].astype(str); Xc_t[c] = Xc_t[c].astype(str)
    Xg = train_fe[cols].copy(); Xg_t = test_fe[cols].copy()
    for c in cat_cols:
        cc = pd.CategoricalDtype(categories=sorted(set(Xg[c].dropna()) | set(Xg_t[c].dropna())))
        Xg[c] = Xg[c].astype(cc); Xg_t[c] = Xg_t[c].astype(cc)
    for m, (a, b) in metas.items():
        Xc[m] = a; Xc_t[m] = b; Xg[m] = a; Xg_t[m] = b
    feats = cols + list(metas); ci = [feats.index(c) for c in cat_cols]
    t0 = time.time()
    oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
    print(f"[v3] GBM tabanlar {time.time()-t0:.0f}s | cat {wm(oc):.3f} lgb {wm(ol):.3f} xgb {wm(ox):.3f}")

    # ---------- lineer üye (5-meta) ----------
    lin_num = num_cols + list(metas)
    Xl = train_fe[num_cols + cat_cols].copy(); Xl_t = test_fe[num_cols + cat_cols].copy()
    for m, (a, b) in metas.items(): Xl[m] = a; Xl_t[m] = b
    lwoof, lalpha, olin, tlin = run_linear_cv(Xl, Xl_t, y, lin_num, cat_cols, folds, w)
    print(f"[v3] lineer wOOF={lwoof:.3f}")

    # ---------- nested-Ridge stack (9 üye: 3 GBM + lin + 5 meta) ----------
    names = ['cat','lgb','xgb','lin'] + list(metas)
    M_oof = np.column_stack([oc, ol, ox, olin] + [metas[m][0] for m in metas])
    M_test = np.column_stack([tc, tl, tx, tlin] + [metas[m][1] for m in metas])
    so, st, swt, sa = nested_ridge(M_oof, M_test, y, w, folds, RIDGE_ALPHAS)
    print(f"[v3] stack_v3 nested wOOF={swt:.4f} (alpha={sa}) | sv2 referans 85.74")

    # ---------- kombo_v2 = 0.75*stack_v3 + 0.25*tabpfn ----------
    tab_o = np.clip(np.load(os.path.join(exp_dir,'oof_tabpfn_train.npy')).astype('float64'),0,100)
    tab_t = np.clip(np.load(os.path.join(exp_dir,'tabpfn_test.npy')).astype('float64'),0,100)
    k_o = np.clip((1-TAB_W)*so + TAB_W*tab_o, 0, 100)
    k_t = np.clip((1-TAB_W)*st + TAB_W*tab_t, 0, 100)
    kwt = wm(k_o)
    # referans: mevcut kombo (kanıtlı 82.78) wOOF'u
    sv2_o = np.load(os.path.join(exp_dir,'oof_stacker_v2.npy')).astype('float64')
    f13_o = np.load(os.path.join(exp_dir,'oof_faz13_blend.npy')).astype('float64')
    old_k = np.clip(0.375*sv2_o + 0.375*f13_o + 0.25*tab_o, 0, 100)
    print(f"\n================ FAZ 20 SONUÇ ================")
    print(f"ESKİ kombo wOOF = {wm(old_k):.4f}  (= public 82.78, kalibrasyon)")
    print(f"YENİ kombo_v2   = {kwt:.4f}  (delta {wm(old_k)-kwt:+.4f})")

    np.save(os.path.join(exp_dir,'oof_stack_v3.npy'), so)
    np.save(os.path.join(exp_dir,'test_stack_v3.npy'), st)
    fn = f"sub_{datetime.date.today().isoformat()}_kombo_v2_stackv3_tabpfn25.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: k_t}).to_csv(os.path.join(sub_dir, fn), index=False)
    log = {'phase':'faz20_stack_v3','date':datetime.date.today().isoformat(),'members':names,
           'stack_v3_woof':swt,'alpha':sa,'kombo_v2_woof':kwt,'old_kombo_woof':wm(old_k)}
    json.dump(log, open(os.path.join(exp_dir,'faz20_log.json'),'w'), indent=2, ensure_ascii=False)
    print(f"[kayıt] oof/test_stack_v3.npy + faz20_log.json + {fn}")


if __name__ == '__main__':
    main()
