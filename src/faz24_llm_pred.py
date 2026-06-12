"""
Datathon 2026 — Faz 24: LLM DIRECT prediction'ı komboya ekle (yeni ortogonal modalite).

GİRDİ: llm_pred (Qwen2.5-7B holistik 0-100 tahmini). corr_y=0.53, mevcut üyelerle 0.55-0.70.
       grainy (13 ayrık değer) -> GBM meta-feature olarak kalibre edilir, lineer stacker'a da üye.

faz23 TABANI (sv2_llm + f13_llm, 3 LLM-feature, kombo 83.98 wOOF / public ~82.73) ÜZERİNE:
  - llm_pred'i 4. LLM GBM-feature olarak ekle (ton/gelisim/somut_basari + llm_pred) -> sv2 & f13 bazları
  - llm_pred'i sv2 nested-ridge stacker'ına 8. ÜYE olarak ekle (TabPFN benzeri, ridge kalibre eder)

VARYANTLAR (anti-overfit, sabit kombo ağırlık 0.375/0.375/0.25):
  A) faz23 referans           (3 feat, llm_pred YOK)        -> bilinen 83.98
  B) +llm_pred GBM-feature      (4 feat, sv2 8-üye değil)
  C) +llm_pred GBM-feat & stacker-üye  (tam)
En düşük wOOF + eski kombo (84.07=public 82.78) kıyas. KARAR public ile.
"""
import os, json, datetime, time
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz5_linear import run_linear_cv
from faz14_seedbag import build_featB_X
from faz16_stacker import nested_ridge_stack, RIDGE_ALPHAS
from faz22_llm_features import build_gbm_oof

LLM_FEATS = ['ton', 'gelisim', 'somut_basari']   # faz22b: overfit etmeyen güçlü alt-küme


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

    # --- LLM girdileri ---
    lf_tr = pd.read_csv(os.path.join(exp_dir, 'llm_feats_train.csv'))
    lf_te = pd.read_csv(os.path.join(exp_dir, 'llm_feats_test.csv'))
    lp_tr = pd.read_csv(os.path.join(exp_dir, 'llm_pred_train.csv'))
    lp_te = pd.read_csv(os.path.join(exp_dir, 'llm_pred_test.csv'))
    assert (train[ID].values == lf_tr[ID].values).all() and (test[ID].values == lf_te[ID].values).all()
    assert (train[ID].values == lp_tr[ID].values).all() and (test[ID].values == lp_te[ID].values).all()
    for c in LLM_FEATS:
        train_fe[f'llm_{c}'] = lf_tr[c].values.astype('float64')
        test_fe[f'llm_{c}'] = lf_te[c].values.astype('float64')
    lp_o = lp_tr['pred'].values.astype('float64'); lp_t = lp_te['pred'].values.astype('float64')
    train_fe['llm_pred'] = lp_o; test_fe['llm_pred'] = lp_t
    llm3 = [f'llm_{c}' for c in LLM_FEATS]
    llm4 = llm3 + ['llm_pred']
    num4 = num_cols + llm4   # temel sayısal + 4 LLM feature (faz23 ile aynı mantık)
    print(f"[faz24] LLM feature setleri: 3={llm3}  4=+llm_pred  | toplam num={len(num4)}")

    emb_tr = np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)); emb_te = np.load(os.path.join(exp_dir, EMB_TEST_NPY))
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    txt_tr = train[TEXT].fillna(''); txt_te = test[TEXT].fillna('')
    tab_o = np.clip(np.load(f'{exp_dir}/oof_tabpfn_train.npy').astype('float64'), 0, 100)
    tab_t = np.clip(np.load(f'{exp_dir}/tabpfn_test.npy').astype('float64'), 0, 100)

    def build_sv2(num_llm, add_pred_member):
        """faz16 no_tabpfn_ridge yapısı; GBM bazlar num_llm ile; opsiyonel llm_pred ek üye."""
        Xc, Xc_t, Xg, Xg_t, ci, feats = build_featB_X(train_fe, test_fe, txt_tr, txt_te, y,
                                                      num_llm, cat_cols, emb_tr, emb_te, bt_tr, bt_te, folds)
        oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
        ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
        ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
        tm_o, tm_e = Xc['text_meta'].values, Xc_t['text_meta'].values
        em_o, em_e = Xc['emb_meta'].values, Xc_t['emb_meta'].values
        lin_num = num_llm + ['text_meta', 'emb_meta', 'berturk_meta']
        Xl = train_fe[num_llm + cat_cols].copy(); Xl_t = test_fe[num_llm + cat_cols].copy()
        Xl['text_meta'], Xl['emb_meta'], Xl['berturk_meta'] = tm_o, em_o, bt_tr
        Xl_t['text_meta'], Xl_t['emb_meta'], Xl_t['berturk_meta'] = tm_e, em_e, bt_te
        _, _, olin, tlin = run_linear_cv(Xl, Xl_t, y, lin_num, cat_cols, folds, w)
        cols_o = [oc, ol, ox, olin, tm_o, em_o, bt_tr]
        cols_t = [tc, tl, tx, tlin, tm_e, em_e, bt_te]
        if add_pred_member:
            cols_o = cols_o + [lp_o]; cols_t = cols_t + [lp_t]
        M_o = np.column_stack(cols_o); M_t = np.column_stack(cols_t)
        so, st, sw, sa = nested_ridge_stack(M_o, M_t, y, w, folds, RIDGE_ALPHAS)
        return so, st, sw, sa, (tm_o, tm_e, em_o, em_e)

    def build_f13_5meta(num_llm, tmeta):
        tm_o, tm_e, em_o, em_e = tmeta
        meta = {'text_meta': (tm_o, tm_e), 'emb_meta': (em_o, em_e), 'berturk_meta': (bt_tr, bt_te)}
        for tag in ['bert128k', 'electra']:
            meta[f'{tag}_meta'] = (np.clip(np.load(f'{exp_dir}/oof_{tag}_train.npy').astype('float64'), 0, 100),
                                   np.clip(np.load(f'{exp_dir}/{tag}_test.npy').astype('float64'), 0, 100))
        return build_gbm_oof(train_fe, test_fe, num_llm, cat_cols, meta, y, folds)

    results = {}
    # B) llm_pred GBM-feature (4 feat), stacker ek üye YOK
    print("\n[B] 4-feat GBM (llm_pred feature), sv2 7-üye...")
    t0 = time.time()
    sv2B_o, sv2B_t, sv2B_w, aB, tmetaB = build_sv2(num4, add_pred_member=False)
    f13B_o, f13B_t, _ = build_f13_5meta(num4, tmetaB)
    kB_o = np.clip(0.375*sv2B_o + 0.375*f13B_o + 0.25*tab_o, 0, 100)
    kB_t = np.clip(0.375*sv2B_t + 0.375*f13B_t + 0.25*tab_t, 0, 100)
    results['B'] = (kB_o, kB_t, sv2B_w)
    print(f"[B] sv2={sv2B_w:.4f}(a={aB}) f13={wm(f13B_o):.4f} kombo={wm(kB_o):.4f}  {time.time()-t0:.0f}s")

    # C) 4-feat GBM + llm_pred sv2 stacker'a 8. üye
    print("\n[C] 4-feat GBM + llm_pred stacker-üye, sv2 8-üye...")
    t0 = time.time()
    sv2C_o, sv2C_t, sv2C_w, aC, tmetaC = build_sv2(num4, add_pred_member=True)
    f13C_o, f13C_t, _ = build_f13_5meta(num4, tmetaC)
    kC_o = np.clip(0.375*sv2C_o + 0.375*f13C_o + 0.25*tab_o, 0, 100)
    kC_t = np.clip(0.375*sv2C_t + 0.375*f13C_t + 0.25*tab_t, 0, 100)
    results['C'] = (kC_o, kC_t, sv2C_w)
    print(f"[C] sv2={sv2C_w:.4f}(a={aC}) f13={wm(f13C_o):.4f} kombo={wm(kC_o):.4f}  {time.time()-t0:.0f}s")

    # referanslar
    old_sv2 = np.load(f'{exp_dir}/oof_stacker_v2.npy').astype('float64')
    old_f13 = np.load(f'{exp_dir}/oof_faz13_blend.npy').astype('float64')
    old_k = np.clip(0.375*old_sv2 + 0.375*old_f13 + 0.25*tab_o, 0, 100)
    faz23_sv2 = np.load(f'{exp_dir}/oof_sv2_llm.npy').astype('float64')

    print(f"\n================ FAZ 24 SONUÇ ================")
    print(f"ESKİ kombo (llm yok)        wOOF = {wm(old_k):.4f}   (= public 82.78)")
    print(f"faz23 kombo (3-feat)        wOOF = 83.9815          (= public 82.73)")
    print(f"[B] +llm_pred feature       wOOF = {wm(results['B'][0]):.4f}")
    print(f"[C] +llm_pred feat&stacker  wOOF = {wm(results['C'][0]):.4f}")

    best = min(['B', 'C'], key=lambda k: wm(results[k][0]))
    bo, bt_pred, _ = results[best]
    print(f"\nEn iyi varyant: {best}  wOOF={wm(bo):.4f}  (faz23'e karşı delta {83.9815-wm(bo):+.4f})")
    print(f"eski-yeni kombo korelasyon: {np.corrcoef(old_k, bo)[0,1]:.4f}")

    today = datetime.date.today().isoformat()
    fn = f"sub_{today}_kombo_llmpred_{best}_woof{wm(bo):.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: bt_pred}).to_csv(os.path.join(sub_dir, fn), index=False)
    np.save(f'{exp_dir}/oof_faz24_{best}.npy', bo); np.save(f'{exp_dir}/test_faz24_{best}.npy', bt_pred)
    json.dump({'phase':'faz24_llm_pred','date':today,'corr_pred_y':0.5293,
               'B_kombo':wm(results['B'][0]),'C_kombo':wm(results['C'][0]),
               'old_kombo':wm(old_k),'faz23_kombo':83.9815,'best':best,'best_kombo':wm(bo),
               'corr_old_new':float(np.corrcoef(old_k,bo)[0,1])},
              open(f'{exp_dir}/faz24_log.json','w'), indent=2, ensure_ascii=False)
    print(f"\n[kayıt] {fn} + oof/test_faz24_{best}.npy + faz24_log.json")


if __name__ == '__main__':
    main()
