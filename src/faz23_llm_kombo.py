"""
Datathon 2026 — Faz 23: LLM 3-feature'ı kombonun HER İKİ GBM üyesine işle.

KANIT: faz22b -> 3 sürekli LLM feature (ton/gelisim/somut_basari) featB-base'i +0.21 iyileştirdi
(31 feat overfit ediyordu; 3 güçlü sinyal berturk'ün yakalamadığı kalibre skoru taşıyor).
Ama kombo'da base sadece 0.375 ağırlık + sv2 hâlâ LLM'siz -> etki +0.045'e seyreliyor.

BU FAZ: kombonun İKİ GBM-türevi üyesini de LLM ile yeniden kur:
  - sv2  -> no_tabpfn_ridge (7 üye, alpha grid), GBM tabanları LLM'li  (faz16 yapısı AYNEN)
  - f13  -> faz22b 3-feat base (zaten LLM'li)
Sonra KANITLI kombo: 0.375*sv2_llm + 0.375*f13_llm + 0.25*tabpfn  (sabit ağırlık, anti-overfit).

wOOF YALNIZ referans (faz9 dersi). Karar public ile. Eski kombo (84.07=public 82.78) kalibrasyon.
"""
import os, json, datetime, time
import numpy as np
import pandas as pd

from faz1_catboost_baseline import resolve_paths, is_str_col, ID, TARGET, TEXT, YEAR
from faz2_text_meta import make_folds, run_catboost_cv, metrics
from faz3_ensemble import run_lgbm_cv, run_xgb_cv, EMB_TRAIN_NPY, EMB_TEST_NPY
from faz4_features import engineer
from faz5_linear import run_linear_cv
from faz14_seedbag import build_featB_X
from faz16_stacker import nested_ridge_stack, RIDGE_ALPHAS
from faz22_llm_features import build_gbm_oof

LLM_SUB = ['ton', 'gelisim', 'somut_basari']   # faz22b: en güçlü, overfit etmeyen alt-küme


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

    # --- LLM 3-feature'ı FE matrisine işle ---
    lf_tr = pd.read_csv(os.path.join(exp_dir, 'llm_feats_train.csv'))
    lf_te = pd.read_csv(os.path.join(exp_dir, 'llm_feats_test.csv'))
    assert (train[ID].values == lf_tr[ID].values).all() and (test[ID].values == lf_te[ID].values).all()
    llm_num = [f'llm_{c}' for c in LLM_SUB]
    for c in LLM_SUB:
        train_fe[f'llm_{c}'] = lf_tr[c].values.astype('float64')
        test_fe[f'llm_{c}'] = lf_te[c].values.astype('float64')
    num_llm = num_cols + llm_num
    print(f"[faz23] LLM feature: {llm_num}")

    emb_tr = np.load(os.path.join(exp_dir, EMB_TRAIN_NPY)); emb_te = np.load(os.path.join(exp_dir, EMB_TEST_NPY))
    bt_tr = np.load(os.path.join(exp_dir, 'oof_berturk_train.npy')).astype('float64')
    bt_te = np.load(os.path.join(exp_dir, 'berturk_test.npy')).astype('float64')
    txt_tr = train[TEXT].fillna(''); txt_te = test[TEXT].fillna('')

    # ============ 1) sv2_llm: faz16 no_tabpfn_ridge yapısı, GBM tabanları LLM'li ============
    print("[faz23] sv2_llm: featB GBM tabanları (LLM'li) + lineer + metalar -> nested ridge")
    t0 = time.time()
    Xc, Xc_t, Xg, Xg_t, ci, feats = build_featB_X(train_fe, test_fe, txt_tr, txt_te, y,
                                                  num_llm, cat_cols, emb_tr, emb_te, bt_tr, bt_te, folds)
    oc, tc, _, _ = run_catboost_cv(Xc, y, Xc_t, ci, folds)
    ol, tl, _ = run_lgbm_cv(Xg, y, Xg_t, cat_cols, folds)
    ox, tx, _ = run_xgb_cv(Xg, y, Xg_t, folds)
    tm_tr, tm_te = Xc['text_meta'].values, Xc_t['text_meta'].values
    em_tr, em_te = Xc['emb_meta'].values, Xc_t['emb_meta'].values
    lin_num = num_llm + ['text_meta', 'emb_meta', 'berturk_meta']
    Xl = train_fe[num_llm + cat_cols].copy(); Xl_t = test_fe[num_llm + cat_cols].copy()
    Xl['text_meta'], Xl['emb_meta'], Xl['berturk_meta'] = tm_tr, em_tr, bt_tr
    Xl_t['text_meta'], Xl_t['emb_meta'], Xl_t['berturk_meta'] = tm_te, em_te, bt_te
    _, _, olin, tlin = run_linear_cv(Xl, Xl_t, y, lin_num, cat_cols, folds, w)
    M_oof = np.column_stack([oc, ol, ox, olin, tm_tr, em_tr, bt_tr])
    M_test = np.column_stack([tc, tl, tx, tlin, tm_te, em_te, bt_te])
    sv2_o, sv2_t, sv2_w, sv2_a = nested_ridge_stack(M_oof, M_test, y, w, folds, RIDGE_ALPHAS)
    print(f"[faz23] sv2_llm wOOF={sv2_w:.4f} (alpha={sv2_a}) | eski sv2 referans 85.7384  {time.time()-t0:.0f}s")

    # ============ 2) f13_llm: faz22b 3-feat base (5-meta) ============
    print("[faz23] f13_llm: 5-meta base (LLM'li, faz22b)")
    meta = {'text_meta': (tm_tr, tm_te), 'emb_meta': (em_tr, em_te), 'berturk_meta': (bt_tr, bt_te)}
    for tag in ['bert128k', 'electra']:
        meta[f'{tag}_meta'] = (np.clip(np.load(f'{exp_dir}/oof_{tag}_train.npy').astype('float64'), 0, 100),
                               np.clip(np.load(f'{exp_dir}/{tag}_test.npy').astype('float64'), 0, 100))
    f13_o, f13_t, _ = build_gbm_oof(train_fe, test_fe, num_llm, cat_cols, meta, y, folds)
    print(f"[faz23] f13_llm wOOF={wm(f13_o):.4f} | eski f13 referans 85.85")

    # ============ 3) kombo = 0.375*sv2_llm + 0.375*f13_llm + 0.25*tabpfn ============
    tab_o = np.clip(np.load(f'{exp_dir}/oof_tabpfn_train.npy').astype('float64'), 0, 100)
    tab_t = np.clip(np.load(f'{exp_dir}/tabpfn_test.npy').astype('float64'), 0, 100)
    # eski kombo (kalibrasyon: public 82.78)
    old_sv2 = np.load(f'{exp_dir}/oof_stacker_v2.npy').astype('float64')
    old_f13 = np.load(f'{exp_dir}/oof_faz13_blend.npy').astype('float64')
    old_k = np.clip(0.375*old_sv2 + 0.375*old_f13 + 0.25*tab_o, 0, 100)
    new_o = np.clip(0.375*sv2_o + 0.375*f13_o + 0.25*tab_o, 0, 100)
    new_t = np.clip(0.375*sv2_t + 0.375*f13_t + 0.25*tab_t, 0, 100)

    print(f"\n================ FAZ 23 SONUÇ ================")
    print(f"ESKİ kombo wOOF   = {wm(old_k):.4f}   (= public 82.78 kalibrasyon)")
    print(f"YENİ kombo (2x-llm) = {wm(new_o):.4f}   (delta {wm(old_k)-wm(new_o):+.4f})")
    print(f"  [faz22b tek-üye-llm kombo referans: 84.0264, delta +0.045]")
    print(f"eski-yeni kombo korelasyon: {np.corrcoef(old_k, new_o)[0,1]:.4f}")

    today = datetime.date.today().isoformat()
    fn = f"sub_{today}_kombo_llm2x_woof{wm(new_o):.2f}.csv"
    pd.DataFrame({ID: test[ID].values, TARGET: new_t}).to_csv(os.path.join(sub_dir, fn), index=False)
    np.save(f'{exp_dir}/oof_sv2_llm.npy', sv2_o); np.save(f'{exp_dir}/test_sv2_llm.npy', sv2_t)
    json.dump({'phase':'faz23_llm_kombo','date':today,'llm_sub':LLM_SUB,
               'sv2_llm_woof':sv2_w,'sv2_alpha':sv2_a,'f13_llm_woof':wm(f13_o),
               'old_kombo_woof':wm(old_k),'new_kombo_woof':wm(new_o),
               'corr_old_new':float(np.corrcoef(old_k,new_o)[0,1])},
              open(f'{exp_dir}/faz23_log.json','w'), indent=2, ensure_ascii=False)
    print(f"\n[kayıt] {fn} + oof/test_sv2_llm.npy + faz23_log.json")


if __name__ == '__main__':
    main()
