"""
Datathon 2026 — Faz 12: ÖĞRENİLMİŞ STACKER (en riskli, son adım).

FİKİR: Basit ağırlık-blend yerine, tüm level-0 OOF'lar üstünde öğrenilmiş birleştirici
(Ridge). Girdiler (her biri tr-OOF + eşleşen test vektörü):
  cat_tuned, lgb_tuned, xgb_tuned (Faz 9) + text_meta + emb_meta + berturk_meta + lin
  (+ varsa yeni transformer OOF'ları: env TRANSFORMER_OOFS='xlmr,electra,...'; her tag için
   experiments/oof_<tag>_train.npy + <tag>_test.npy beklenir).

⚠️ OOF-ŞİŞME RİSKİ YÜKSEK (embedding dersi: emb wOOF -1.08 dedi, public yalnız -0.33).
   ÖNLEM: NESTED CV — stacker katsayıları yalnız fold-k-train OOF'larında öğrenilir, fold-k-val'a
   uygulanır (level-0 değerleri zaten fold-dışı). Yine de kazancı DİSKONTO et, PUBLIC HAKEMİ.

GİRDİ: önce faz9 koş (oof/test_*_tuned.npy). ÇIKTI: oof_stack.npy, test_stack.npy,
       faz12_stacker_log.json, submission. Basit blend'i (faz9) public'te geçerse TUT.
"""
import os, json, datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from faz1_catboost_baseline import ID, TARGET, YEAR
from faz2_text_meta import metrics
from faz9_tune import prepare

ALPHAS = [0.1, 0.3, 1.0, 3.0, 10.0]
TRANSFORMER_OOFS = [t for t in os.environ.get('TRANSFORMER_OOFS', '').split(',') if t]


def linear_meta(num_tr, num_te, y, folds):
    """Sayısal-only Ridge level-0 (median impute + standardize, leakage-free OOF + test)."""
    oof = np.zeros(len(num_tr)); test = np.zeros(len(num_te))
    for tr, va in folds:
        med = np.nanmedian(num_tr[tr], axis=0)
        Xtr = np.where(np.isnan(num_tr[tr]), med, num_tr[tr])
        Xva = np.where(np.isnan(num_tr[va]), med, num_tr[va])
        Xte = np.where(np.isnan(num_te), med, num_te)
        sc = StandardScaler().fit(Xtr)
        r = Ridge(alpha=10.0).fit(sc.transform(Xtr), y[tr])
        oof[va] = r.predict(sc.transform(Xva)); test += r.predict(sc.transform(Xte)) / len(folds)
    return np.clip(oof, 0, 100), np.clip(test, 0, 100)


def stack_oof(Z_tr, y, folds, alpha):
    """Nested stacker OOF: Ridge yalnız fold-k-train'de fit, fold-k-val'a uygulanır."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        sc = StandardScaler().fit(Z_tr[tr])
        r = Ridge(alpha=alpha).fit(sc.transform(Z_tr[tr]), y[tr])
        oof[va] = r.predict(sc.transform(Z_tr[va]))
    return np.clip(oof, 0, 100)


def stack_fit_test(Z_tr, y, Z_te, alpha):
    sc = StandardScaler().fit(Z_tr)
    r = Ridge(alpha=alpha).fit(sc.transform(Z_tr), y)
    return np.clip(r.predict(sc.transform(Z_te)), 0, 100), dict(zip(LEVEL0_NAMES, r.coef_))


LEVEL0_NAMES = []  # main() doldurur (log için)


def main():
    global LEVEL0_NAMES
    P = prepare()
    y, w, years, te_prop, folds = P['y'], P['w'], P['years'], P['te_prop'], P['folds']
    cat_cols = P['cat_cols']; Xg, Xg_t = P['Xg'], P['Xg_t']
    exp_dir, sub_dir, test, meta = P['exp_dir'], P['sub_dir'], P['test'], P['meta']

    feats, tests, names = [], [], []

    def add(name, tr_vec, te_vec):
        feats.append(np.asarray(tr_vec, float)); tests.append(np.asarray(te_vec, float)); names.append(name)

    # 1) Tuned GBM'ler (Faz 9)
    for k in ('cat', 'lgb', 'xgb'):
        op = os.path.join(exp_dir, f'oof_{k}_tuned.npy'); tp = os.path.join(exp_dir, f'test_{k}_tuned.npy')
        if os.path.exists(op) and os.path.exists(tp):
            add(f'{k}_tuned', np.load(op), np.load(tp))
        else:
            print(f"[uyarı] {k}_tuned bulunamadı — önce faz9 koş. Atlanıyor.")
    # 2) Meta sinyaller (her biri tr+te hazır)
    for m, (a, b) in meta.items():
        add(m, a, b)
    # 3) Lineer (sayısal-only Ridge, inline)
    meta_names = list(meta)
    num_only = [c for c in Xg.columns if c not in cat_cols and c not in meta_names]
    num_tr = Xg[num_only].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    num_te = Xg_t[num_only].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    lin_oof, lin_te = linear_meta(num_tr, num_te, y, folds)
    add('lin', lin_oof, lin_te)
    # 4) Yeni transformer OOF'ları (varsa)
    for t in TRANSFORMER_OOFS:
        op = os.path.join(exp_dir, f'oof_{t}_train.npy'); tp = os.path.join(exp_dir, f'{t}_test.npy')
        if os.path.exists(op) and os.path.exists(tp):
            add(t, np.load(op).astype(float), np.load(tp).astype(float))
            print(f"[stacker] transformer eklendi: {t}")
        else:
            print(f"[uyarı] transformer '{t}' OOF/test bulunamadı — atlanıyor.")

    Z_tr = np.column_stack(feats); Z_te = np.column_stack(tests); LEVEL0_NAMES = names
    print(f"[stacker] {len(names)} level-0: {names}")

    # --- alpha seçimi (küçük grid; wOOF ile, kazanç diskonto edilir) ---
    best = None
    for a in ALPHAS:
        oof = stack_oof(Z_tr, y, folds, a)
        _, wm, _ = metrics(y, oof, w, years, te_prop)
        print(f"  alpha={a:<5}: wOOF={wm:.4f}")
        if best is None or wm < best[0]:
            best = (wm, a, oof)
    s_w, alpha, s_oof = best
    s_plain, s_w, s_by = metrics(y, s_oof, w, years, te_prop)
    s_te, coefs = stack_fit_test(Z_tr, y, Z_te, alpha)

    # --- Basit blend referansı (Faz 9) ---
    ref_p = os.path.join(exp_dir, 'oof_blend_tuned.npy')
    ref_w = None
    if os.path.exists(ref_p):
        _, ref_w, _ = metrics(y, np.load(ref_p), w, years, te_prop)

    print("\n================ FAZ 12 SONUÇ (stacker, ağırlıklı OOF) ================")
    print(f"En iyi Ridge alpha={alpha} | stacker wOOF={s_w:.4f} (düz {s_plain:.4f})")
    if ref_w is not None:
        print(f"Basit blend (Faz 9) wOOF={ref_w:.4f} | stacker kazancı {ref_w - s_w:+.4f} "
              f"⚠️ OOF şişebilir → PUBLIC HAKEMİ (diskonto et)")
    print("Level-0 katsayıları:", {k: round(v, 3) for k, v in coefs.items()})
    print("\nYıl-bazlı stacker OOF MSE:")
    print(s_by.round(3).to_string())

    np.save(os.path.join(exp_dir, 'oof_stack.npy'), s_oof)
    np.save(os.path.join(exp_dir, 'test_stack.npy'), s_te)
    log = {'phase': 'faz12_stacker', 'date': datetime.date.today().isoformat(),
           'level0': names, 'best_alpha': alpha, 'coefs': {k: float(v) for k, v in coefs.items()},
           'stacker': {'plain': s_plain, 'weighted': s_w, 'by_year': s_by.round(4).to_dict()},
           'ref_simple_blend_weighted': ref_w,
           'gain_vs_blend': (ref_w - s_w) if ref_w is not None else None,
           'WARN': 'OOF inflation risk high; discount gain, validate on public.'}
    with open(os.path.join(exp_dir, 'faz12_stacker_log.json'), 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    sub = pd.DataFrame({ID: test[ID].values, TARGET: s_te})
    fname = f"sub_{datetime.date.today().isoformat()}_stacker_woof{s_w:.2f}.csv"
    sub.to_csv(os.path.join(sub_dir, fname), index=False)
    print(f"\n[kayıt] oof_stack.npy + test_stack.npy + faz12_stacker_log.json + {fname}")


if __name__ == '__main__':
    main()
