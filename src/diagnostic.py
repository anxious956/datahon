"""
Datathon 2026 — Faz 0 teşhis scripti (CONTEXT.md §9 genişletilmiş).

Kaggle'da kullanım: yeni bir notebook hücresine bu dosyanın içeriğini yapıştır
veya `notebooks/00_diagnostic.ipynb`'i doğrudan Kaggle'a yükle.

Amaç: model eğitmeden veriyi tanımak ve §8'deki bilinmeyenleri teyit etmek:
  1. /kaggle/input klasör adı
  2. satır/kolon sayısı + dtype dağılımı
  3. mentor_feedback_text dili (TR/EN/karışık)
  4. career_success_score dağılımı
  5. sample_submission birebir yapısı
  6. eksik değerler (özellikle metinde)
Sonunda Faz 0 teslimatı olan sabit-ortalama submission'ı üretir.
"""
import pandas as pd, numpy as np, glob, os

ID, TARGET, TEXT = 'student_id', 'career_success_score', 'mentor_feedback_text'


def main():
    # --- 1. Klasörü bul ve yükle ---
    base = glob.glob('/kaggle/input/*')[0]
    print("input klasörü:", base, os.listdir(base))
    train = pd.read_csv(f'{base}/train.csv')
    test = pd.read_csv(f'{base}/test_x.csv')   # DİKKAT: test.csv değil
    sub = pd.read_csv(f'{base}/sample_submission.csv')
    print("train:", train.shape, "| test_x:", test.shape, "| sub:", sub.shape)

    # --- 2. sample_submission yapısı ---
    print("\n[sample_submission]", list(sub.columns)); print(sub.head())

    # --- 3. Hedef dağılımı ---
    y = train[TARGET]
    print("\n[hedef]"); print(y.describe())
    print("skew:", round(y.skew(), 4), "| var(pop):", round(y.var(ddof=0), 4),
          "<- sabit-ortalama submission'ın beklenen public MSE'si")

    # --- 4. Metin dili ---
    txt = train[TEXT]
    print("\n[metin] NaN:", int(txt.isna().sum()), "| örnekler:")
    for t in txt.dropna().head(3).tolist():
        print("  -", t)
    s = txt.dropna().head(2000).str.lower()
    tr = s.str.contains(r'[çğışöü]', regex=True).mean()
    print("TR'ye özgü karakter oranı:", round(float(tr), 3),
          "=>", "TR ağırlıklı" if tr > 0.3 else "EN/karışık (manuel doğrula)")

    # --- 5. Eksik değerler ---
    na = train.isna().sum(); na = na[na > 0]
    print("\n[eksik]", na.to_dict() if len(na) else "yok")

    # --- 6. Kolon envanteri ---
    print("\n[dtype]"); print(train.dtypes.value_counts())
    cat_cols = [c for c in train.columns if train[c].dtype == 'object' and c not in (ID, TEXT)]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    print("kategorik (%d):" % len(cat_cols), cat_cols)
    print("sayısal (%d):" % len(num_cols), num_cols)

    # --- 7. Faz 0 submission ---
    id_col = sub.columns[0]
    tgt_col = [c for c in sub.columns if c != id_col][0]
    out = sub.copy()
    out[id_col] = test[id_col].values if id_col in test.columns else sub[id_col].values
    out[tgt_col] = np.clip(float(y.mean()), 0, 100)
    out.to_csv('/kaggle/working/sub_phase0_constant_mean.csv', index=False)
    print("\n[Faz 0] sabit tahmin =", round(float(y.mean()), 4),
          "-> sub_phase0_constant_mean.csv yazıldı")


if __name__ == '__main__':
    main()
