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
          "=>", "TR agirlikli" if tr > 0.3 else "EN/karisik (manuel dogrula)")

    # --- 5. Eksik değerler ---
    na = train.isna().sum(); na = na[na > 0]
    print("\n[eksik]", na.to_dict() if len(na) else "yok")

    # --- 6. Kolon envanteri ---
    # Robust string tespiti: pandas 2.x -> 'object', pandas 3.x -> 'str' (StringDtype).
    print("\n[dtype]"); print(train.dtypes.value_counts())
    def is_str_col(col):
        return (col.dtype == 'object' or pd.api.types.is_string_dtype(col)) \
            and not pd.api.types.is_numeric_dtype(col)
    cat_cols = [c for c in train.columns if c not in (ID, TEXT) and is_str_col(train[c])]
    num_cols = [c for c in train.columns if c not in (ID, TARGET, TEXT, *cat_cols)]
    print("kategorik (%d):" % len(cat_cols), cat_cols)
    print("sayisal (%d):" % len(num_cols), num_cols)

    # --- 7. Faz 0 submission ---
    # DIKKAT: sample_submission yalnizca format ornegi (cok az satir). Gercek
    # submission test_x'in TUM satirlari icin, test ID'lerinden kurulmali.
    id_col = sub.columns[0]
    tgt_col = [c for c in sub.columns if c != id_col][0]
    out = pd.DataFrame({id_col: test[id_col].values,
                        tgt_col: np.clip(float(y.mean()), 0, 100)})
    assert len(out) == len(test)
    out.to_csv('/kaggle/working/sub_phase0_constant_mean.csv', index=False)
    print("\n[Faz 0] sabit tahmin =", round(float(y.mean()), 4),
          "| satir =", len(out), "-> sub_phase0_constant_mean.csv yazildi")


if __name__ == '__main__':
    main()
