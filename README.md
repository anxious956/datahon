# Datathon 2026 — Career Success Score Prediction

BTK Akademi Datathon 2026 yarışması için çalışma reposu. Öğrenci profillerinden
**`career_success_score`** (0–100, sürekli) tahmini. Regresyon, metrik **MSE**.

## Başlangıç (Claude Code)

1. Önce **[`CONTEXT.md`](./CONTEXT.md)** dosyasını oku — problem tanımı, kolonlar,
   kurallar, leaderboard durumu ve pipeline planının tamamı orada.
2. Veri Kaggle'da bağlı: `/kaggle/input/<klasör>/` → `train.csv`, `test_x.csv`,
   `sample_submission.csv`. (Not: test dosyasının adı `test_x.csv`.)
3. İlk iş `CONTEXT.md` §9'daki teşhis kodunu çalıştır ve §8'deki bilinmeyenleri teyit et.

## Önerilen yapı

```
.
├── CONTEXT.md              # tüm yarışma bağlamı (önce bunu oku)
├── README.md
├── notebooks/              # Kaggle / keşif notebookları
├── src/                    # pipeline kodu (data, features, model, cv)
├── experiments/            # OOF skorları, deney logları
└── submissions/            # sub_<tarih>_<model>_oof<skor>.csv
```

## Anahtar kurallar (özet — detay CONTEXT.md'de)

- Günde **5 submission**, finalde **2 seçim**. Leaderboard **%60 public / %40 private**.
- Tahminleri her zaman **`np.clip(pred, 0, 100)`**.
- OOF MSE'yi takip et; public LB'ye overfit etme.
- Kod açıklanabilir olmalı — jüri her kararı soracak.
- Ayrı süreç: **btkakademi.gov.tr başvurusu zorunlu** (kod dışı).
