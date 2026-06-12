# Datathon 2026 — Tam Çözüm Özeti (Handoff)

**Hedef:** Her öğrenci için `career_success_score` (0–100 sürekli) tahmini.
**Metrik:** Ağırlıklı MSE (düşük = iyi). Validasyon: yıl-yeniden-ağırlıklı OOF ("wOOF").
**En iyi public:** **82.65** (lider 80.7). Kaggle'a yüklenen son dosya: `sub_2026-06-12_kombo_llmpred_B_woof83.90.csv`.

---

## 1. Veri & Validasyon Çerçevesi
- `train.csv` (10.000 satır, id STU_000001–010000) + `test_x.csv` (10.000, id STU_010001–020000). Örtüşme yok, ardışık bloklar.
- **44 sayısal** + birkaç **kategorik** + 1 uzun **metin** (`mentor_feedback_text`) + `graduation_year`.
- **Dağılım kayması:** train/test yıl dağılımı farklı → her satıra **yıl-yeniden-ağırlık** `w = p_test(yıl)/p_train(yıl)` uygulanıyor. Tüm OOF metrikleri bu ağırlıkla (`wOOF`).
- **5-fold** sabit fold'lar (tüm modeller AYNI fold'ları kullanır → leakage-free meta üretimi).
- **Public köprü kalibrasyonu** (3 nokta, stabil ≈ −1.25 sapma, lehimize):
  | kombo wOOF | public |
  |---|---|
  | 84.07 | 82.78 |
  | 83.98 | 82.73 |
  | 83.90 | **82.65** |
  → wOOF düştükçe public düşüyor (sağlıklı genelleşme, overfit yok).

## 2. Feature Mühendisliği (`engineer()`, faz4)
- Ham 44 sayısal + türetilmiş oranlar/etkileşimler (eng_cols).
- Kategorikler: CatBoost'ta native, LGBM/XGB'de category dtype, lineer modelde target-encoding.

## 3. Taban Modeller (meta katmanı)
Hepsi 5-fold OOF üretir (leakage-free):
1. **featB GBM üçlüsü:** CatBoost + LightGBM + XGBoost (sade paramlar, seed=42). Sabit blend (0.7/0.2/0.1).
2. **text_meta** — char+word TF-IDF üzerinde fold-içi meta-tahmin (faz2).
3. **emb_meta** — multilingual-e5 embedding (768-d) üzerinde meta (faz2.2).
4. **berturk_meta** — BERTurk fine-tune OOF (Kaggle GPU). corr_y ≈ 0.70. **En güçlü tekil metin sinyali (+2.22 wOOF).**
5. **bert128k_meta + electra_meta** — ek Türkçe transformer fine-tune OOF'ları (kombo +0.37).
6. **lineer Ridge** — num + 3 meta üzerinde (stacker üyesi).
7. **TabPFN** — tabular foundation model OOF (Kaggle). Ortogonal modalite, blend +0.66.

## 4. Üst Katman: iki güçlü taban + stacker
- **stacker_v2 (sv2)** — faz16: 7 üyeli nested-ridge stacker (cat/lgb/xgb/lin/text_meta/emb_meta/berturk_meta). Nested CV ile alpha seçimi (overfit-korumalı).
- **faz13_blend (f13)** — featB 3-GBM sabit blend, 5 meta ile (text/emb/berturk/bert128k/electra).
- **KOMBO (final mimari):**
  ```
  kombo = 0.375 · sv2 + 0.375 · f13 + 0.25 · tabpfn   (sabit ağırlık, anti-overfit)
  ```
  Eski kombo (LLM'siz) = public **82.78**.

## 5. RADİKAL kaldıraç — LLM (asıl ilerleme buradan geldi)
Qwen2.5-7B-Instruct (Kaggle GPU T4×2, zero-shot, hedef KULLANILMADI):

**(A) LLM feature extraction** — her metinden ayrık semantik feature:
- ton(1-10), gelisim(1-10), somut_basari(0/1) + 28 beceri bayrağı.
- `ton` corr_y = **0.49**. Ama 31 feature topluca **overfit** etti.
- **Çözüm:** sadece 3 güçlü sürekli sinyal (`ton, gelisim, somut_basari`) → base +0.21 wOOF.
- Kombonun **HER İKİ** GBM üyesine (sv2+f13) eklendi → kombo 84.07 → **83.98** → public **82.73**.

**(B) LLM direct prediction** — Qwen holistik olarak 0–100 skor tahmin etti:
- corr_y = **0.53** (grainy, 13 ayrık değer). GBM meta-feature olarak eklendi (ağaçlar kalibre etti).
- 4. LLM feature (`llm_pred`) olarak sv2+f13 bazlarına → kombo 83.98 → **83.90** → public **82.65**. ⭐
- (Stacker'a ayrı üye eklemek ekstra katkı yapmadı — sade feature en iyisi.)

**Mekanizma:** "yeni sinyali GBM feature'ı yap" kaldıracı public'e tutarlı çevrildi (berturk +2.22, bert128k/electra +0.37, LLM-feat +0.05, LLM-pred +0.08).

## 6. Son Gün — Yapı/Leak Avı (5 hipotez, HEPSİ ELENDİ)
| Keşif | Sonuç |
|---|---|
| Residual decision-tree | R²=0.009; CV-correction wOOF'u kötüleştirdi → **koşullu yapı yok** |
| id / satır-sırası leak | corr(y,id)=−0.009, blok deseni yok → **leak yok** |
| Band-sınıflandırma (metin→hedef bandı) | acc %35 (baseline %24); blend her ağırlıkta zarar → **metin yapısal kodlama yok** |
| Metin-kNN (e5) + near-duplicate | blend zarar; max-cosine >0.99 oranı **%0** → **şablon-eşleme leak yok** |
| Formül/quantization leak | lineer R² düşük, residual gürültü → **deterministik formül yok** |

**Teşhis:** Gizli yapı/sızıntı yok. Lider 80.7 ya public'e overfit ya da aynı sinyalleri marjinal daha iyi sıkıyor (RMSE farkı sadece ~0.1).

## 7. FINAL TABLO
| Aday | Public | wOOF | Risk | Rol |
|---|---|---|---|---|
| **kombo_llmpred_B** ⭐ | **82.65** | 83.90 | düşük | ANA — yüklenen son CSV |
| kombo_llm2x | 82.73 | 83.98 | düşük | B ile 0.999 korele (sigorta değeri yok) |
| **kombo_newbase** ⭐ | 82.78 | 84.07 | çok düşük | **LLM'siz** çeşitlilik sigortası |
| stackerv2+tabpfn a25 | 83.15 | — | düşük | eski nesil |

## 8. FINAL 2 SEÇİM (kilitlenecek)
1. **`kombo_llmpred_B` (82.65)** — ana silah.
2. **`kombo_newbase` (82.78)** — LLM'siz sigorta (private'ta LLM feature'ları ters köşe yaparsa).
> llm2x DEĞİL, çünkü B ile 0.999 korele → gerçek çeşitlilik yedeği LLM'siz model.

## 9. Sıradaki opsiyonel (düşük EV)
- Seed-bag (kombo'yu 5 seed ortala): beklenen +0.01-0.02, ~30 dk. Marjinal.
- Transductive target-encoding: ≤0.03. Marjinal.

---
**Özet:** featB-GBM + 5 metin-meta + TabPFN → nested-stacker kombo'su, üzerine 2 LLM kaldıracı (feature extraction + direct prediction) ile 82.78 → **82.65**. Yapı avı tüm leak hipotezlerini eledi; pozisyon top-10 eşiğinde (82.63), köprü 3 noktada stabil lehimize.
