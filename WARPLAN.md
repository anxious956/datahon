# SAVAŞ PLANI — MSE'yi indir (lider 82.00, biz 84.10)

> Tek hedef: public MSE'yi mümkün olduğunca düşür. Sıralama mantığı: **önce güvenli+bedava,
> sonra riskli.** Her adım birikimli; her birikimli adımın katkısı AYRI raporlanır ki yarın
> 5 submission hakkını hangi varyantlara harcayacağımıza veriyle karar verelim.

## Mevcut durum (giriş noktası)
- **En iyi public = 84.10** (Faz 7-B: FE + 3 GBM + text_meta(charword) + emb_meta + berturk_meta).
- Karşılık gelen **ağırlıklı OOF (wOOF) = 86.09.** Köprü (test-yıl-ağırlıklı OOF ≈ public)
  bu model sınıfında +1.99 offset verdi (önceki rejimlerde +1.16/+1.17/+0.43). → Köprü artık
  bu sınıfta KARARSIZ; lokal kazançları public ile doğrulamak ŞART.
- GBM'ler şimdiye kadar **default-yakını** paramlarla koştu (cat: depth6/lr0.03/l2=3,
  lgb: leaves31/lr0.03/l2=3, xgb: depth6/lr0.03/l2=3). **Hiç sistematik tune edilmedi** → en bariz açık.

## Değişmez kurallar (köprü bütünlüğü)
- **Aynı fold'lar:** `make_folds` (yıl×hedef-desil stratify, SEED=42). Asla değiştirme.
- **Hedef metrik:** test-yıl-ağırlıklı OOF MSE (`metrics()` → weighted). Düz OOF'a GÜVENME.
- **Her tahmin `np.clip(0,100)`.**
- **Meta-feature'lar fold-içi OOF** üretilir (leakage-free): text_meta, emb_meta, berturk_meta.
  Tuning sırasında BİR KEZ hesaplanıp tüm trial'larda yeniden kullanılır (GBM paramından bağımsız).
- Her adım `experiments/` altına log + OOF `.npy` yazar; yıl kırılımı (özellikle 2025-2026) raporlanır.

---

## ADIM 1 — GBM HİPERPARAMETRE TUNING  `src/faz9_tune.py`  [GÜVENLİ+BEDAVA, ÖNCE BU]
**Neden:** En bariz açık. Klasik +0.2–0.5 kaynağı, köprü GBM tarafında sağlam.
- Optuna (TPE) ile **her GBM ayrı** tune edilir. Objektif = winner feature setinde 5-fold
  **wOOF MSE** (aynı fold'lar). ~60 trial/model (ayarlanabilir).
- Feature seti = Faz 7 winner: `engineer()` FE sayısal + kategorik + text_meta + emb_meta + berturk_meta.
  (Tuned paramların final ensemble'a birebir transfer olması için.)
- Arama uzayı: **depth/num_leaves, learning_rate, l2/reg_lambda, reg_alpha, subsample,
  colsample, min_child, bagging.** `iterations/n_estimators` TUNE EDİLMEZ — yüksek tavan +
  early-stopping fold başına en iyi iterasyonu zaten bulur (daha robust).
- Çıktı: `faz9_tuned_params.json`, tuned per-model OOF/test, tuned blend, submission, log.
- **Beklenti:** wOOF 86.09 → ~85.6–85.9 (+0.2–0.5). Public ile doğrulanacak.

## ADIM 2 — SEED BAGGING  (faz9 içine entegre)  [BEDAVA, GÜVENLİ]
**Neden:** Varyans düşürür, bedava +0.1–0.3.
- Tuned her model **N seed** (varsayılan [42,1337,2024,7,99]) ile koşulur; OOF ve test
  tahminleri seed-ortalanır. Fold'lar sabit kalır (yalnız model random_seed değişir).
- Tuning biter bitmez otomatik eklenir → faz9 çıktısı zaten "tuned+bagged".
- **Beklenti:** ek +0.1–0.3 wOOF.

## ADIM 3 — ÇOKLU-MODEL TRANSFORMERS  `notebooks/multimodel_transformers_kaggle.ipynb`  [HAZIR ✅]
**Durum:** XLM-R LR=1e-5 override + fold-başı tek-sefer tensör slice optimizasyonu **commit edildi**
(`e7c112a`). Kullanıcı Kaggle'da koşturacak → paralel ilerler. Çıktı OOF'lar geldiğinde
Adım 5 (stacker) girdilerine eklenir.

## ADIM 4 — PSEUDO-LABELING  `src/faz11_pseudo.py`  [RİSKLİ, geç-yılı hedefler]
**Neden:** 2025-2026 test örnekleri train'de az; en kararlı tahminleri etiketleyip katmak
geç-yıl sinyalini güçlendirebilir.
- **Seçim:** 3 GBM'in test tahmin std'si DÜŞÜK (en kararlı) örnekler; özellikle 2025-2026.
- **LEAKAGE-GÜVENLİ TASARIM (kritik):** Pseudo-label'lar fold-DIŞI üretilir. Fold k için
  pseudo-test etiketi = SADECE fold-k-train'de eğitilmiş modelin test tahmini (faz9'un fold-başı
  test tahminleri). Fold k'nın modeli (fold-k-train + güvenli-pseudo-test) ile eğitilip fold-k-val
  tahmin edilir → val sızıntısı YOK, OOF dürüst kalır.
- **DİKKAT:** Pseudo-labeling OOF'u yapay iyileştirebilir (kendi tahminini doğrulama). Kazanç
  MUTLAKA public'te doğrulanır; OOF'a tek başına GÜVENME.
- **Beklenti:** belirsiz; geç-yılda +, genelde gürültü riski. Public hakemi.

## ADIM 5 — ÖĞRENİLMİŞ STACKER  `src/faz12_stacker.py`  [EN RİSKLİ, son]
**Neden:** Basit blend yerine tüm OOF'lar üstünde öğrenilmiş birleştirici.
- Girdi: tüm OOF'lar (cat/lgb/xgb tuned + lin + text + emb + berturk + yeni transformerlar).
- Model: Ridge (küçük alpha) veya küçük-LGBM. **Nested CV** ile fit (stacker overfit'ini önle):
  stacker katsayıları yalnız fold-train OOF'larında öğrenilir.
- **OOF-şişme riski yüksek** (embedding dersi: emb wOOF -1.08 dedi, public yalnız -0.33).
  Kazancı DİSKONTO et, public doğrulaması şart.
- **Beklenti:** blend'e marjinal; public'te basit blend'i geçerse tut.

---

## Birikimli raporlama (yarın submit seçimi için)
| Varyant | İçerik | wOOF | proj public | Not |
|---|---|---|---|---|
| V0 winner | Faz 7-B (mevcut) | 86.09 | 84.10 (gerçek) | referans/robust yedek |
| V1 tuned | Adım 1 | — | — | tuning kazancı |
| V2 tuned+bagged | Adım 1+2 | — | — | **muhtemel ana aday** |
| V3 +transformers | +Adım 3 | — | — | Kaggle OOF gelince |
| V4 +pseudo | +Adım 4 | — | — | geç-yıl hedef |
| V5 +stacker | +Adım 5 | — | — | en riskli |

**5 submission hak dağılımı (öneri):** V2 (tuned+bagged), V3 (+transformers), V4 (+pseudo),
ve V0 (robust yedek/hedge). 5.'yi public sonuçlarına göre seç. Public %60 / private %40 —
private'ı belirleyen robustluk; tek bir public sıçramasına overfit etme.

> Köprü bu model sınıfında kararsız (offset +1.99) → her birikimli adımın wOOF kazancı
> public'e 1:1 yansımayabilir. Karar: lokalde tüm varyantları üret, public'te az sayıda
> kritik submit ile doğrula, private için en robust 2'yi finale seç.
