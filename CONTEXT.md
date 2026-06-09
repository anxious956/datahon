# DATATHON 2026 — Çalışma Bağlamı (CONTEXT)

> Bu dosya, yarışmaya dair tüm bilgi birikimini içerir. Claude Code: işe başlamadan
> önce bunu baştan sona oku. Amaç, bağlamı tekrar anlatmaya gerek kalmadan doğrudan
> pipeline geliştirmeye geçebilmek.

---

## 1. Problem Özeti

- **Yarışma:** Datathon 2026 — BTK Akademi (Google + Girişimcilik Vakfı iş birliği), Kaggle üzerinde *private* community prediction competition.
- **Görev:** Öğrencilerin **`career_success_score`** değerini tahmin et. 0–100 aralığında **sürekli** bir hedef değişken.
- **Problem türü:** **Regresyon** (sınıflandırma DEĞİL).
- **Metrik:** **MSE (Mean Squared Error)** — düşük skor daha iyi. Kare alındığı için uç (çok yüksek/çok düşük skorlu) öğrencilerdeki hatalar toplam kaybı domine eder; orada doğru tahmin orantısız önemli.

## 2. Veri Dosyaları

`/kaggle/input/<klasör-adı>/` altında (klasör adını `ls` ile doğrula):

| Dosya | Açıklama |
|---|---|
| `train.csv` | Eğitim seti. `career_success_score` (hedef) DAHİL. |
| `test_x.csv` | ⚠️ Test seti. Adı `test.csv` DEĞİL, **`test_x.csv`**. Hedef sütunu YOK. |
| `sample_submission.csv` | Örnek submission formatı (~71 B). Muhtemelen `student_id` + `career_success_score`, ama AÇILDIĞINDA DOĞRULA. |

Toplam ~11.32 MB, 3 dosya. train.csv satır sayısı henüz teyit edilmedi.

## 3. Kolonlar (Data sözlüğünden)

**ID:** `student_id`
**Hedef (yalnız train):** `career_success_score` (0–100, sürekli)

**⚠️ Metin / NLP alanı:** `mentor_feedback_text` — mentor değerlendirme metni. Öğrencinin gelişimi, teknik yaklaşımı, iletişimi ve potansiyeli hakkında serbest metin. **İşin ayrışma noktası burası** (aşağıda Strateji'ye bak). DİLİ (TR/EN) açıldığında ilk iş kontrol et — embedding modeli seçimi buna bağlı.

**Kategorik:** `department`, `university_tier`, `target_role`, `hobby`, `preferred_social_media_platform`

**Sayısal (görünüm itibariyle):**
`application_year`, `age`, `graduation_year`, `cgpa`, `english_exam_score`, `attendance_rate`,
`failed_courses_count`, `coding_score`, `problem_solving_score`, `data_structures_score`,
`sql_score`, `machine_learning_score`, `backend_score`, `frontend_score`, `cloud_score`,
`devops_score`, `project_quality_score`, `real_client_project_count`, `internship_count`,
`internship_duration_months`, `freelance_project_count`, `hackathon_count`, `hackathon_awards`,
`portfolio_score`, `github_repo_count`, `github_avg_stars`, `open_source_contribution_count`,
`linkedin_profile_score`, `cv_quality_score`, `technical_interview_score`, `hr_interview_score`,
`communication_score`, `teamwork_score`, `leadership_score`, `presentation_score`,
`certification_count`, `bootcamp_count`, `applications_sent`, `interviews_attended`

> Not: Kesin dtype'lar (int/float/object) açıldığında `train.dtypes` ile teyit edilecek. Data sözlüğü 49 alan açıklıyor; sayfadaki "95 columns" özeti 3 dosyanın toplamı olabilir — train'in gerçek kolon sayısını doğrula.

## 4. Kurallar (kritik olanlar)

- **Submission limiti: günde 5.** (Eski PDF'te "3" yazıyordu — GEÇERSİZ. Bağlayıcı olan Kaggle'ın kendi limiti: 5/gün, her gece 23:59'da sıfırlanır.)
- **Final seçimi: 2 submission.** En iyi public skor + en robust/CV-güvenli olan seçilmeli; public LB'ye overfit etme.
- **Leaderboard: %60 public / %40 private.** Final sıralama PRIVATE ile belli olur. Public'e overfit etmek tuzak.
- **Tek Kaggle hesabı.** Birden fazla hesap = diskalifiye.
- **btkakademi.gov.tr başvurusu zorunlu** (ayrı süreç, kod dışı). Yapılmazsa ilk 10'a girilse bile diskalifiye.
- **Takım adı** btkakademi başvurusundaki ile birebir aynı olmalı.
- **AI araçları (ChatGPT/Claude) serbest** — AMA jüri "bunu neden yazdın?" diye soracak, her satırı açıklayabilmek şart.
- **Harici veri:** kurallarda net yasak yok; takımlar arası Kaggle dışı transfer yasak. Kullanılırsa kaynağı belgele.
- **İlk 10 → jüriye 5+3 dk online sunum + notebook/kod paylaşımı ZORUNLU.**
- **Kaggle 1.'liği ≠ yarışma birinciliği.** Kazanan jüri sunumuyla belirlenir. Temiz, açıklanabilir notebook + güçlü NLP hikayesi rank'tan daha önemli olabilir.

## 5. Leaderboard Durumu (yarışma başında, public)

İlk 10 çok dar bantta: **86.43 – 87.78 MSE.** İlk 3 birbirinden ~0.09 fark. En iyi: **86.427** (Kerem Bişici). Code sekmesi başlangıçta boştu (paylaşılan notebook yok), Discussion boştu.

**Yorum:** Bu dar yığılma muhtemelen herkesin sayısal kolonlarla benzer bir GBM atıp aynı tabana toplandığını gösteriyor. `mentor_feedback_text` büyük olasılıkla henüz tam sömürülmedi → metinde **kullanılmamış sinyal** olma ihtimali yüksek. Gerçek taban 86'nın altında olabilir.

**Teşhis submission'ı:** İlk iş — hedefin ortalamasını sabit tahmin edip submit et. Bu, "hiçbir şey öğrenmeden" MSE'yi (≈ hedef varyansı) verir. 86 ile kıyaslayınca modellerin ne kadar sinyal çıkardığı ve kalan headroom görülür.

### 📒 Submission / deney günlüğü (gerçek public skorlar)

| Tarih | Model | Düz OOF | Ağırlıklı OOF | **Public** | Not |
|---|---|---|---|---|---|
| 2026-06-09 | Faz 0 sabit-ortalama (76.94) | (var) 230.6 | — | **274.72** | temporal kayma ortaya çıktı |
| 2026-06-09 | Faz 1 CatBoost (metinsiz, 44 feat) | 81.19 | 92.38 | **91.22** | köprü kalibre edildi |
| 2026-06-09 | Faz 2.0 +metin (word TF-IDF→Ridge) | 77.73 | 88.49 | (proj ~87.3, submit yok) | metin −3.89 ağırlıklı, çoğu 2025 |
| 2026-06-09 | Faz 2.1 +word+char(3,5)-gram | 77.24 | 87.83 | **86.66** | char +0.66; köprü 2. kez doğrulandı |
| 2026-06-09 | Faz 2.2 +e5-base emb_meta (KEEP) | 76.50 | 86.75 | **86.32** | emb OOF −1.08 AMA public yalnız −0.33! |

### ⚠️ KÖPRÜ UYARISI — embedding meta-feature OOF'u ŞİŞİYOR
- TF-IDF/GBM köprüsü 2 kez ±0.04 birebir tuttu (offset ~1.17). Ama embedding ekleyince
  offset **1.17 → 0.43**'e düştü: ağırlıklı OOF 86.75 dedi, public **86.32** geldi.
- Embedding'in OOF kazancı (−1.08) gerçeğe ancak **~⅓** yansıdı (gerçek public kazancı −0.33).
- **Sebep:** 768-boyut yoğun embedding üzerine Ridge meta-feature, paylaşılan-fold stacking'de
  OOF'u overfit ediyor (seyrek TF-IDF bunu yapmıyor). → **embedding/meta OOF kazançlarını DİSKONTO et,
  public'e güven.** GBM model-çeşitliliği (Faz 3) kazancı daha "gerçek" beklenir.
- Embedding yine de KEEP (gerçek +0.33, en iyi public 86.32) ama beklenti düşük tutulacak.

| 2026-06-09 | Faz 3 blend text_meta (cat+lgb+xgb) | 76.89 | 87.46 | (proj ~86.29) | GBM diversity +0.37 (güvenilir köprü) |
| 2026-06-09 | Faz 3 blend +emb_meta | 76.34 | 86.62 | (gerçek ~86.2*) | *emb-şişme diskonto; submit yok |
| 2026-06-09 | Faz 4 KÜRATÖRLÜ FE + blend | 75.95 | **86.27** | (proj ~86.0) | +0.35 güvenilir; tam FE batırdı, küratör kazandı |
| 2026-06-09 | Faz 5 +lineer Ridge (4-yönlü) | 75.85 | 86.24 | (~86.0) | +0.04 marjinal; lineer geç-yılda zayıf |
| 2026-06-09 | Faz 6 sample-weight EĞİTİM | 77.47 | 87.13 | — | -0.85 NET KAYIP; reddedildi |

### ⛔ SAMPLE-WEIGHT EĞİTİM DENENDİ, REDDEDİLDİ (2026-06-09) — kritik içgörü
- GBM'leri test-yıl ağırlığıyla eğittik (geç-yıl up-weight) + early stopping ağırlıklı val'da.
- Sonuç: blend wOOF 86.27 -> 87.13 (-0.85). Yıl kırılımı: geç-yıl 2025/2026 yalnız +0.24/+0.25
  (mikroskobik) iyileşti; erken/orta yıllar -2..-3.5 BOZULDU. Ağırlıklı toplamda net kayıp.
- **İÇGÖRÜ:** Geç-yıl yüksek MSE'si model yanlış-odaklanması DEĞİL; mevcut feature'larla büyük
  ölçüde İNDİRGENEMEZ (irreducible noise). Reweighting oraya yardım etmiyor (+0.24 tavan),
  sadece öğrenilebilir erken-yılı feda ediyor.
- **STRATEJİK SONUÇ:** İlk 3'e fark (~1.3) reweighting'le KAPANMAZ; yalnız YENİ SİNYAL ile kapanır
  (geç-yılda yeni bilgi = BERTurk text). DS residual üst sınırı +0.5-1 -> ilk 3 zor ama BERTurk
  tek gerçek koz. Eval-ağırlığı (köprü) korunur; EĞİTİM-ağırlığı kullanılmaz.

### 📐 LİNEER MODEL DERSİ (2026-06-09, DS-analizi takibi)
- DS bulgusu: hedef büyük ölçüde lineer (sadece-sayısal Ridge R²≈0.573). Tam feature setiyle lineer
  düz OOF 85.26 (R²≈0.63) — GBM'e (76, R²≈0.67) yakın. Hipotez sağlamdı.
- AMA ağırlıklı (geç-yıl) OOF: lineer **98.6** vs GBM **86.6**. Lineer 2025-2026'da çok zayıf;
  örnek-ağırlığı da kurtarmadı (97.1). 4-yönlü blend kazancı sadece **+0.04** (lin ağırlık 0.05).
- İçgörü: hedef erken-yılda lineer ama MSE'nin yaşadığı geç-yıl (yüksek varyans + kayma) kısmı
  nonlineer/shift yapısı taşıyor; GBM orayı yakalıyor, lineer yakalayamıyor. Lineer = gerçek
  çeşitlilik ama zayıf üye → küçük ağırlıkla tutulur (zarsız, robustluk), plato-kıran DEĞİL.
- Plato-kıran tek koz hâlâ **BERTurk** (text); DS residual testi: metnin bağımsız sinyali az
  (+0.5-1 beklenti). Strateji: FE(+0.35)+lineer(+0.04)+BERTurk(+0.5-1) BİRİKİMLİ ~85.3 hedefi.

### 🔧 FE DERSİ (2026-06-09)
- 26-feature geniş FE wOOF'u +0.6 KÖTÜLEŞTİRDİ (86.62→87.22). Suçlular: yıl-türevleri
  (drift'i kodluyor → geç-yıl metriğinde zarar) + bölme-oranları (NaN gürültüsü).
- Küratörlü 11-feature (skill/soft/interview mean + maxed_skills + 7 NaN bayrağı) **−0.35** kazandırdı.
- Genel kural: GBM oran/etkileşimi kendi yakalıyor; sadece **global özet (agregasyon) + eksik-sinyali**
  (NaN flag) net pozitif. Yıl-türevi feature drift altında ZARARLI.

### 📌 PLATO (2026-06-09) — mevcut özelliklerle ~86.2 bandı
- LGBM(89.2)/XGB(90.0) CatBoost'tan (87.83) kötü; blend cat'e yaslanıyor, diversity kazancı zayıf.
- Ensemble tek başına yetmiyor. Gerçek ilerleme için: **(a) CatBoost hiperparametre tuning**
  (workhorse o), **(b) feature engineering** (zengin sayısal kolonlardan oran/etkileşim/agregasyon),
  **(c) geç-yıl yüksek-varyans için hedef dönüşümü** denenebilir.
- Bugün 4 submission kullanıldı (limit 5/gün), 1 hak kaldı; yarışmaya 5 gün var. Son hakkı
  marjinal ensemble'a harcamak yerine lokalde daha büyük kazanç aranmalı.

**Köprü offset SABİT (2 rejim):** Faz 1 ağırlıklı 92.38→public 91.22 (offset +1.16); Faz 2.1 ağırlıklı 87.83→public 86.66 (offset +1.17). Ağırlıklı OOF, public'i **~1.17 pesimist sabit offset**le izliyor → kalan geliştirmeler lokalde, ağırlıklı OOF düşüşü public'e ~1:1 yansır. **Faz 2.1 = güçlü final aday** (LB best 86.427'ye 0.23 fark, embedding'siz/açıklanabilir).

### 🔑 KÖPRÜ MÜHÜRLENDİ — test-yıl-ağırlıklı OOF ≈ public
Faz 1: ağırlıklı OOF **92.38** vs gerçek public **91.22** → fark **+1.16** (%1.3, **pesimist** yönde, yani güvenli).
Düz OOF (81.19) ise public'ten 10 puan iyimserdi → **düz OOF'a GÜVENME, ağırlıklıyı kullan.**
**Sonuç:** Faz 2+ lokalde, **submission harcamadan** ilerletilebilir; ağırlıklı OOF'taki iyileşme public'e
en az o kadar yansır. (İsteğe bağlı ince ayar: ağırlığa `graduation_year` da katılırsa ~1 puanlık
pesimist sapma kapanabilir; şu an gerek yok, tek-`application_year` ağırlığı güvenli yönde.)


## 6. Strateji / Pipeline Planı

**Faz 0 — Teşhis (bugün):**
- Veriyi yükle, `student_id`/hedef/metin dili/sample_submission yapısını teyit et.
- Sabit-ortalama submission at (hem katılımı kilitler hem varyans tabanını verir).

**Faz 1 — Baseline:**
- 5-fold CV. Hedefi binleyip **stratified** yaparsan CV daha stabil (sürekli hedef için binned-stratified).
- **CatBoost** veya LightGBM. CatBoost kategorikleri native yer → encoding derdi yok, ilk tercih.
- Tüm sayısal + kategorik kolonlar. Bu, ~86 tabanının nereden geldiğini doğrular.

**Faz 2 — Metin (ayrışma noktası):**
- Önce ucuz: **TF-IDF + TruncatedSVD** → GBM'e ek feature.
- Sonra: **multilingual sentence embeddings** (dil TR ise ona uygun model). Embedding'leri feature olarak ver.
- En güçlü: text-only bir **transformer'ı fine-tune** et, OOF tahminini meta-feature yap.

**Faz 3 — Ensemble:**
- GBM OOF + text-model OOF tahminlerini **OOF üzerinde ağırlık tuning'iyle** blend et. (Public LB'de değil — OOF'ta.)

**Her submission'da ZORUNLU son adım:**
- Tahminleri **`np.clip(pred, 0, 100)`** ile aralığa sıkıştır. MSE olduğu için aralık dışı tahmin kesin daha kötü. Bedava kazanç.

## 7. Çalışma Kuralları (Claude Code için)

- Her model değişikliğinde **OOF MSE'yi** logla ve `experiments/` altında tut. Public LB tek başına yanıltıcı.
- Notebook/script'leri **açıklanabilir** tut — jüri her kararı soracak. Neden-niçin yorumları ekle.
- Submission dosyalarını `submissions/` altında tarih+skor ile isimlendir (örn. `sub_2026-06-09_catboost_oof12.34.csv`).
- Random seed sabitle, sonuçlar tekrarlanabilir olsun.
- `test_x.csv` adını unutma (test.csv değil).
- Kategorik dtype'ları ve eksik değerleri açılışta raporla.

## 8. Açıldığında İLK Teyit Edilecekler — ✅ TEYİT EDİLDİ (2026-06-09)

Yerel veri zip'i (`datathon2026.zip`) üzerinde teşhis çalıştırıldı. Sonuçlar:

- [x] **Klasör adı:** Kaggle'da `/kaggle/input/*` ile otomatik bulunuyor (zip kökünde 3 dosya).
- [x] **Boyut:** `train` = **10000 × 47**, `test_x` = **10000 × 46** (hedef yok). Kolonlar:
  `student_id` (ID) + `career_success_score` (hedef) + `mentor_feedback_text` (metin)
  + **5 kategorik** + **39 sayısal**. dtype: 26 float, 14 int, 7 str.
- [x] **Metin dili:** **TÜRKÇE** (TR'ye özgü karakter oranı 0.998, TR-kelime 0.996 vs EN 0.003).
  10000/10000 benzersiz, NaN yok, boş yok, uzunluk ~143–447 char (ort. ~274).
  → Embedding için **Türkçe/multilingual** model gerekir (örn. `dbmdz/bert-base-turkish`,
  `intfloat/multilingual-e5`, ya da TR-uyumlu sentence-transformers).
- [x] **Hedef dağılımı:** mean **76.94**, std **15.19**, min **0**, max **100**,
  skew **-0.45** (sola çarpık, düşük skorlu kuyruk), kurt -0.15. Aralık tam 0–100, taşma yok.
  **var(pop) = 230.61** → sabit-ortalama submission'ın beklenen public MSE'si ≈ **230.6**.
- [x] **sample_submission:** kolonlar `student_id, career_success_score`. ⚠️ **SADECE 2 SATIR** —
  bu bir FORMAT örneği (değerleri rastgele; biri 123.94 ile 100 üstü). Gerçek submission
  `test_x`'in 10000 satırının TAMAMI için, **test_x ID'lerinden** kurulmalı. `sub.copy()` KULLANMA.
- [x] **Eksik değerler (train):** `internship_duration_months` 1657, `english_exam_score` 953,
  `github_avg_stars` 910, `open_source_contribution_count` 910, `hr_interview_score` 780,
  `linkedin_profile_score` 668, `portfolio_score` 364. (test_x'te benzer oranlar.)
  **Metin alanında NaN YOK.** → CatBoost NaN'ı native yönetir; lineer/embedding adımında impute gerekir.

**Kategorik kolonlar (kardinalite):** `department` (7), `university_tier` (4, Tier 1–4),
`target_role` (11), `hobby` (8), `preferred_social_media_platform` (6). Hepsinde NaN yok.

> ⚙️ **pandas 3 notu:** pandas 3.x metni `object` yerine `str` (StringDtype) işaretler;
> `dtype=='object'` kontrolü kategorikleri kaçırır. Teşhis kodu `is_string_dtype` ile düzeltildi.

> 📉 **Headroom gözlemi:** sabit-ortalama MSE ≈ **230.6**, public LB en iyi ≈ **86.4**.
> Yani sayısal+kategorik sinyal MSE'yi ~%63 düşürüyor; metin (`mentor_feedback_text`) henüz
> büyük olasılıkla tam sömürülmedi → asıl ayrışma orada. Faz 2'nin önceliği yüksek.

### ⚠️ Faz 0 submission sonucu — TEMPORAL KAYMA (2026-06-09, KRİTİK)

Sabit-ortalama (76.94) submission'ı atıldı → **Public MSE = 274.72.**
Beklenen 230.6 değil! Fark = **+44.1**. Bu rastgele örnekleme gürültüsü DEĞİL (~10σ).

**Teşhis: train/test ayrımı ZAMANSAL.**
- `application_year`/`graduation_year` ~0.6 std kaymış; diğer 37 sayısal ve 5 kategorik
  özellik neredeyse aynı (std_diff < 0.03). → tek kayma kaynağı **yıl**.
- **train yıllara ~uniform** (2019–2026, her biri ~%12); **test geç yıllara ağırlıklı**
  (2024+2025+2026 ≈ test'in **%62'si**, train'de ~%35).
- **Hedef yılla değişiyor:** geç kohortlar hem **daha düşük ortalama** hem **daha yüksek varyans**:
  2019–2023 → mean ~77.5, std ~13; **2025 → mean 75.5, std 18.3; 2026 → mean 74.2, std 18.0**.
- **Doğrulama:** train'in yıl-bazlı (mean,var) istatistiklerini test'in yıl oranlarıyla
  ağırlıklandırınca sabit-baseline tahmini = **266.8** (gerçek 274.7'ye çok yakın). Hipotez sağlam.
- Train **2019–2026'nın tamamını** kapsıyor → görülmemiş yıl YOK; bu *prior-probability shift*
  (yıl oranları kaymış), sert ekstrapolasyon değil. Yönetilebilir.

**Faz 1 için bağlayıcı sonuç — DÜZ RANDOM K-FOLD CV YANILTICI OLUR:**
- Random fold'lar train'in "230 dünyası"nı ölçer; gerçek test "≈275 dünyası". Yani düz OOF MSE
  public'ten **sistematik olarak iyimser** çıkar (tam da bugün gördüğümüz 230 vs 274).
- **Çözüm:** CV'yi test-temsili yap. Ana takip metriği = **test-yıl-ağırlıklı OOF MSE**
  (her OOF hatasını `test_yıl_oranı / train_yıl_oranı` ile ağırlıkla → importance weighting).
  Bu sayı public LB'yi izlemeli. Ayrıca **OOF'u yıla göre kır** ve özellikle 2025–2026'da raporla.
- Fold'ları hedef-bin + (tercihen) yıl ile stratify et. Model **seçimi** için göreli sıralama
  düz CV'de de geçerli; ama mutlak public tahmini için yıl-ağırlıklı sayıyı kullan.
- `application_year`/`graduation_year` hem güçlü feature hem kayma sürücüsü — modele dahil et,
  ama yıl-aşırı genelleme yeteneğini izle (geç yıllar daha yüksek varyans = içsel olarak daha zor;
  MSE'nin yaşadığı yer orası → metin sinyali geç kohortta orantısız değerli olabilir).

## 9. Hızlı Teşhis Kodu (ilk çalıştır)

```python
import pandas as pd, glob, os
base = glob.glob('/kaggle/input/*')[0]   # klasör adını otomatik bul
print("input klasörü:", base, os.listdir(base))

train = pd.read_csv(f'{base}/train.csv')
test  = pd.read_csv(f'{base}/test_x.csv')
sub   = pd.read_csv(f'{base}/sample_submission.csv')

print(train.shape, test.shape)
print(sub.head())                                  # submission formatı
print(train['career_success_score'].describe())    # hedef dağılımı
print(train['mentor_feedback_text'].head(3).tolist())  # metin dili?
print(train.isna().sum()[lambda s: s > 0])         # eksik değerler
print(train.dtypes.value_counts())                 # dtype dağılımı
```
