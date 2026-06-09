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
