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

## 8. Açıldığında İLK Teyit Edilecekler (henüz bilinmiyor)

- [ ] `/kaggle/input/` altındaki klasör adı
- [ ] train.csv satır sayısı + gerçek kolon sayısı + dtype'lar
- [ ] `mentor_feedback_text` dili (TR / EN / karışık)
- [ ] hedefin dağılımı (describe: mean, std, min, max, çarpıklık)
- [ ] `sample_submission.csv`'nin birebir kolon yapısı
- [ ] eksik değer (NaN) durumu, özellikle metin alanında

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
