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
| 2026-06-09 | Faz 7-B BERTurk feature (en iyi) | 75.65 | **86.09** | **84.10** | offset +1.99! public beklenenden çok iyi |

### ⚪ Faz 20 stack_v3: taban cephesi de DOYDU (kombo'ya katkı yok)
- stack_v3 (9-üye nested-Ridge, 5-meta) wOOF 85.44 (sv2 85.74'ten iyi) AMA kombo'ya eklenince
  tüm karışımlar 84.07-84.11 gürültü bandında (sv3<->sv2 0.997, <->f13 0.998 korele).
- ESKİ kombo (82.78) en iyi kalıyor. Submission harcanmadı. Kalan umut: (A) LLM-feature (yeni
  sinyal türü, Kaggle'da koşuyor) + (B) adversarial (düşük beklenti, test ediliyor).

### 🚀 KOMBO public=82.78 (+0.37!) — metin-meta feature mekanizması YİNE çalıştı
- 0.375*sv2 + 0.375*(featB+bert128k_meta+electra_meta) + 0.25*tabpfn -> public **82.7815**.
- Top-10 kapısı 82.63 -> ~#11, kapıya 0.15. wOOF tahmini (-0.51 -> proj 82.65) bu kez TUTTU
  (yapısal değişiklik translate ediyor; reweight oyunları etmiyor — pattern netleşti).
- Board: #1 81.33, top10 82.63. Ekran "3 DAYS TO GO" diyor (yarın gece değil — teyit et).
- KALAN KANITLI LEVER: 5-model stacker'ı (sv2) yeni metalarla yeniden kur -> kombo_v2.

### ❌ Faz 19: electra da REDUNDANT — METİN CEPHESİ KAPANDI
- electra wOOF 148.3, <->berturk 0.939, <->bert128k 0.928. Anchor'a en iyi katkı +0.03 ağırlıkla
  wOOF -0.10 (gürültü; wOOF bu boyutta 2 kez yanılttı). Submit YOK.
- ÜÇ Türkçe transformer (berturk/bert128k/electra) birbirine ~0.93-0.96 -> hepsi aynı metin sinyali.
- SONUÇ: tek işe yarayan çeşitlilik FARKLI MODALİTE idi (TabPFN). Metin doygun. Agresif keşif bitti.
- FİNAL: a=0.20 (83.185) ana aday. Public çanağı min ~a=0.25 tahmini (parabolik: ~83.15, marjinal).

### ❌ Faz 18: bert128k REDUNDANT (berturk ile 0.96 korele, blend'i bozuyor)
- bert128k OOF (Kaggle): wOOF 148.1 (berturk 146.7'den kötü, geç-yıl zayıf 2025:178/2026:181).
- KORELASYON: bert128k<->berturk 0.9626 (!), <->stacker_v2 0.79, <->tabpfn 0.68.
- berturk ZATEN stacker_v2'de -> bert128k yeni sinyal DEĞİL, aynı metnin zayıf kopyası.
- anchor(83.185)'a +0.05/+0.10 ekle -> wOOF 84.72/85.08 (YÜKSELIYOR=kötü). Submit edilmeyecek.
- TabPFN'den fark: TabPFN farklı MODALITE (tabular FM, ortogonal) -> yardım etti; bert128k aynı aile.
- DERS: metin transformerları doygun (hepsi berturk'e ~0.96). electra (farklı ön-eğitim) son umut.

### ⛔ Faz 17 mega-stacker: wOOF-OVERFIT tuzağı (submit edilmedi)
- Anchor (0.8 stacker_v2+0.2 tabpfn)=public 83.185, wOOF 84.70 -> offset 1.51.
- Ridge stacker wOOF 83.88 (+0.82) AMA tabpfn ağırlığını 0.49'a çıkarıyor. DOĞRUDAN public kanıtı
  aleyhte: a=0.20(83.185)<a=0.35(83.248) -> tabpfn artırmak public'i KÖTÜLEŞTİRDİ. Ridge 0.49 -> kötü.
- Yıl-farkında LGBM-stacker da overfit (wOOF 85.18, kötü). Faz9 tuning tuzağı tekrar.
- KARAR: lokal stacker/re-weight cephesi KAPALI (wOOF seçemiyor). Tek gerçek lever: YENİ sinyal
  (transformerlar). Onları KONSERVATİF (sabit küçük ağırlık) blend'le, wOOF-argmin DEĞİL.

### 🚀🎯 BREAKTHROUGH: TabPFN blend public=83.19 (TRANSLATE ETTİ! +0.66)
- a=0.20 (stacker_v2 83.85 + %20 TabPFN) -> public **83.185** (-0.66!). Korelasyon 0.96 olmasına
  RAĞMEN hata-çeşitliliği gerçek + public'e yansıdı. Eski lider 83.70'ti -> muhtemelen #1 bölgesi.
- Tuning-overfit'inin AKSİNE: TabPFN translate etti (yapısal bağımsız model, wOOF'a optimize değil).
- Offset(a=0.20)=84.61-83.19=1.42. wOOF optimum a=0.35 (84.23, plato 0.30-0.40) -> proj public ~82.8.
- a=0.35 PUBLIC=83.248 (a=0.20'den KÖTÜ!). wOOF optimum(0.35) != public optimum(~0.20). Public çanağı:
  a0->83.85, a0.20->83.19, a0.35->83.25. EN İYİ a=0.20 (83.185). wOOF ince ayarda yine yanıltı.
- KARAR: a=0.20 (83.185) = mevcut en iyi, final aday. a-tuning doygun (kazanç <0.05). TabPFN = büyük kaldıraç oldu.

### 🔬 Faz 16: TabPFN v2 (ham tabular) — korelasyon 0.96 (eşiği geçti) AMA blend wOOF +1.8
- TabPFN v2 (Kaggle GPU, sadece 39 sayısal+5 kategorik, text YOK): düz OOF 78.81 (Faz1 81.19'dan iyi!),
  ağırlıklı 89.98 (featB 86.09'dan zayıf — text yok). Yıl: 2025/26'da belirgin daha kötü (112.7/109.9).
- KORELASYON (OOF): TabPFN<->featB 0.9601, <->tuned GBM ~0.957, <->berturk 0.6922.
  -> %80 "bağımsızlık" eşiğini GEÇTİ (saf kurala göre eklenmez; ikisi de aynı tabular feature'da).
- AMA BLEND: (1-a)featB + a*TabPFN sweep -> a=0.30'da wOOF 84.29 (+1.80), HER YIL uniform iyileşme
  (2025:+1.30, 2026:+2.54). Sebep: pred-corr 0.96 ama HATA yapısı farklı (transformer vs GBM) -> varyans düşer.
- Tuning-overfit'inden FARKLI: TabPFN bağımsız eğitildi, wOOF'a optimize edilmedi (yalnız 'a' wOOF'tan).
  Yapısal daha güvenilir AMA yine de PUBLIC ŞART (wOOF kesin değil).
- AKSIYON: en iyi base (stacker_v2 public 83.85) + TabPFN blend, 3 aday: a15/a20/a30. a=0.20 ile public test;
  iyiyse a=0.30'a çık, kötüyse TabPFN translate etmiyor (tuning gibi) -> bırak.
- stacker_v2 (public 83.85) artık MEVCUT EN İYİ (featB 84.10'u geçti); stackerv2<->TabPFN test corr 0.9703.

### 🧭 STRATEJİK DURUM (2026-06-10, tuning dersi sonrası)
- **En iyi public = featB 84.10 (#3).** Plato ~84.1. Liderler 83.70/83.93 (fark 0.40).
- **PUBLIC kanıtı:** public'i hareket ettiren TEK şey yapısal metin sinyali (berturk 86.32->84.10, +2.22).
  Tek minimal bert-base bile +2.22 verdi -> metin sinyalinde DEV headroom. GBM/tabular DOYGUN.
- **wOOF ARTIK İNCE AYRIMDA GÜVENİLMEZ** (tuning -0.60 wOOF ama public kötü). Büyük yapısal eklemeler
  hâlâ wOOF'ta görünür; ince GBM kararları PUBLIC ile verilmeli. wOOF'a argmin-optimize ETME (overfit).
- **PLAN (azaltma için, öncelik):**
  1. TRANSFORMER ÇEŞİTLİLİĞİ (electra/xlmr/bert128k) -> featB'ye FEATURE olarak (berturk gibi).
     Kaggle'da koşuyor. Gelince: `python3 src/faz13_integrate_transformers.py` (auto-detect) -> submit -> public.
  2. Entegrasyon KONSERVATİF: default GBM paramları + SABİT blend [0.7/0.2/0.1] (faz13 doğrulandı,
     transformersız featB'yi birebir reproduce ediyor). wOOF yalnız referans.
  3. GBM churn YOK (tuning/bagging/stacker hepsi nötr-veya-zararlı; aynı wOOF-overfit ailesi).
- **Submit etme:** tuned(84.19)/bagged/stacker -> wOOF-overfit, ~84.1-84.2, slot israfı. pseudo(84.30)
  -> farklı mekanizma, düşük beklenti; isteğe bağlı 1 hakla test edilebilir (public karar verir).
- **faz13** = transformer entegrasyon pipeline'ı HAZIR + doğrulanmış (en önemli artefakt).

### ⚠️🔑 Faz 9 tuned PUBLIC=84.19 — wOOF KAZANCI TRANSLATE ETMEDİ (metrik overfit!)
- tuned wOOF 85.49 (-0.60) AMA public **84.19** vs featB 84.10 -> **0.09 KÖTÜ**. Offset +1.99'dan +1.30'a kaydı.
- TEŞHİS: 150 trial'ı test-yıl-AĞIRLIKLI OOF'a karşı optimize ettik. Bu ağırlıklandırma train'in AZ olan
  geç-yıl satırlarını yukarı çekiyor; ona sıkı optimize etmek o spesifik train satırlarına overfit etti,
  gerçek test geç-yıl dağılımına DEĞİL. wOOF düştü (reweighted train'i daha iyi fitledik) ama public düşmedi.
- KÖPRÜ DERSİ: wOOF büyük değişikliklerde (text/berturk) tuttu AMA ince hiperparametre ayrımında GÜVENİLMEZ.
  Bundan sonra ince kararları wOOF ile DEĞİL public ile ver. Default-yakını paramlar (featB) daha iyi genelliyor.
- SONUÇ: en iyi public hâlâ **featB 84.10**. tuned/bagged/stacker hepsi ~84.1-84.2 (wOOF-overfit aynı aile).
  PLATO ~84.1. Kırmak için YAPISAL yeni sinyal gerek -> transformer çeşitliliği (tek gerçek umut).
- pseudo (wOOF 84.30) artık DAHA şüpheli (en çok inflasyon); farklı mekanizma ama beklenti düşük.

### 🎯 Faz 9: GBM Optuna TUNING — BÜYÜK KAZANÇ (wOOF 86.09 -> 85.49, +0.60)
- Savaş planı #1. 3 GBM ayrı tune (Optuna TPE, test-yıl-ağırlıklı OOF hedefi, aynı fold'lar, fold-pruning).
- Tuned tekil: cat 86.07 (default 86.89, +0.82!), lgb 86.71, xgb 86.38 (default xgb berbattı).
- Blend cat0.50/lgb0.25/xgb0.25 -> wOOF **85.4920** (Faz7-B 86.09'dan **+0.60**). En büyük tek kazanç.
- Köprü tutarsa (offset +1.99) proj public ~83.5 -> #1 bölgesi (lider 83.70). GBM köprüsü sağlam,
  bu kazanç GERÇEK (tuning honest). Tuned OOF/test -> oof/test_tuned_*.npy (bagging/stacker kullanır).
- Not: arama depth<=8 + iter 1500/2500 (hız); final retrain 3000/5000. best_params faz9_tuning_log.json.
- Submission hazır: sub_2026-06-10_tuned_blend_woof85.49.csv (YARIN submit adayı #1).

### 🤔 Faz 10-11: bagging (nötr) + pseudo (şüpheli +1.28)
- Faz 10 SEED BAGGING (#2): blend 85.58 vs tuned 85.49 (-0.086, NÖTR). Bireysel iyileşti ama
  blend çeşitliliği azaldı. Değer: varyans düşük = private robustluk (yedek aday).
- Faz 11 PSEUDO (#4): wOOF **84.30** (-1.28!) AMA ⚠️ OOF-ŞİŞMESİ kuvvetle muhtemel. 3000 test
  satırı (en düşük 3-GBM std, ~%60'ı 2024-2026) kendi tahminiyle etiketlenip train'e katıldı.
  Ağırlıklı metrik geç-yılı vurguluyor -> model kendi emin geç-yıl tahminini pekiştirip OOF'u
  yapay düşürüyor olabilir. GERÇEKLİĞİ PUBLIC'te doğrulanmalı; OOF'a TEK BAŞINA güvenME.
  Submit adayı (riskli yüksek-getiri).

### ❌ Faz 8: explicit sentiment feature REDDEDİLDİ (wOOF -0.31)
- Formül-avının tek bulgusu: net=pos-neg kelime farkı hedefle 0.383 korele (lokalde doğrulandı 0.383).
- featB'ye (winner) net+pos+neg eklendi, TEK değişken: wOOF 86.09 -> **86.40 (-0.31, KÖTÜ)**.
- CatBoost importance düşük (net 0.27, pos/neg ~0.1). Sebep: text_meta(TF-IDF)+emb_meta+berturk_meta
  bu yönü ZATEN içeriyor -> explicit count redundant + gürültü. Faz 6 gibi reddedildi.
- engineer() default sentiment=False (winner yolu değişmedi). Ders: TEXT TAM DOYGUN, ham korelasyon
  != ek katkı. Tek umut çoklu-model çeşitliliği (faz multimodel).

### 🚀🤔 Faz 7-B public=84.10 — BEKLENENDEN ÇOK İYİ (offset anomalisi)
- wOOF 86.09 -> public **84.10**. Offset +1.99 (önceki: F1 +1.16, F2.1 +1.17, F2.2 +0.43).
- Public, wOOF tahmininden ~2 puan İYİ (e5-şişme'nin TERSİ). Önceki LB #1 84.96'yı geçtik -> muhtemelen #1.
- Public sıçraması (86.32->84.10 = +2.22) wOOF iyileşmesinden (0.66) ~3x büyük -> köprü bu model
  sınıfı için kararsız. İki olasılık: (a) FE+ensemble+berturk gerçekten public'e iyi genelliyor,
  (b) public %60 alt-kümesi lehimize. PRIVATE (%40) final'i belirler -> robustluk şart.
- AKSIYON: Faz 7-B = 1 final aday. 2. aday için robust/farklı bir varyant seç (public-şans hedge'i).
  Köprü artık bu sınıfta güvenilmez -> kalan kararlarda dikkatli, public ile teyit.

### 🤖 BERTurk SONUCU (2026-06-09)
- Fine-tuned BERTurk standalone OOF: düz 130.11 / ağırlıklı 146.67 -> donuk e5'i (157/180) ~34 puan geçti.
  Fold seçimleri 0-3 ep1, fold4 ep0 (hiç ep2 -> MAX=3 doğru).
- ENTEGRASYON: (A) sabit-ağırlık blend BAŞARISIZ — her α>0 wOOF'u kötüleştirdi (BERTurk-only çok zayıf,
  decorrelation yetersiz). (B) FEATURE olarak +0.18 wOOF (86.27->86.09), çoğu 2026(+1.03).
- DÜRÜST: BERTurk standalone muhteşem ama ensemble'a MARJİNAL (+0.18) — text_meta+emb_meta text sinyalini
  zaten yakalamıştı. DS residual üst sınırı +0.5-1 idi, +0.18 onun da altında -> text sinyali DOYGUN.
- +0.18 OOF e5-gibi ŞİŞEBİLİR -> public şart. En iyi model = Faz 7-B (FE+3GBM+text+emb+berturk, wOOF 86.09).

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
