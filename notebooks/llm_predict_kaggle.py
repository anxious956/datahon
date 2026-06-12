# ===================== LLM DIRECT PREDICTOR — TEK HÜCRE =====================
import os, glob, json, re, time
import numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "GPU YOK! Settings->Accelerator->GPU T4 x2 -> Restart"
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())

cands = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
         if any(k in p.lower() for k in ('qwen','gemma'))]
assert cands, "Model yok! Add Input -> Models -> 'Qwen2.5 7B Instruct' ekle"
MODEL_PATH = sorted(cands, key=len)[0]
print("MODEL:", MODEL_PATH)

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map='auto')
model.eval()

base = [p for p in glob.glob('/kaggle/input/*') if os.path.exists(f'{p}/train.csv')]
if not base:
    hits = glob.glob('/kaggle/input/**/train.csv', recursive=True); base=[os.path.dirname(hits[0])]
base = base[0]
train = pd.read_csv(f'{base}/train.csv'); test = pd.read_csv(f'{base}/test_x.csv')
TEXT='mentor_feedback_text'; ID='student_id'

# Kalibrasyon: train target agregatı (mean~77, p10~56, p90~98). Per-row label DEĞİL -> sızıntısız.
SYS = ("Sen bir teknik işe alım / kariyer değerlendirme uzmanısın. Sana bir öğrenci hakkında "
       "mentor geri bildirimi verilecek. Bu öğrencinin KARİYER BAŞARI SKORUNU 0-100 arası tahmin et.\n"
       "Ölçek: 100=olağanüstü/sekt-üstü, 90=çok güçlü, 77=tipik/ortalama öğrenci, 60=zayıf yönleri belirgin, "
       "40=ciddi eksikler, 0-20=çok yetersiz. Çoğu öğrenci 55-98 bandındadır ama hak ediyorsa uç ver.\n"
       "Metindeki övgü/eleştiri dengesine, somut başarılara, teknik derinliğe ve mentorun güven tonuna bak. "
       "Abartılı övgü->yüksek, çekinceli/eleştirel ton->düşük.\n"
       'SADECE şu JSON’u döndür, başka hiçbir şey yazma: {"skor": N}  (N = 0-100 tam sayı)')

def make_prompts(texts):
    msgs = [[{"role":"system","content":SYS},{"role":"user","content":t[:1400]}] for t in texts]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

def parse_one(s):
    m = re.search(r'"skor"\s*:\s*(-?\d+(?:\.\d+)?)', s)
    if not m:
        m = re.search(r'(\d{1,3}(?:\.\d+)?)', s)   # fallback: ilk sayı
    if not m: return {'pred': 77.0, 'parse_ok': 0}
    v = float(m.group(1))
    return {'pred': float(np.clip(v, 0, 100)), 'parse_ok': 1}

@torch.no_grad()
def predict(texts, tag, bs=24):
    part = f'/kaggle/working/llm_pred_partial_{tag}.csv'
    done = 0; rows=[]
    if os.path.exists(part):
        prev = pd.read_csv(part); rows = prev.to_dict('records'); done = len(rows)
        print(f"[{tag}] devam: {done} hazır")
    t0=time.time()
    for s in range(done, len(texts), bs):
        batch = texts[s:s+bs]
        enc = tok(make_prompts(batch), return_tensors='pt', padding=True, truncation=True, max_length=560).to(model.device)
        out = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        rows += [parse_one(d) for d in dec]
        if (s//bs) % 8 == 0:
            el=time.time()-t0; done_n=len(rows)-done
            eta=(len(texts)-len(rows))/max(done_n/el,1e-9)/60 if done_n else -1
            print(f"[{tag}] {len(rows)}/{len(texts)}  ({el/60:.1f}dk, ETA {eta:.0f}dk)", flush=True)
        if (s//bs) % 16 == 0 and len(rows)>done:
            pd.DataFrame(rows).to_csv(part, index=False)
    df = pd.DataFrame(rows); df.to_csv(part, index=False)
    return df

for tag, frame in [('train', train), ('test', test)]:
    pr = predict(frame[TEXT].fillna('').tolist(), tag)
    pr.insert(0, ID, frame[ID].values)
    pr.to_csv(f'/kaggle/working/llm_pred_{tag}.csv', index=False)
    print(f"[{tag}] BİTTİ -> llm_pred_{tag}.csv | parse_ok={pr['parse_ok'].mean():.3f} "
          f"pred: mean={pr['pred'].mean():.1f} std={pr['pred'].std():.1f} "
          f"min={pr['pred'].min():.0f} max={pr['pred'].max():.0f}")

# hızlı sağlama: train tahmini ile hedef korelasyonu (varsa) -> sinyal gücü
tr_pred = pd.read_csv('/kaggle/working/llm_pred_train.csv')
if 'career_success_score' in train.columns:
    c = np.corrcoef(tr_pred['pred'].values, train['career_success_score'].values)[0,1]
    print(f"\n>>> corr(llm_pred, target) = {c:.4f}   (berturk ~0.7+; >0.4 ise güçlü blend üyesi)")
print("\nHEPSİ BİTTİ — llm_pred_train.csv + llm_pred_test.csv indir, bana yükle.")
