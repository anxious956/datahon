# ============= LLM 14B + FEW-SHOT DIRECT PREDICTOR — TEK HÜCRE (max sinyal) =============
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','-U','bitsandbytes>=0.46.1'], check=False)
import gc, torch
gc.collect(); torch.cuda.empty_cache()

# Qwen2.5-14B-Instruct + few-shot. 7B zero-shot 0.53 verdi; daha BÜYÜK model + few-shot ->
# berturk(0.70)'in kaçırdığı yeni sinyal şansı en yüksek. Çıktı: llm_pred2_{train,test}.csv
# KURULUM: Add Input -> Models -> "Qwen2.5 14B Instruct" (transformers, 14b-instruct) + Datathon 2026
#          GPU T4 x2, Internet on. Süre ~3-3.5 saat (14B yavaş). Ara kayıt var, kesilirse devam eder.
import os, glob, json, re, time
import numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "GPU YOK! Settings->Accelerator->GPU T4 x2 -> Restart"
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())

# en BÜYÜK qwen modelini seç (14b > 7b); boyut anahtar kelimesine göre öncele
cands = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
         if any(k in p.lower() for k in ('qwen','gemma'))]
assert cands, "Model yok! Add Input -> Models -> 'Qwen2.5 14B Instruct'"
def rank(p):
    pl=p.lower()
    for sz,r in [('32b',0),('14b',1),('9b',2),('7b',3)]:
        if sz in pl: return r
    return 9
MODEL_PATH = sorted(cands, key=lambda p:(rank(p), len(p)))[0]
print("MODEL:", MODEL_PATH)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
if tok.pad_token is None: tok.pad_token = tok.eos_token
# 4-bit NF4: 14B ~8GB -> T4'e rahat sığar, generation OOM olmaz, kalite ~fp16
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                             device_map={'':0}); model.eval()   # tek GPU
print('4-bit yüklendi, GPU bellek:', round(torch.cuda.memory_allocated(0)/1e9,1),'GB')

base = [p for p in glob.glob('/kaggle/input/*') if os.path.exists(f'{p}/train.csv')]
if not base:
    hits = glob.glob('/kaggle/input/**/train.csv', recursive=True); base=[os.path.dirname(hits[0])]
base = base[0]
train = pd.read_csv(f'{base}/train.csv'); test = pd.read_csv(f'{base}/test_x.csv')
TEXT='mentor_feedback_text'; ID='student_id'; TGT='career_success_score'

def pick_shots(seed, n_per_band=2):
    rng = np.random.RandomState(seed); shots=[]
    tr = train.copy(); tr['_len']=tr[TEXT].str.len()
    for lo in range(40,100,10):
        sub = tr[(tr[TGT]>=lo)&(tr[TGT]<lo+10)].nsmallest(40,'_len')
        if len(sub): shots += list(sub.sample(min(n_per_band,len(sub)),random_state=rng).index)
    rng.shuffle(shots); return shots
def shot_block(idxs):
    lines=["Örnek değerlendirmeler ve gerçek skorları:"]
    for i in idxs:
        t=str(train.loc[i,TEXT])[:360].replace("\n"," ")
        lines.append(f'- Metin: "{t}" -> skor: {int(round(train.loc[i,TGT]))}')
    return "\n".join(lines)
SYS=("Sen bir teknik kariyer değerlendirme uzmanısın. Mentor geri bildiriminden öğrencinin "
     "KARİYER BAŞARI SKORUNU 0-100 tahmin et. Ölçek: 100=olağanüstü,77=tipik,40=ciddi eksik. "
     "Çoğu öğrenci 55-98 bandında. Örneklerin skorlama mantığını taklit et, ince ayrımları yakala.\n")
def make_prompts(texts, sb):
    msgs=[[{"role":"system","content":SYS+sb+'\nSADECE JSON döndür: {"skor": N}'},
           {"role":"user","content":t[:1100]}] for t in texts]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
def parse_one(s):
    m=re.search(r'"skor"\s*:\s*(-?\d+(?:\.\d+)?)',s) or re.search(r'(\d{1,3}(?:\.\d+)?)',s)
    return 77.0 if not m else float(np.clip(float(m.group(1)),0,100))

@torch.no_grad()
def predict(texts, tag, sb, bs=12):   # 14B -> küçük batch
    part=f'/kaggle/working/fs14_{tag}.csv'; done=0; rows=[]
    if os.path.exists(part): rows=pd.read_csv(part)['pred'].tolist(); done=len(rows); print(f"[{tag}] devam {done}")
    t0=time.time()
    for s in range(done,len(texts),bs):
        enc=tok(make_prompts(texts[s:s+bs],sb),return_tensors='pt',padding=True,truncation=True,max_length=1024).to(model.device)
        out=model.generate(**enc,max_new_tokens=14,do_sample=False,pad_token_id=tok.pad_token_id)
        dec=tok.batch_decode(out[:,enc['input_ids'].shape[1]:],skip_special_tokens=True)
        rows+=[parse_one(d) for d in dec]
        if (s//bs)%20==0:
            el=time.time()-t0; dn=len(rows)-done
            print(f"[{tag}] {len(rows)}/{len(texts)} ({el/60:.1f}dk ETA {(len(texts)-len(rows))/max(dn/el,1e-9)/60:.0f}dk)",flush=True)
        if (s//bs)%25==0: pd.DataFrame({'pred':rows}).to_csv(part,index=False)
    pd.DataFrame({'pred':rows}).to_csv(part,index=False); return np.array(rows)

# 2 örnek-seti ortala (14B yavaş -> 2 seed yeterli)
for which,frame in [('train',train),('test',test)]:
    acc=[predict(frame[TEXT].fillna('').tolist(), f'{which}_s{si}', shot_block(pick_shots(seed)))
         for si,seed in enumerate([11,29])]
    pred=np.mean(acc,0)
    pd.DataFrame({ID:frame[ID].values,'pred':pred}).to_csv(f'/kaggle/working/llm_pred2_{which}.csv',index=False)
    print(f"[{which}] BİTTİ mean={pred.mean():.1f} std={pred.std():.1f}")
tp=pd.read_csv('/kaggle/working/llm_pred2_train.csv')
print(f"\n>>> 14B few-shot corr(pred2,target)={np.corrcoef(tp['pred'],train[TGT])[0,1]:.4f}  (7B zero-shot 0.53; >0.58 KAZANÇ)")
print("İNDİR: llm_pred2_train.csv + llm_pred2_test.csv")
