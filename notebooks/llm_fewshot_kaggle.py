# ===================== LLM FEW-SHOT DIRECT PREDICTOR v2 — TEK HÜCRE =====================
# Zero-shot corr 0.53 verdi. Few-shot: prompt'a train'den skor-bandına stratified 12 etiketli
# örnek göm -> kalibrasyon güçlenir. 2 farklı örnek-seti ile ortala. Çıktı: llm_pred2_{train,test}.csv
import os, glob, json, re, time
import numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "GPU YOK! Settings->Accelerator->GPU T4 x2 -> Restart"
print("GPU:", torch.cuda.get_device_name(0), "x", torch.cuda.device_count())

cands = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json', recursive=True)
         if any(k in p.lower() for k in ('qwen','gemma'))]
assert cands, "Model yok! Add Input -> Models -> 'Qwen2.5 7B Instruct'"
MODEL_PATH = sorted(cands, key=len)[0]; print("MODEL:", MODEL_PATH)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map='auto'); model.eval()

base = [p for p in glob.glob('/kaggle/input/*') if os.path.exists(f'{p}/train.csv')]
if not base:
    hits = glob.glob('/kaggle/input/**/train.csv', recursive=True); base=[os.path.dirname(hits[0])]
base = base[0]
train = pd.read_csv(f'{base}/train.csv'); test = pd.read_csv(f'{base}/test_x.csv')
TEXT='mentor_feedback_text'; ID='student_id'; TGT='career_success_score'

# skor-bandına stratified örnek seç (her bant ~2 örnek), kısa metin tercih (token tasarrufu)
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
        t = str(train.loc[i,TEXT])[:380].replace("\n"," ")
        lines.append(f'- Metin: "{t}" -> skor: {int(round(train.loc[i,TGT]))}')
    return "\n".join(lines)

SYS = ("Sen bir teknik kariyer değerlendirme uzmanısın. Mentor geri bildiriminden öğrencinin "
       "KARİYER BAŞARI SKORUNU 0-100 tahmin et. Ölçek: 100=olağanüstü,77=tipik,40=ciddi eksik. "
       "Çoğu öğrenci 55-98 bandında. Aşağıdaki örneklerin skorlama mantığını taklit et.\n")
def make_prompts(texts, shotblock):
    msgs=[[{"role":"system","content":SYS+shotblock+'\nSADECE JSON döndür: {"skor": N}'},
           {"role":"user","content":t[:1200]}] for t in texts]
    return [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]

def parse_one(s):
    m=re.search(r'"skor"\s*:\s*(-?\d+(?:\.\d+)?)',s) or re.search(r'(\d{1,3}(?:\.\d+)?)',s)
    if not m: return 77.0
    return float(np.clip(float(m.group(1)),0,100))

@torch.no_grad()
def predict(texts, tag, shotblock, bs=16):
    part=f'/kaggle/working/fs_{tag}.csv'; done=0; rows=[]
    if os.path.exists(part): rows=pd.read_csv(part)['pred'].tolist(); done=len(rows); print(f"[{tag}] devam {done}")
    t0=time.time()
    for s in range(done,len(texts),bs):
        enc=tok(make_prompts(texts[s:s+bs],shotblock),return_tensors='pt',padding=True,truncation=True,max_length=1100).to(model.device)
        out=model.generate(**enc,max_new_tokens=16,do_sample=False,pad_token_id=tok.pad_token_id)
        dec=tok.batch_decode(out[:,enc['input_ids'].shape[1]:],skip_special_tokens=True)
        rows+=[parse_one(d) for d in dec]
        if (s//bs)%10==0:
            el=time.time()-t0; dn=len(rows)-done
            print(f"[{tag}] {len(rows)}/{len(texts)} ({el/60:.1f}dk ETA {(len(texts)-len(rows))/max(dn/el,1e-9)/60:.0f}dk)",flush=True)
        if (s//bs)%20==0: pd.DataFrame({'pred':rows}).to_csv(part,index=False)
    pd.DataFrame({'pred':rows}).to_csv(part,index=False); return np.array(rows)

# 2 örnek-seti ortala
for which,frame in [('train',train),('test',test)]:
    acc=[]
    for si,seed in enumerate([11,29,47]):
        sb=shot_block(pick_shots(seed))
        acc.append(predict(frame[TEXT].fillna('').tolist(), f'{which}_s{si}', sb))
    pred=np.mean(acc,0)
    out=pd.DataFrame({ID:frame[ID].values,'pred':pred}); out.to_csv(f'/kaggle/working/llm_pred2_{which}.csv',index=False)
    print(f"[{which}] BİTTİ mean={pred.mean():.1f} std={pred.std():.1f}")

tp=pd.read_csv('/kaggle/working/llm_pred2_train.csv')
print(f"\n>>> few-shot corr(pred2,target)={np.corrcoef(tp['pred'],train[TGT])[0,1]:.4f}  (zero-shot 0.53; >0.58 ise kazanç)")
print("İNDİR: llm_pred2_train.csv + llm_pred2_test.csv")
