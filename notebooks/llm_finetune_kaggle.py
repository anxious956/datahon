# ============ Qwen2.5-7B LoRA REGRESYON fine-tune (text -> career_success_score) ============
# Metin sentetik (hedeften üretilmiş). berturk-base corr 0.70 verdi. LLM daha güçlü geri-çözücü
# -> corr 0.80+ hedefi = top-10 kaldıracı. 4-bit + LoRA -> T4'e sığar. 5-fold OOF + test.
# KURULUM: TEMİZ yeni notebook -> Add Input: Qwen2.5 7B Instruct + Datathon 2026 -> GPU T4 -> Run All
# Çıktı: oof_qwenft_train.npy + qwenft_test.npy -> indir, bana yükle. Süre ~3-5 saat.
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','-U','peft','bitsandbytes>=0.46.1','accelerate'],check=False)
import glob, random, numpy as np, pandas as pd, torch
assert torch.cuda.is_available(), "GPU YOK! Settings->Accelerator->GPU T4 -> Restart"
print("GPU:", torch.cuda.get_device_name(0))
import gc; gc.collect(); torch.cuda.empty_cache()

SEED,N_SPLITS=42,5
MAX_LEN, LR, EPOCHS, BS, ACCUM = 384, 1e-4, 2, 8, 2
ID,TARGET,TEXT,YEAR='student_id','career_success_score','mentor_feedback_text','application_year'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

cands=[os.path.dirname(p) for p in glob.glob('/kaggle/input/**/config.json',recursive=True) if 'qwen' in p.lower() and '7b' in p.lower()]
assert cands, "Qwen 7B yok! Add Input -> Models -> Qwen2.5 7B Instruct"
MODEL=sorted(cands,key=len)[0]; print("MODEL:",MODEL)

base=[p for p in glob.glob('/kaggle/input/*') if os.path.exists(f'{p}/train.csv')]
if not base:
    h=glob.glob('/kaggle/input/**/train.csv',recursive=True); base=[os.path.dirname(h[0])]
base=base[0]
train=pd.read_csv(f'{base}/train.csv'); test=pd.read_csv(f'{base}/test_x.csv')
y=train[TARGET].values.astype('float32')
txt_tr=train[TEXT].fillna('').tolist(); txt_te=test[TEXT].fillna('').tolist()
from sklearn.model_selection import StratifiedKFold
tbin=pd.qcut(train[TARGET].values,10,labels=False,duplicates='drop')
strat=train[YEAR].astype(str)+'_'+pd.Series(tbin).astype(str)
folds=list(StratifiedKFold(N_SPLITS,shuffle=True,random_state=SEED).split(train,strat))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
tok=AutoTokenizer.from_pretrained(MODEL);
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side='right'
def enc(texts): return tok(texts,truncation=True,max_length=MAX_LEN,padding='max_length',return_tensors='pt')
Etr=enc(txt_tr); Ete=enc(txt_te)
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
dev='cuda'

@torch.no_grad()
def predict(model,E,bs=16):
    model.eval(); out=[]
    ids,am=E['input_ids'],E['attention_mask']
    for s in range(0,len(ids),bs):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            o=model(input_ids=ids[s:s+bs].to(dev),attention_mask=am[s:s+bs].to(dev)).logits.squeeze(-1)
        out.append(o.float().cpu().numpy())
    return np.concatenate(out)

oof=np.zeros(len(train)); test_pred=np.zeros(len(test))
for fi,(tr,va) in enumerate(folds):
    print(f"\n===== FOLD {fi} =====",flush=True)
    mu,sd=float(y[tr].mean()),float(y[tr].std())
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=1,quantization_config=bnb,device_map={'':0})
    model.config.pad_token_id=tok.pad_token_id
    model=prepare_model_for_kbit_training(model)
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='SEQ_CLS',
                    target_modules=['q_proj','k_proj','v_proj','o_proj'])
    model=get_peft_model(model,lora); model.config.use_cache=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR)
    scaler=torch.cuda.amp.GradScaler()
    idx=np.array(tr)
    for ep in range(EPOCHS):
        model.train(); np.random.shuffle(idx); opt.zero_grad()
        for bi,s in enumerate(range(0,len(idx),BS)):
            b=idx[s:s+BS]
            ii=Etr['input_ids'][b].to(dev); aa=Etr['attention_mask'][b].to(dev)
            yb=torch.tensor((y[b]-mu)/sd,dtype=torch.float16).to(dev)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out=model(input_ids=ii,attention_mask=aa).logits.squeeze(-1)
                loss=torch.nn.functional.mse_loss(out.float(),yb.float())/ACCUM
            scaler.scale(loss).backward()
            if (bi+1)%ACCUM==0:
                scaler.step(opt); scaler.update(); opt.zero_grad()
            if bi%100==0: print(f"  ep{ep} step{bi} loss{loss.item()*ACCUM:.3f}",flush=True)
    pv=predict(model,{'input_ids':Etr['input_ids'][va],'attention_mask':Etr['attention_mask'][va]})
    oof[va]=pv*sd+mu
    test_pred+=(predict(model,Ete)*sd+mu)/N_SPLITS
    c=np.corrcoef(oof[va],y[va])[0,1]; print(f"  fold{fi} corr={c:.4f}",flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

oof=np.clip(oof,0,100); test_pred=np.clip(test_pred,0,100)
np.save('/kaggle/working/oof_qwenft_train.npy',oof); np.save('/kaggle/working/qwenft_test.npy',test_pred)
print(f"\n>>> Qwen-FT corr(oof,target)={np.corrcoef(oof,y)[0,1]:.4f}  (berturk 0.70; >0.78 = BÜYÜK kazanç)")
print("İNDİR: oof_qwenft_train.npy + qwenft_test.npy")
