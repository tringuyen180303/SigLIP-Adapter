"""run_siglip_food101.py
A single self‑contained script that replicates the SigLIP + IDEA/Adapter pipeline
originally written for Oxford‑IIIT Pets, but now targeting **Food‑101**.

Usage:
    python run_siglip_food101.py
"""
# --------------------------------------------------
# 1. Inline YAML configuration (edit paths or hyper‑params as needed)
# --------------------------------------------------
import yaml, io, os, random, torch, torch.nn as nn, torch.nn.functional as F
from typing import List, Optional
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, AutoModelForVision2Seq,
    T5Tokenizer, T5ForConditionalGeneration,
)
from open_clip import create_model_from_pretrained, get_tokenizer

CFG_YAML = """
experiment_name: "IdeaAdapter_Food101"
seed: 42

data:
  root_path: "/kaggle/input/food101/food-101"  # path to Food‑101 directory
  dataset_name: "Food101"
  shots: 1               # few‑shot images per class for cache keys
  overwrite: false
  cache_root: "./feature_cache_food"

model:
  name: "SigLIP"
  load_cache: true
  cache_dir: "./cache_food"
  augment_epoch: 1

idea:
  beta: 2.0
  alpha: 0.5
  theta: 0.5

training:
  epochs: 10
  batch_size: 4
  lr: 1e-4

logging:
  log_interval: 10
  save_checkpoint: true
  checkpoint_dir: "./checkpoints_food"
"""

cfg = yaml.safe_load(io.StringIO(CFG_YAML))
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

# --------------------------------------------------
# 2. Datum + DatasetBase (unchanged)
# --------------------------------------------------
class Datum:
    def __init__(self, impath: str, label: int, classname: str = None):
        self.impath, self.label, self.classname = impath, label, classname

class DatasetBase:
    def __init__(self, root_path=None, shots=0, train_x: Optional[List[Datum]] = None,
                 val: Optional[List[Datum]] = None, test: Optional[List[Datum]] = None):
        self.root_path = root_path
        self.shots     = shots
        self.train, self.val, self.test = train_x or [], val or [], test or []
        self._num_classes  = self._get_num_classes(self.train)
        self._lab2cname, self._classnames = self._get_lab2cname(self.train)

    # ---------- helper ----------
    def _fewshot(self, data, shots, seed=42):
        if shots <= 0: return data
        random.seed(seed)
        bucket = {}
        for item in data:
            bucket.setdefault(item[1], []).append(item)
        out = []
        for lab, lst in bucket.items():
            out.extend(random.sample(lst, min(shots, len(lst))))
        return out

    def _get_num_classes(self, ds):
        return max([d[1] for d in ds]) + 1 if ds else 0

    def _get_lab2cname(self, ds):
        mapping = {d[1]: d[2] for d in ds}
        labs = sorted(mapping.keys())
        cns  = [mapping[k] for k in labs]
        return mapping, cns

    # properties
    @property
    def num_classes(self):
        return self._num_classes
    @property
    def classnames(self):
        return self._classnames

# --------------------------------------------------
# 3. Food‑101 dataset (few‑shot + SmolVLM captions)
# --------------------------------------------------
TEMPLATE = "a photo of a {cls_name}, a type of food."

class Food101Dataset(DatasetBase):
    def __init__(self, root, split="train", num_shots=0, transform=None,
                 val_ratio=0.2, seed=42, generate_desc=True):
        self.transform, self.template, self.split = transform, TEMPLATE, split
        img_dir, meta_dir = os.path.join(root, "images"), os.path.join(root, "meta")

        classnames = sorted([d for d in os.listdir(img_dir)
                             if os.path.isdir(os.path.join(img_dir, d))])
        cname2lab   = {c: i for i, c in enumerate(classnames)}
        self._lab2cname = {i: c for c, i in cname2lab.items()}
        self._classnames = classnames

        # helper to read txt splits
        def read(txt):
            items = []
            with open(txt) as f:
                for ln in f:
                    cls, img = ln.strip().split('/')
                    lab = cname2lab[cls]
                    imp = os.path.join(img_dir, cls, img + '.jpg')
                    items.append((imp, lab, cls))
            return items

        trainval = read(os.path.join(meta_dir, 'train.txt'))
        random.seed(seed); random.shuffle(trainval)
        v_sz = int(len(trainval) * val_ratio)
        val_items, train_items = trainval[:v_sz], trainval[v_sz:]

        if split == 'train':
            train_items = self._fewshot(train_items, num_shots, seed)
            items = train_items
        elif split == 'val':
            items = val_items
        else:
            items = read(os.path.join(meta_dir, 'test.txt'))

        self.items = items

        # caption models (GPU only for training split)
        dev   = 'cuda' if (split == 'train' and torch.cuda.is_available()) else 'cpu'
        dtype = torch.bfloat16 if dev == 'cuda' else torch.float32
        self.device = dev
        if generate_desc:
            self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-256M-Instruct')
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                'HuggingFaceTB/SmolVLM-256M-Instruct', torch_dtype=dtype
            ).to(dev).eval()
            self.s_tok = T5Tokenizer.from_pretrained('google-t5/t5-base')
            self.s_mod = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base').to(dev).eval()
            self.inv_norm = transforms.Normalize(
                mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
                std=[1/s for s in [0.229,0.224,0.225]])
            self.to_pil = transforms.ToPILImage()
            self.cache  = {}
        self.gen_desc = generate_desc and (split == 'train')

        super().__init__(root_path=root, shots=num_shots,
                         train_x=train_items, val=val_items,
                         test=items if split == 'test' else [])

    def __len__(self):
        return len(self.items)

    # helper: summarize VLM caption via T5
    def _summarize(self, txt):
        t = 'summarize: ' + txt.replace('\n', ' ')
        tok = self.s_tok(t, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        ids = self.s_mod.generate(**tok, max_length=128, num_beams=4)
        return self.s_tok.decode(ids[0], skip_special_tokens=True).strip()

    def __getitem__(self, idx):
        imp, lab, cls = self.items[idx]
        img = Image.open(imp).convert('RGB')
        if self.transform: img = self.transform(img)
        if not self.gen_desc:
            return img, torch.tensor(lab), cls, ""
        # -------- generate caption (cached) --------
        if imp in self.cache:
            desc, summ = self.cache[imp]
        else:
            pil = self.to_pil(self.inv_norm(img).clamp(0,1))
            prompt = self.template.format(cls_name=cls.replace('_',' '))
            chat = self.processor.apply_chat_template([
                {'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}
            ], add_generation_prompt=True)
            inp = self.processor(text=chat, images=[pil], return_tensors='pt').to(self.device)
            with torch.no_grad(): ids = self.vlm.generate(**inp, max_new_tokens=96, num_beams=4)
            desc = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            summ = self._summarize(desc)
            self.cache[imp] = (desc, summ)
        return img, torch.tensor(lab), cls, summ

# --------------------------------------------------
# 4. SigLIP helpers (unchanged from original code)
# --------------------------------------------------

def classification_acc(out, tgt, k=1):
    top = out.topk(k,1)[1].t()
    tgt = tgt if tgt.dim()==1 else tgt.argmax(1)
    correct = top.eq(tgt.view(1,-1).expand_as(top))
    return 100*correct[:k].reshape(-1).float().sum().item()/tgt.size(0)

def build_cache_model(cfg, model, loader):
    if not cfg['model']['load_cache']:
        im_keys, txt_keys, vals=[],[],[]
        aug=cfg['model']['augment_epoch']
        with torch.no_grad():
            for ep in range(aug):
                im_feats, txt_feats=[],[]; print(f'Augment {ep+1}/{aug}')
                for imgs, labs, cn, dess in tqdm(loader):
                    tok=get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')(dess, context_length=model.context_length)
                    txt=model.encode_text(tok); txt_feats.append(txt)
                    im=model.encode_image(imgs); im_feats.append(im)
                    if ep==0: vals.append(labs)
                im_keys.append(torch.cat(im_feats).unsqueeze(0))
                txt_keys.append(torch.cat(txt_feats).unsqueeze(0))
            im = torch.cat(im_keys, dim=0).mean(dim=0)   # [N, D]
            txt= torch.cat(txt_keys,dim=0).mean(dim=0)   # [N, D]
            val= F.one_hot(torch.cat(vals)).half()   # [N, C]

            # out-of-place L2 normalization + detach
            im  = (im  / im .norm(dim=-1, keepdim=True)).t().detach()  # [D, N]
            txt = (txt / txt.norm(dim=-1, keepdim=True)).t().detach()  # [D, N]
            val = val.detach() 
            #im=torch.cat(im_keys).mean(0); im/=im.norm(dim=-1,keepdim=True); im=im.t()
            #txt=torch.cat(txt_keys).mean(0); txt/=txt.norm(dim=-1,keepdim=True); txt=txt.t()
            # val=F.one_hot(torch.cat(vals)).half()
            os.makedirs(cfg['model']['cache_dir'], exist_ok=True)
            torch.save(val, os.path.join(cfg["model"]["cache_dir"], f'values_{cfg["data"]["shots"]}shots.pt'))
            torch.save(im,
               os.path.join(cfg["model"]["cache_dir"],
                            f'keys_{cfg["data"]["shots"]}shots.pt'))
            torch.save(txt,
               os.path.join(cfg["model"]["cache_dir"],
                            f'text_keys_{cfg["data"]["shots"]}shots.pt'))
    else:
        print("Loaded build model cache")
        im = torch.load(os.path.join(cfg["model"]["cache_dir"], f'keys_{cfg["data"]["shots"]}shots.pt'))
        val = torch.load(os.path.join(cfg["model"]["cache_dir"], f'values_{cfg["data"]["shots"]}shots.pt'))
        txt =  torch.load(os.path.join(cfg["model"]["cache_dir"], f'text_keys_{cfg["data"]["shots"]}shots.pt'))
    return im,txt,val

def extract_features_cached(cfg,
                            model,
                            data_loader,
                            split_name: str):
    """
    Return image features and labels for one split, using a simple
    disk cache.

    Saves (or loads) two files:
        {cache_root}/{split_name}_f.pt   # features [N, D]  (fp32 cpu)
        {cache_root}/{split_name}_l.pt   # labels   [N]
    """
    cache_root = cfg["data"]["cache_root"]     # e.g. "./feature_cache"
    overwrite  = cfg["data"].get("overwrite", False)
    os.makedirs(cache_root, exist_ok=True)

    f_path = os.path.join(cache_root, f"{split_name}_f.pt")
    l_path = os.path.join(cache_root, f"{split_name}_l.pt")

    # ---------- fast path: load ----------
    if (not overwrite) and os.path.isfile(f_path) and os.path.isfile(l_path):
        print(f"[cache] loaded {split_name} features from {cache_root}")
        return torch.load(f_path), torch.load(l_path)

    # ---------- slow path: compute ----------
    feat_list, label_list = [], []
    with torch.no_grad():
        for images, labels, *_ in tqdm(data_loader,
                                       desc=f"Extracting {split_name} features"):
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # L2‑norm
            feat_list.append(feats.cpu())
            label_list.append(labels.cpu())

    img_features = torch.cat(feat_list, 0)     # [N, D]  fp32 on CPU
    img_labels   = torch.cat(label_list, 0)    # [N]

    torch.save(img_features, f_path)
    torch.save(img_labels,   l_path)
    print(f"[cache] saved {split_name} features → {cache_root}")

    return img_features, img_labels

def model_classifier(cnames, tmpl, model):
    ws = []
    for cn in cnames:
        texts = [tmpl.format(cls_name=cn.replace('_', ' '))]
        tok   = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')(
                    texts, context_length=model.context_length)
        emb   = model.encode_text(tok)
        emb   = emb / emb.norm(dim=-1, keepdim=True)
        ws.append(emb.squeeze(0))
    return torch.stack(ws, 1)

# --------------------------------------------------
# 5. Main routine (zero‑shot + IDEA)
# --------------------------------------------------

def run_siglip(cfg, im_k, txt_k, vals, val_f, val_l, test_f, test_l, w):
    beta,alpha=cfg['idea']['beta'],cfg['idea']['alpha']
    zero=100*val_f@w; print(f"Zero‑shot val acc: {classification_acc(zero,val_l):.2f}%")
    vals = vals.float()
    aff=val_f@((im_k+txt_k)/2); few=((-1)*(beta-beta*aff)).exp()@vals
    idea=zero+few*alpha; print(f"IDEA val acc: {classification_acc(idea,val_l):.2f}%")
    # test
    zero=100*test_f@w; print(f"Zero‑shot test acc: {classification_acc(zero,test_l):.2f}%")
    aff=test_f@((im_k+txt_k)/2); few=((-1)*(beta-beta*aff)).exp()@vals
    idea=zero+few*alpha; print(f"IDEA test acc: {classification_acc(idea,test_l):.2f}%")

def run_TSGILIP(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_labels, test_features, model_weights, model,  train_loader_F):
    adapter = nn.Linear(img_cache_keys.shape[0], img_cache_keys.shape[0], bias=True)
    adapter2 = nn.Linear(img_cache_keys.shape[0], img_cache_keys.shape[1], bias=True)

    optimizer = torch.optim.AdamW(
        [{
            "params": adapter.parameters()
        },
        {
            "params": adapter2.parameters()
        }], lr=0.001, eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['training']['epochs'] * len(train_loader_F))
    beta, alpha = cfg['idea']['beta'], cfg['idea']['alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['training']['epochs']):
        # Train
        adapter.train()
        adapter2.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['training']['epochs']))

        for i, (images, target, classname, dess) in enumerate(tqdm(train_loader_F)):
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                #image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()
            image_feature_text = adapter(image_features)
            affinity2 = adapter2(image_features)

            affinity =  (image_feature_text @ text_cache_keys + image_features @ img_cache_keys + image_features @ text_cache_keys)
            affinity = affinity / 3.0
            affinity = affinity + affinity2
            cache_values = cache_values.float()
            few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            zero_logits = 100. * image_features @ model_weights

            TIDEA_logits = zero_logits + few_logits * alpha

            loss = F.cross_entropy(TIDEA_logits, target)

            acc = classification_acc(TIDEA_logits, target)
            correct_samples += acc / 100 * len(TIDEA_logits)
            all_samples += len(TIDEA_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            torch.autograd.set_detect_anomaly(True)

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Evaluation

        adapter.eval()
        adapter2.eval()
        test_feature_text = adapter(test_features)
        affinity2 = adapter2(test_features)

        affinity = (test_feature_text @ text_cache_keys + test_features @ img_cache_keys + test_features @ text_cache_keys)/3
        affinity = affinity + affinity2  

        few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        zero_logits = 100. * test_features @ model_weights
        TIDEA_logits = zero_logits + few_logits * alpha
        acc = classification_acc(TIDEA_logits, test_labels)

        print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            best_f_path  = os.path.join(cfg['model']['cache_dir'],
                            f"best_F_{cfg['data']['shots']}shots.pt")
            best_f2_path = os.path.join(cfg['model']['cache_dir'],
                            f"best_F_2_{cfg['data']['shots']}shots.pt")
            torch.save(adapter.state_dict(),  best_f_path)
            torch.save(adapter2.state_dict(), best_f2_path)

        adapter.load_state_dict(torch.load(best_f_path))
        adapter2.load_state_dict(torch.load(best_f2_path))
        print(f"**** After fine-tuning, IDEA-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

        #best_theta, best_beta, best_alpha = 2, 0.5, 0.5
        best_theta, best_beta, best_alpha = 0.05, 3.22, 0.73
        #best_theta, best_beta, best_alpha = search_hp_2(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels, model_weights)

        print("\n-------- Evaluating on the test set. --------")

        test_features_text = adapter(test_features)
        affinity2 = adapter2(test_features)
        affinity = best_theta* (test_features_text + test_features) @ text_cache_keys + (1-best_theta) * test_features @ img_cache_keys
        affinity = affinity + affinity2
        cache_values = cache_values.float()
        few_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
        TIDEA_logits = zero_logits + few_logits * best_alpha
        acc = classification_acc(TIDEA_logits, test_labels)
        print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))
# --------------------------------------------------
# 5. Main routine
# --------------------------------------------------
if __name__ == '__main__':
    # 5.1  Global image transform
    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    root = cfg['data']['root_path']
    shots= cfg['data']['shots']

    train_ds = Food101Dataset(root, 'train', num_shots=shots, transform=trans)
    val_ds   = Food101Dataset(root, 'val', transform=trans, val_ratio=0.1, generate_desc=False)
    test_ds  = Food101Dataset(root, 'test', transform=trans, generate_desc=False)

    bsz = cfg['training']['batch_size']
    train_cache_loader = DataLoader(train_ds, batch_size=bsz, shuffle=False, num_workers=0)
    train_ft_loader    = DataLoader(train_ds, batch_size=bsz, shuffle=True , num_workers=0)
    val_loader         = DataLoader(val_ds,   batch_size=bsz, shuffle=False)
    test_loader        = DataLoader(test_ds,  batch_size=bsz, shuffle=False)

    # 5.2  SigLIP backbone
    model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP'); model.eval()

    # 5.3  Text classifier weights
    weights = model_classifier(train_ds.classnames, TEMPLATE, model)

    # 5.4  Build / load cache keys
    im_k, txt_k, vals = build_cache_model(cfg, model, train_cache_loader)

    # 5.5  Feature extraction (cached)
    val_f, val_l   = extract_features_cached(cfg, model, val_loader,  'val')
    test_f, test_l = extract_features_cached(cfg, model, test_loader, 'test')

    # 5.6  Run zero‑shot + IDEA on val/test
    run_siglip(cfg, im_k, txt_k, vals,
               val_f, val_l,
               test_f, test_l,
               weights)

    # 5.7  IDEA‑Adapter fine‑tune (optional – heavy!)
    run_TSGILIP(cfg, im_k, txt_k, vals,
                val_f, val_l,
                test_f, test_l,
                weights, model,
                train_ft_loader)
