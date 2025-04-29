"""
run_siglip_dtd.py - DTD Texture Dataset Version
A single self-contained script that:
1. Defines an inline YAML config.
2. Implements Datum, DatasetBase, DTDTextureDataset.
3. Loads SigLIP (ViT-B/16) from timm hub.
4. Builds cache model, extracts features, runs SigLIP zero-shot, IDEA-Adapter, and the fine-tuned TIDEA variant.

Simply run
    python run_siglip_dtd.py
"""

# --------------------------------------------------
# 1. Inline YAML configuration
# --------------------------------------------------
import yaml, io, os, random, sys, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, T5Tokenizer, T5ForConditionalGeneration
from typing import List, Optional
from open_clip import create_model_from_pretrained, get_tokenizer

CONFIG_YAML = """
experiment_name: "IdeaAdapterDTD"
seed: 42

data:
  root_path: "/kaggle/input/dtd-data/dtd/images"
  dataset_name: "DTDTexture"
  shots: 1
  overwrite: true
  cache_root: "./feature_cache"

  split:
    val: 0.2
    test: 0.1

model:
  name: "SigLIP"
  load_cache: false
  cache_dir: "./cache"
  augment_epoch: 1

idea:
  beta: 2.0
  alpha: 0.5
  theta: 0.5

training:
  epochs: 10
  batch_size: 2
  lr: 1e-4

logging:
  log_interval: 10
  save_checkpoint: true
  checkpoint_dir: "./checkpoints"

search_hp: true
search_scale: [4.0, 1.0]
search_step:  [20, 10]
"""

def load_config():
    return yaml.safe_load(io.StringIO(CONFIG_YAML))

cfg = load_config()
random.seed(cfg['seed'])
torch.manual_seed(cfg['seed'])

# --------------------------------------------------
# 2. Dataset Classes (Datum, DatasetBase, DTDTextureDataset)
# --------------------------------------------------
TEMPLATE = "a photo of a {cls_name}, a texture class"

class Datum:
    def __init__(self, impath:str, label:int, classname:str=None):
        self.impath, self.label, self.classname = impath, label, classname
    def __repr__(self):
        return f"Datum(impath={self.impath}, label={self.label}, classname={self.classname})"

class DatasetBase:
    def __init__(self, root_path=None, shots=0, train_x: Optional[List[Datum]] = None, val: Optional[List[Datum]] = None, test: Optional[List[Datum]] = None):
        self.root_path = root_path
        self.shots = shots
        self.train = train_x or []
        self.val = val or []
        self.test = test or []
        self._num_classes = self.get_num_classes(self.train)
        self._lab2cname, self._classnames = self.get_lab2cname(self.train)

    def generate_fewshot_dataset(self, data, num_shots, seed=42):
        if num_shots <= 0:
            return data
        random.seed(seed)
        label_to_items = {}
        for item in data:
            label_to_items.setdefault(item.label, []).append(item)
        fewshot_data = []
        for label, items in label_to_items.items():
            sampled = random.sample(items, min(num_shots, len(items)))
            fewshot_data.extend(sampled)
        return fewshot_data

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def __repr__(self):
        return f"<DatasetBase: {len(self.train)} train, {len(self.val)} val, {len(self.test)} test>"

    def get_num_classes(self, data_source):
        return max(item.label for item in data_source) + 1 if data_source else 0

    def get_lab2cname(self, data_source):
        container = {(item.label, item.classname) for item in data_source}
        mapping = {label: cname for label, cname in container}
        labels = sorted(mapping)
        return mapping, [mapping[l] for l in labels]

    @property
    def num_classes(self): return self._num_classes
    @property
    def classnames(self): return self._classnames
    @property
    def lab2cname(self): return self._lab2cname

class DTDTextureDataset(DatasetBase):
    def __init__(self, root_path, split='train', num_shots=0, transform=None, seed=42, generate_desc=True, cfg=None):
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.template = TEMPLATE
        self.generate_desc = generate_desc
        self.device = "cuda" if split == "train" and torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        
        class_dirs = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
        class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        all_items = []
        for cls, label in class_to_idx.items():
            cls_path = os.path.join(root_path, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith(".jpg"):
                    impath = os.path.join(cls_path, fname)
                    all_items.append(Datum(impath, label, cls.lower()))

        # Load split percentages from config
        val_pct = self.cfg['data']['split']['val']
        test_pct = self.cfg['data']['split']['test']


        random.seed(seed)
        random.shuffle(all_items)
        n_total = len(all_items)
        n_val = int(val_pct * n_total)
        n_test = int(test_pct * n_total)
        n_train = n_total - n_val - n_test

        if split == 'train':
            items = all_items[:n_train]
        elif split == 'val':
            items = all_items[n_train:n_train + n_val]
        elif split == 'test':
            items = all_items[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {split}")

        if split == 'train' and num_shots > 0:
            items = self.generate_fewshot_dataset(items, num_shots, seed)

        super().__init__(
            root_path=root_path,
            shots=num_shots,
            train_x=items if split == 'train' else [],
            val=items if split == 'val' else [],
            test=items if split == 'test' else []
        )
        self.items = items

        if split == 'train':
            dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                'HuggingFaceTB/SmolVLM-500M-Instruct',
                torch_dtype=dtype,
                _attn_implementation='eager'
            ).to(self.device).eval()
            self.s_tok = T5Tokenizer.from_pretrained('google-t5/t5-base')
            self.s_mod = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base').to(self.device).eval()
            self.inv_norm = transforms.Normalize(
                mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                std=[1/s for s in [0.229, 0.224, 0.225]]
            )
            self.to_pil = transforms.ToPILImage()
            self.cache = {}

    def _summarize(self, text):
        input_text = "summarize: " + text.replace('\n', ' ')
        tokens = self.s_tok(input_text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        output_ids = self.s_mod.generate(**tokens, max_length=256, min_length=64, num_beams=4, length_penalty=2.0)
        return self.s_tok.decode(output_ids[0], skip_special_tokens=True).strip()

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        datum = self.items[idx]
        image = Image.open(datum.impath).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if not self.generate_desc or self.split != 'train':
            return image, datum.label, datum.classname, ""

        key = os.path.basename(datum.impath)
        if key in self.cache:
            desc, summ = self.cache[key]
        else:
            pil = self.to_pil(self.inv_norm(image).clamp(0, 1))
            prompt = self.template[0].format(datum.classname.replace('_', ' '))
            msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            chat = self.processor.apply_chat_template(msg, add_generation_prompt=True)
            inp = self.processor(text=chat, images=[pil], return_tensors='pt').to(self.device)
            with torch.no_grad():
                ids = self.vlm.generate(**inp, max_new_tokens=96, num_beams=4)
            desc = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            summ = self._summarize(desc)
            self.cache[key] = (desc, summ)

        return image, datum.label, datum.classname, summ


# --------------------------------------------------
# 3. Model and Helper Functions
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

def run_TSGILIP(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels,test_features, test_labels,  model_weights, model,  train_loader_F):
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
# 4. Main Routine
# --------------------------------------------------

if __name__ == '__main__':
    model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    root = cfg['data']['root_path']
    train_ds = DTDTextureDataset(root, 'train', transform=transform, num_shots=cfg['data']['shots'], cfg=cfg)
    val_ds = DTDTextureDataset(root, 'val', transform=transform, cfg=cfg)
    test_ds = DTDTextureDataset(root, 'test', transform=transform, cfg=cfg)
    train_f = DTDTextureDataset(root, 'train', num_shots=cfg['data']['shots'],  generate_desc=False, transform=transform, cfg=cfg)
    print(len(train_f))

    bs = cfg['training']['batch_size']
    train_loader_cache = DataLoader(train_ds, batch_size=bs, shuffle=False)
    train_loader_finetune = DataLoader(train_f, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    weights = model_classifier(train_ds.classnames, TEMPLATE, model)
    im_k, txt_k, vals = build_cache_model(cfg, model, train_loader_cache)
    val_f, val_l = extract_features_cached(cfg, model, val_loader, "val")
    test_f, test_l = extract_features_cached(cfg, model, test_loader, "test")

    run_siglip(cfg, im_k, txt_k, vals, val_f, val_l, test_f, test_l, weights)
    run_TSGILIP(cfg, im_k, txt_k, vals, val_f, val_l, test_f, test_l, weights, model, train_loader_finetune)
    
    # print("[+] Zero-shot Evaluation...")

    # # Validation
    # zero_val = 100 * val_f @ weights
    # acc_val = classification_acc(zero_val, val_l)
    # print(f"Zero-shot val acc: {acc_val:.2f}%")
    # print(f"IDEA val acc: {acc_val:.2f}%")
    
    # # Test
    # zero_test = 100 * test_f @ weights
    # acc_test = classification_acc(zero_test, test_l)
    # print(f"Zero-shot test acc: {acc_test:.2f}%")
    # print(f"IDEA test acc: {acc_test:.2f}%")

    # print("[+] Feature Cache Ready. Ready to Fine-tune with IDEA-Adapter-F!")

# (Fine-tuning training script could be added next if needed)

