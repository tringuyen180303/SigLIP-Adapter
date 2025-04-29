import os
import random
from PIL import Image
from datasets.utils import DatasetBase
import torch
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