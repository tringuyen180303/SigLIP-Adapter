import os, random, torch
from typing import List, Optional
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForVision2Seq, T5Tokenizer, T5ForConditionalGeneration
from datasets.utils import DatasetBase, Datum
TEMPLATE = (
    "This is a photo of a {cls_name}. "
    "Please describe the object’s shape, colour, texture, pose, background, "
    "and camera angle. Only mention what you can see."
)

class Caltech101Dataset(DatasetBase):
    def __init__(self, root, split='train', num_shots = 0, val_ratio=0.2, test_ratio=0.1, transform=None, seed=42, generate_desc=False):
        self.transform = transform
        self.template = TEMPLATE
        self.split = split
        self.generate_desc = generate_desc

        self.device = 'cuda' if split == 'train' and torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32

        self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')
        self.vlm = AutoModelForVision2Seq.from_pretrained(
            'HuggingFaceTB/SmolVLM-500M-Instruct', torch_dtype=dtype, _attn_implementation='eager'
        ).to(self.device).eval()
        self.s_tok = T5Tokenizer.from_pretrained('google-t5/t5-base')
        self.s_mod = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base').to(self.device).eval()

        categories_path = os.path.join(root, '101_ObjectCategories')
        class_names = sorted(d for d in os.listdir(categories_path) if os.path.isdir(os.path.join(categories_path, d)))
        self.class_to_idx = {cls: i for i, cls in enumerate(class_names)}
        all_items = []
        for cls in class_names:
            folder = os.path.join(categories_path, cls)
            for fname in os.listdir(folder):
                if fname.endswith(".jpg"):
                    all_items.append((os.path.join(folder, fname), self.class_to_idx[cls], cls.lower()))

        random.seed(seed)
        random.shuffle(all_items)
        n_total = len(all_items)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)

        self.train_items = [Datum(*item) for item in all_items[n_val + n_test:]]
        self.val_items = [Datum(*item) for item in all_items[:n_val]]
        self.test_items = [Datum(*item) for item in all_items[n_val:n_val + n_test]]

        if split == 'train':
            if num_shots > 0:
                few_shots = self.generate_fewshot_dataset(self.train_items, num_shots)
                self.items = few_shots
            else:
                self.items = self.train_items   # <-- FIXED this line
        elif split == 'val':
            self.items = self.val_items
        elif split == 'test':
            self.items = self.test_items
        else:
            raise ValueError("Invalid split name")


        self.inv_norm = transforms.Normalize(mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
                                             std=[1/s for s in [0.229, 0.224, 0.225]])
        self.to_pil = transforms.ToPILImage()
        self.cache = {}

        super().__init__(root_path=root, train_x=self.train_items, val=self.val_items, test=self.test_items)

    def __len__(self):
        return len(self.items)

    def _summ(self, txt):
        t = "summarize: " + txt.replace('\n', ' ')
        tok = self.s_tok(t, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        ids = self.s_mod.generate(**tok, max_length=256, min_length=64, num_beams=4, length_penalty=2.0)
        return self.s_tok.decode(ids[0], skip_special_tokens=True).strip()

    def __getitem__(self, idx):
        item = self.items[idx]
        path, label, cls_name = item.impath, item.label, item.classname
    
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
    
        if not self.generate_desc or self.split != 'train':
            return img, label, cls_name, ""
    
        key = os.path.basename(path)
        if key in self.cache:
            desc, summ = self.cache[key]
        else:
            pil = self.to_pil(self.inv_norm(img).clamp(0, 1))
            prompt = self.template.format(cls_name=cls_name.replace('_', ' '))
            msg = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
            chat = self.processor.apply_chat_template(msg, add_generation_prompt=True)
            inp = self.processor(text=chat, images=[pil], return_tensors='pt').to(self.device)
            with torch.no_grad():
                ids = self.vlm.generate(**inp, max_new_tokens=96, num_beams=4)
            desc = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            summ = self._summ(desc)
            self.cache[key] = (desc, summ)
    
        return img, label, cls_name, summ
