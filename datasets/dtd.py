import os, random, torch
from typing import List, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForVision2Seq, T5Tokenizer, T5ForConditionalGeneration

TEMPLATE = ["a photo of a {}, a texture class"]

class Datum:
    def __init__(self, impath: str, label: int, classname: str = None):
        self.impath, self.label, self.classname = impath, label, classname
    def __repr__(self):
        return f"Datum(impath={self.impath}, label={self.label}, classname={self.classname})"

class DatasetBase:
    def __init__(self, root_path=None, shots=0, train_x: Optional[List[Datum]] = None,
                 val: Optional[List[Datum]] = None, test: Optional[List[Datum]] = None):
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
    def __init__(self, root_path, split='train', num_shots=0, transform=None,
                 seed=42, generate_desc=True):
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.template = TEMPLATE
        self.generate_desc = generate_desc
        self.device = "cuda" if split == "train" and torch.cuda.is_available() else "cpu"

        # Gather all texture class directories
        class_dirs = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
        class_to_idx = {cls: i for i, cls in enumerate(class_dirs)}

        # Populate dataset items
        items = []
        for cls, label in class_to_idx.items():
            cls_path = os.path.join(root_path, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith(".jpg"):
                    impath = os.path.join(cls_path, fname)
                    items.append(Datum(impath, label, cls.lower()))

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
