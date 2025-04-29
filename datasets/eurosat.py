import os, random, torch
import pandas as pd
from typing import List, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForVision2Seq, T5Tokenizer, T5ForConditionalGeneration
from datasets.utils import DatasetBase, Datum

TEMPLATE = "a satellite image of a {cls_name}, a type of terrain"

class EuroSATDataset(DatasetBase):
    def __init__(self, root_path, split='train', num_shots=0, transform=None,
                 seed=42, generate_desc=True, val_ratio=0.2):
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.template = TEMPLATE
        self.generate_desc = generate_desc
        self.device = "cuda" if split == "train" and torch.cuda.is_available() else "cpu"

        def load_csv_items(csv_file):
            df = pd.read_csv(csv_file)
            if 'Unnamed: 0' in df.columns:
                df = df.drop(columns=['Unnamed: 0'])
            return [Datum(
                        os.path.join(root_path, row['Filename']),
                        int(row['Label']),
                        row['ClassName'].lower().replace(" ", "_"))
                    for _, row in df.iterrows()]

        train_items = load_csv_items(os.path.join(root_path, "train.csv"))
        split_items = load_csv_items(os.path.join(root_path, f"{split}.csv"))

        if split == 'train' and num_shots > 0:
            train_items = self.generate_fewshot_dataset(train_items, num_shots, seed)

        super().__init__(
            root_path=root_path,
            shots=num_shots,
            train_x=train_items,
            val=split_items if split == 'validation' else [],
            test=split_items if split == 'test' else []
        )
        self.items = train_items if split == 'train' else split_items

        if split == 'train':
            dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
            attn = 'eager'
            self.processor = AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct')
            self.vlm = AutoModelForVision2Seq.from_pretrained(
                'HuggingFaceTB/SmolVLM-500M-Instruct',
                torch_dtype=dtype,
                _attn_implementation=attn
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
        output_ids = self.s_mod.generate(**tokens, max_length=256, min_length=64,
                                         num_beams=4, length_penalty=2.0)
        return self.s_tok.decode(output_ids[0], skip_special_tokens=True).strip()

    def __len__(self):
        return len(self.items)

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
