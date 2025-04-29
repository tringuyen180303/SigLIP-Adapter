import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.utils import DatasetBase, Datum
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    T5ForConditionalGeneration,
    T5Tokenizer,
)


#template = ["a photo of a {}, a type of pet"]
template = "This is a photo of a {cls_name}. " \
           "Please describe the object’s shape, colour, texture, pose, background, " \
           "and camera angle. Only mention what you can see."

class OxfordPetsDataset(DatasetBase):
    def __init__(self, root_path, split='train', num_shots=0, transform=None, val_ratio=0.2, seed=42):
        self.root_path = root_path
        self.image_dir = os.path.join(self.root_path, "images")
        self.annotation_dir = os.path.join(self.root_path, "annotations")
        self.transform = transform
        self.template = template
        self.split = split
        # Always load full trainval (for class information)
        full_trainval = self.read_split(os.path.join(self.annotation_dir, "trainval.txt"))
        train_items, val_items = self.split_trainval(full_trainval, val_ratio, seed)
        
        # Set splits for train/val/test
        if split in ['train', 'val']:
            self.train = train_items
            self.val = val_items
            self.items = self.train if split == 'train' else self.val
            # Apply few-shot sampling if specified.
            if num_shots > 0:
                self.train = self.generate_fewshot_dataset(self.train, num_shots)
                self.items = self.train
        elif split == 'test':
            test_items = self.read_split(os.path.join(self.annotation_dir, "test.txt"))
            desired_size = len(val_items)
            if len(test_items) > desired_size:
                random.seed(seed)
                test_items = random.sample(test_items, desired_size)
            self.items = test_items
            self.test = self.items
            # Use training split from trainval to compute number of classes
            self.train = train_items

        # Load SmolVLM and T5
        device       = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype  = torch.bfloat16 if device=="cuda" else torch.float32
        attn_impl    = "flash_attention_2" if device=="cuda" else "eager"
        
        self.processor = AutoProcessor.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
        )
        self.vlm_model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            torch_dtype=torch_dtype,
            _attn_implementation=attn_impl
        ).to(device).eval()

        self.summ_tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
        self.summ_model     = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")\
                                                            .to(device).eval()

        self.device = device

        # 3) cache for captions so we don’t regen every epoch
        self._caption_cache = {}

        # 4) helper to un-normalize back to PIL
        inv_norm = transforms.Normalize(
            mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
            std=[1/s for s in [0.229,0.224,0.225]]
        )
        self.to_pil = transforms.ToPILImage()
        self.inv_norm = inv_norm

        # Now, pass non-empty lists to the base class
        super().__init__(root_path=root_path, shots=seed,
                         train_x=self.train if self.train else [],
                         val=self.val if split in ['train', 'val'] else [],
                         test=self.test if split == 'test' else [])
 
    def read_split(self, file_path):
        """
        Reads an annotation file and returns a list of tuples:
        (impath, label, classname).
        Assumes each line in the file has four fields.
        """
        items = []
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            img_name, label, _, _ = parts
            img_name = img_name + ".jpg"
            impath = os.path.join(self.image_dir, img_name)
            label = int(label) - 1 
            classname = "_".join(img_name.split("_")[:-1]).lower()
            items.append((impath, label, classname))
        return items
    
    def split_trainval(self, items, val_ratio, seed):
        """
        Shuffles and splits 'items' into training and validation sets.
        """
        random.seed(seed)
        random.shuffle(items)
        total = len(items)
        val_size = int(total * val_ratio)
        val_items = items[:val_size]
        train_items = items[val_size:]
        return train_items, val_items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        impath, label, classname = self.items[index]
        image = Image.open(impath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.split != 'train':
            # skip expensive generation; give a placeholder
            return image, label, classname, ""
        
        cache_key = os.path.basename(impath)
        if cache_key in self._caption_cache:
            desc, summary = self._caption_cache[cache_key]
        else:
            # un‑normalize → PIL for processor
            pil_img = self.to_pil(self.inv_norm(image).clamp(0,1))

            prompt = template.format(cls_name=classname.replace("_"," "))
            messages = [{
                "role":"user",
                "content":[
                    {"type":"image"},
                    {"type":"text", "text": prompt}
                ]
            }]
            txt_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(
                text=txt_prompt,
                images=[pil_img],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out_ids = self.vlm_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=4
                )
            desc = self.processor.batch_decode(
                out_ids, skip_special_tokens=True
            )[0].strip()
            summary = self.summarize_text(desc)

            self._caption_cache[cache_key] = (desc, summary)

        return image, label, classname, summary
    
    def summarize_text(self, text: str) -> str:
        inp = "summarize: " + text.replace("\n"," ")
        tok = self.summ_tokenizer(
            inp, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        ids = self.summ_model.generate(
            **tok,
            max_length=256,
            min_length=64,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        return self.summ_tokenizer.decode(ids[0], skip_special_tokens=True).strip()



# Example usage:
if __name__ == '__main__':
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Set the dataset root directory.
    root_path = "../data/oxfordpets"
    
    # Define image transforms.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset instances for each split.
    train_dataset = OxfordPetsDataset(root_path, split='train', transform=transform, num_shots=1)
    #print("num classes", train_dataset.num_classes)
    #print("classnames", train_dataset.classnames)
    #print("lab2cname", train_dataset.lab2cname)
    val_dataset = OxfordPetsDataset(root_path, split='val', transform=transform)
    test_dataset = OxfordPetsDataset(root_path, split='test', transform=transform)
    print("classname", val_dataset._num_classes)
    print("classname", test_dataset._num_classes)
    print("Train set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))
    print("Test set size:", len(test_dataset))
    
    # Create DataLoaders.
    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Example: iterate over one batch from each DataLoader.
    for images, labels, classnames, summary in val_loader:
        print("Train Batch - images shape:", images.shape)
        print("Train Batch - labels:", labels)
        print("Train Batch - classnames:", classnames)
        print("Summary", summary)
        break