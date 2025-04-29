import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, T5ForConditionalGeneration, T5Tokenizer
from torchvision import transforms
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch_dtype = torch.bfloat16
    attn_impl  = "flash_attention_2"
else:
    DEVICE = torch.device("cpu")
    torch_dtype = torch.float32
    attn_impl  = "eager"
print(f"Using device: {DEVICE}")

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model     = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    _attn_implementation=attn_impl,
).to(DEVICE)
model.eval()

# 2) Oxford Pets dataset (reuse your class)
from datasets import OxfordPetsDataset  # adjust import as needed

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
test_ds = OxfordPetsDataset(root_path="data/oxfordpets",
                            split="test",
                            transform=transform)
loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

# 3) Prompt template
TEMPLATE = (
    "This is a photo of a {cls_name}. "
    "Please describe the object’s shape, colour, texture, pose, background, "
    "and camera angle. Only mention what you can see."
)

# 4) Helper to un-normalize and PIL‑ify
inv_norm = transforms.Normalize(
    mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
    std=[1/s for s in [0.229,0.224,0.225]]
)
to_pil = transforms.ToPILImage()

# 5) Loop, generate, and save
out_folder = "pet_descriptions"
os.makedirs(out_folder, exist_ok=True)

SUM_MODEL = "google-t5/t5-small"
summ_tokenizer = T5Tokenizer.from_pretrained(SUM_MODEL)
summ_model     = T5ForConditionalGeneration.from_pretrained(SUM_MODEL).to(DEVICE)
summ_model.eval()

def summarize_text(text: str,
                   max_input_length: int = 512,
                   max_summary_length: int = 256,
                   min_summary_length: int = 64) -> str:
    # prep T5 prompt
    inp = "summarize: " + text.strip().replace("\n", " ")
    tokenized = summ_tokenizer(
        inp,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    ).to(DEVICE)
    summary_ids = summ_model.generate(
        **tokenized,
        max_length=max_summary_length,
        min_length=min_summary_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    return summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

with torch.no_grad():
    for idx, (img_tensor, label, cls_name) in enumerate(loader):
        cls_name = cls_name[0].replace("_", " ")  # e.g. "british shorthair"
        
        # build the user message
        prompt_text = TEMPLATE.format(cls_name=cls_name)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",  "text": prompt_text}
            ]
        }]

        # un-normalize → PIL
        pil_img = to_pil(inv_norm(img_tensor[0]).clamp(0,1))

        # prepare chat inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=prompt,
            images=[pil_img],
            return_tensors="pt"
        ).to(DEVICE)
        print("Prompt", prompt)
        # generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4
        )
        desc = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        
        summary = summarize_text(desc)

        # 7) Save everything and print side‑by‑side
        out_path = os.path.join(out_folder, f"{idx:04d}_{cls_name.replace(' ','_')}.json")
        with open(out_path, "w") as f:
            json.dump({
                "index": idx,
                "classname": cls_name,
                "prompt":    prompt_text,
                "description": desc,
                "summary":     summary
            }, f, indent=2)

        print(f"[{idx}] {cls_name}")
        print("  Full  ➜", desc)
        print("  Brief ➜", summary, "\n")