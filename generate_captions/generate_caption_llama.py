from transformers import AutoModelForCausalLM, AutoTokenizer, \
                         AutoModelForSeq2SeqLM, pipeline
import torch
import json
# --- LLaMA captioner -------------------------------------------------
llama_id      = "huggyllama/llama-30b"
llama_tok     = AutoTokenizer.from_pretrained(llama_id)
llama_model   = AutoModelForCausalLM.from_pretrained(llama_id,
                                                     torch_dtype=torch.float32,
                                                     )

def caption_llama(image_tensor, cls_name):
    prompt = f"This is a photo of a {cls_name}. " \
             f"Please describe the object with shape, colour, texture, pose, " \
             f"background and camera angle. Only mention what you can see."
    inputs  = llama_tok(prompt, return_tensors="pt").to(image_tensor.device)
    generated = llama_model.generate(**inputs, max_new_tokens=120)[0]
    return llama_tok.decode(generated, skip_special_tokens=True)
# ---------------------------------------------------------------------
# --- BART summariser (<77 tokens) ------------------------------------
summariser = pipeline("summarization",
                      model="facebook/bart-large-cnn",
                      device=0)

def compress(text):
    out = summariser(text, max_length=75, min_length=30)[0]["summary_text"]
    return out

if __name__ == "__main__":
    import os
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from datasets import OxfordPetsDataset  # adjust your import

    # 0) Make sure your dependencies are installed:
    #    pip install torch torchvision transformers accelerate

    # 1) Prepare your dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = OxfordPetsDataset(
        root_path="data/oxfordpets",
        split="test",
        transform=transform
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 2) Loop over the first few examples
    for idx, (img_tensor, label, cls_name) in enumerate(loader):
        cls_name = cls_name[0].replace("_"," ")
        device = img_tensor.device

        # 3) Run LLaMA captioner
        full_caption = caption_llama(img_tensor, cls_name)
        print(f"\n[{idx}] FULL CAPTION:\n  {full_caption}")

        # 4) Run BART summarizer
        summary = compress(full_caption)
        print(f"    SUMMARY:\n  {summary}")

        # 5) Optionally save to disk
        out = {
            "classname": cls_name,
            "full_caption": full_caption,
            "summary": summary
        }
        os.makedirs("llama_captions", exist_ok=True)
        with open(f"llama_captions/{idx:03d}_{cls_name.replace(' ','_')}.json", "w") as f:
            json.dump(out, f, indent=2)

        if idx >= 4:
            break
