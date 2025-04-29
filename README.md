# SigLIP-Adapter: Parameter-Efficient Tuning for Zero- and Few-Shot Vision-Language Tasks

Vision–Language Models (VLMs) such as **CLIP** (Contrastive Language-Image
Pre-training) have revolutionised transfer learning in computer vision:
with the *same frozen backbone* they already achieve impressive
**zero-shot** and **few-shot** accuracy on dozens of datasets.  
Yet re-training or full fine-tuning remains costly, and naïvely ignoring
the cross-modal information hidden in image–text pairs leaves accuracy
on the table.

We introduce **SigLIP-Adapter**, a lightweight add-on that upgrades
Google’s **SigLIP** (Sigmoid Loss CLIP) for both zero-shot and few-shot
inference **without** touching the frozen backbone.

1. **Caption generation**  
   We optionally create a rich description for each support image via
   SmolVLM‐256M and compress it to ≤ 64 tokens with a T5 summariser.
2. **Key / Value cache**  
   *Keys* = frozen embeddings of those images and captions.  
   *Values* = one-hot class labels.
3. **Inference**  
   *Test image* ➜ SigLIP image encoder ➜ similarity to all keys  
   *Few-shot knowledge* is aggregated and blended with *zero-shot
   knowledge* (plain text-prompt logits) to yield final predictions.
4. **(Optional) PEFT**  
   Two adapter blocks (projector + latent mixer) are trained for a handful
   of epochs, then frozen. They reweight the similarities and improve
   accuracy—especially when only 1-16 shots per class are available.


---

## 1  Set-up

```bash
# ❶ create / activate your favourite virtual-env first
python -m venv .venv
source .venv/bin/activate        
# ❷ install all Python dependencies
pip install -r requirements.txt
```

## 2 Generate Captions

script | vision-language model(s) | note
generate_caption_llama.py | LLaMA encoder + BART-Large summariser | baseline
generate_caption_smolVLM.py | SmolVLM-256M-Instruct + T5-Small | ★ fast & memory-light

```
# example
python generate_caption_llama.py

# example
python generate_caption_smolVLM.py

```
