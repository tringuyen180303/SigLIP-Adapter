# SigLIP-Adapter: Parameter-Efficient Tuning for Zero- and Few-Shot Vision-Language Tasks

Vision–Language Models (VLMs) such as **CLIP** (Contrastive Language-Image
Pre-training) have revolutionised transfer learning in computer vision:
with the *same frozen backbone* they already achieve impressive
**zero-shot** and **few-shot** accuracy on dozens of datasets.  
Yet re-training or full fine-tuning remains costly, and naïvely ignoring
the cross-modal information hidden in image–text pairs leaves accuracy
on the table. This is fixed by the Enhanced CLIP-Adapter ((T)IDEA) framework, which
introduces a mutli-modal adapter instead of the vision focused adapter. This allows
us to train the adapter on the image–description pairs. [(T)IDEA Repository](https://github.com/FourierAI/IDEA)

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


## Text Generation Pipeline
![Good Text Generation Examples](/text_generation_examples/Text_Pipeline.png)


## Good Text Generation Examples

![Good Text Generation Examples](/text_generation_examples/Good_Text_Generation_Examples.png)

## Bad Text Generation Examples

![Bad Text Generation Examples](/text_generation_examples/Bad_Text_Generation_Examples.png)



## Slides and Papers

- [SigLIP-Adapter slides](https://github.com/tringuyen180303/SigLIP-Adapter/blob/main/docs/SigLIP_Adapter.pptx)
- [Project report](https://github.com/tringuyen180303/SigLIP-Adapter/blob/main/docs/SigLIP-Adapter.pdf)

---

## 1  Set-up

```bash
# ❶ create / activate your favourite virtual-env first
python -m venv .venv
source .venv/bin/activate        
# ❷ install all Python dependencies
pip install -r requirements.txt
```

Download the datasets and place them in the `data/` folder.

[Oxford Pets](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)

[Food101](https://www.kaggle.com/datasets/dansbecker/food-101)

[Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)

[EuroSat](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)

[Describable Textures Dataset](https://www.kaggle.com/datasets/jmexpert/describable-textures-dataset-dtd)

## 2 Generate Captions

script | vision-language model(s) | note

generate_caption_llama.py | LLaMA encoder + BART-Large summariser | baseline

generate_caption_smolVLM.py | SmolVLM-256M-Instruct + T5-Small | ★ fast & memory-light

```bash
# example
python generate_captions/generate_caption_llama.py

# example
python generate_captions/generate_caption_smolVLM.py

```

## 3 Run the five benchmark experiments

```bash
# Oxford-IIIT Pets   (37 classes)
python model_oxfordpets.py

# Food-101           (101 classes)
python model_food101.py

# EuroSAT            (10 land-use classes)
python model_eurosat.py

# DTD (Describable Textures, 47 classes)
python model_dtd.py

# Caltech-101        (101 object categories)
python model_caltech101.py
```
