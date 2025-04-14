from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
import clip
import random
from typing import List, Optional
from open_clip import create_model_from_pretrained, get_tokenizer
import os
# def classification_acc(logits, labels):
#     """
#     logits: torch.Tensor, shape [N, num_classes]
#     labels: torch.Tensor, shape [N], integer ground-truth labels

#     Returns: float accuracy in [0, 100]
#     """
#     # Predicted label is the argmax of the logits
#     preds = logits.argmax(dim=1)
    
#     # Compare predictions vs. ground truth
#     correct = (preds == labels).sum().item()
#     total = labels.size(0)
    
#     # Accuracy as a percentage
#     accuracy = 100.0 * correct / total
#     return accuracy
def classification_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    target = target.argmax(dim=1)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_cache_model(cfg, model, train_loader_cache):
    # Use the nested config keys for model settings.
    if cfg["model"]["load_cache"] == False:
        im_cache_keys = []
        text_cache_keys = []
        cache_values = []
        augment_epoch = cfg["model"].get("augment_epoch", 1)
        
        with torch.no_grad():
            # Data augmentation for the cache model.
            for augment_idx in range(augment_epoch):
                train_im_features = []
                train_text_features = []
                print(f'Augment Epoch: {augment_idx + 1} / {augment_epoch}')
                for i, (images, target, dess) in enumerate(tqdm(train_loader_cache)):
                    #texts = clip.tokenize(dess, truncate=True).cuda()
                    tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
                    texts = tokenizer(dess, context_length=model.context_length)
                    text_features = model.encode_text(texts)
                    train_text_features.append(text_features)

                    images = images
                    image_features = model.encode_image(images)
                    train_im_features.append(image_features)
                    if augment_idx == 0:
                        target = target
                        cache_values.append(target)
                im_cache_keys.append(torch.cat(train_im_features, dim=0).unsqueeze(0))
                text_cache_keys.append(torch.cat(train_text_features, dim=0).unsqueeze(0))
        
        im_cache_keys = torch.cat(im_cache_keys, dim=0).mean(dim=0)
        im_cache_keys /= im_cache_keys.norm(dim=-1, keepdim=True)
        im_cache_keys = im_cache_keys.permute(1, 0)

        text_cache_keys = torch.cat(text_cache_keys, dim=0).mean(dim=0)
        text_cache_keys /= text_cache_keys.norm(dim=-1, keepdim=True)
        text_cache_keys = text_cache_keys.permute(1, 0)

        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()
        
        torch.save(cache_values, os.path.join(cfg["model"]["cache_dir"], f'values_{cfg["data"]["shots"]}shots.pt'))
    else:
        im_cache_keys = torch.load(os.path.join(cfg["model"]["cache_dir"], f'keys_{cfg["data"]["shots"]}shots.pt'))
        cache_values = torch.load(os.path.join(cfg["model"]["cache_dir"], f'values_{cfg["data"]["shots"]}shots.pt'))
        text_cache_keys = None  # Modify as needed.
    
    return im_cache_keys, text_cache_keys, cache_values

def extract_features(model, data_loader):
    """
    Extracts image and text features from the model for the given data_loader.
    """
    img_features = []
    text_features = []
    
    with torch.no_grad():
        for images, labels, classnames in tqdm(data_loader):
            # Forward pass to get image features
            #images = images.to(model.device)
            img_feats = model.encode_image(images) # shape: [batch_size, feature_dim]
            img_features.append(img_feats.cpu())

            # Forward pass to get text features
            tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
            text = tokenizer(classnames, context_length=model.context_length)
            text_feats = model.encode_text(text)
            text_features.append(text_feats.cpu())

    img_features = torch.cat(img_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    print("img feat shape", img_features.shape)
    print("text feature shape", text_features.shape)

    return img_features, text_features

def model_classifier(classnames, template, model):
    with torch.no_grad():
        model_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
            texts = tokenizer(texts, context_length=model.context_length)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            model_weights.append(class_embedding)
        model_weights = torch.stack(model_weights, dim=1)
    return model_weights


def load_local_model(model_name, local_path="models", download_id='hf-hub:timm/ViT-B-16-SigLIP'):
    os.makedirs(local_path, exist_ok=True)
    model_file = os.path.join(local_path, f"{model_name}_state.pth")
    
    # Always create the model architecture first.
    model, preprocess = create_model_from_pretrained(download_id)
    
    if os.path.exists(model_file):
        print("Loading model parameters from disk")
        model.load_state_dict(torch.load(model_file))
    else:
        print("Downloading model parameters and saving to disk")
        # At this point, model already has the downloaded parameters.
        torch.save(model.state_dict(), model_file)
    
    model.eval()
    return model, preprocess


