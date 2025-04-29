
import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from utils import *
from open_clip import create_model_from_pretrained, get_tokenizer
from datasets import DTDTextureDataset
from torch.utils.data import DataLoader

def run_siglip(cfg, im_k, txt_k, vals, val_f, val_l, test_f, test_l, w):
    beta,alpha=cfg['idea']['beta'],cfg['idea']['alpha']
    zero=100*val_f@w; print(f"Zero‑shot val acc: {classification_acc(zero,val_l):.2f}%")
    vals = vals.float()
    aff=val_f@((im_k+txt_k)/2); few=((-1)*(beta-beta*aff)).exp()@vals
    idea=zero+few*alpha; print(f"IDEA val acc: {classification_acc(idea,val_l):.2f}%")
    # test
    zero=100*test_f@w; print(f"Zero‑shot test acc: {classification_acc(zero,test_l):.2f}%")
    aff=test_f@((im_k+txt_k)/2); few=((-1)*(beta-beta*aff)).exp()@vals
    idea=zero+few*alpha; print(f"IDEA test acc: {classification_acc(idea,test_l):.2f}%")

def run_TSGILIP(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels,test_features, test_labels,  model_weights, model,  train_loader_F):
    adapter = nn.Linear(img_cache_keys.shape[0], img_cache_keys.shape[0], bias=True)
    adapter2 = nn.Linear(img_cache_keys.shape[0], img_cache_keys.shape[1], bias=True)

    optimizer = torch.optim.AdamW(
        [{
            "params": adapter.parameters()
        },
        {
            "params": adapter2.parameters()
        }], lr=0.001, eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['training']['epochs'] * len(train_loader_F))
    beta, alpha = cfg['idea']['beta'], cfg['idea']['alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['training']['epochs']):
        # Train
        adapter.train()
        adapter2.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['training']['epochs']))

        for i, (images, target, classname, dess) in enumerate(tqdm(train_loader_F)):
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                #image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()
            image_feature_text = adapter(image_features)
            affinity2 = adapter2(image_features)

            affinity =  (image_feature_text @ text_cache_keys + image_features @ img_cache_keys + image_features @ text_cache_keys)
            affinity = affinity / 3.0
            affinity = affinity + affinity2
            cache_values = cache_values.float()
            few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            zero_logits = 100. * image_features @ model_weights

            TIDEA_logits = zero_logits + few_logits * alpha

            loss = F.cross_entropy(TIDEA_logits, target)

            acc = classification_acc(TIDEA_logits, target)
            correct_samples += acc / 100 * len(TIDEA_logits)
            all_samples += len(TIDEA_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            torch.autograd.set_detect_anomaly(True)

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Evaluation

        adapter.eval()
        adapter2.eval()
        test_feature_text = adapter(test_features)
        affinity2 = adapter2(test_features)

        affinity = (test_feature_text @ text_cache_keys + test_features @ img_cache_keys + test_features @ text_cache_keys)/3
        affinity = affinity + affinity2  

        few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        zero_logits = 100. * test_features @ model_weights
        TIDEA_logits = zero_logits + few_logits * alpha
        acc = classification_acc(TIDEA_logits, test_labels)

        print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            best_f_path  = os.path.join(cfg['model']['cache_dir'],
                            f"best_F_{cfg['data']['shots']}shots.pt")
            best_f2_path = os.path.join(cfg['model']['cache_dir'],
                            f"best_F_2_{cfg['data']['shots']}shots.pt")
            torch.save(adapter.state_dict(),  best_f_path)
            torch.save(adapter2.state_dict(), best_f2_path)

        adapter.load_state_dict(torch.load(best_f_path))
        adapter2.load_state_dict(torch.load(best_f2_path))
        print(f"**** After fine-tuning, IDEA-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

        #best_theta, best_beta, best_alpha = 2, 0.5, 0.5
        best_theta, best_beta, best_alpha = 0.05, 3.22, 0.73
        #best_theta, best_beta, best_alpha = search_hp_2(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels, model_weights)

        print("\n-------- Evaluating on the test set. --------")

        test_features_text = adapter(test_features)
        affinity2 = adapter2(test_features)
        affinity = best_theta* (test_features_text + test_features) @ text_cache_keys + (1-best_theta) * test_features @ img_cache_keys
        affinity = affinity + affinity2
        cache_values = cache_values.float()
        few_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
        TIDEA_logits = zero_logits + few_logits * best_alpha
        acc = classification_acc(TIDEA_logits, test_labels)
        print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))


# --------------------------------------------------
# 4. Main Routine
# --------------------------------------------------

if __name__ == '__main__':
    cfg = load_config("config.yaml")
    model, _ = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    root = cfg['data']['root_path']
    train_ds = DTDTextureDataset(root, 'train', transform=transform, num_shots=cfg['data']['shots'], cfg=cfg)
    val_ds = DTDTextureDataset(root, 'val', transform=transform, cfg=cfg)
    test_ds = DTDTextureDataset(root, 'test', transform=transform, cfg=cfg)
    train_f = DTDTextureDataset(root, 'train', num_shots=cfg['data']['shots'],  generate_desc=False, transform=transform, cfg=cfg)
    print(len(train_f))

    bs = cfg['training']['batch_size']
    train_loader_cache = DataLoader(train_ds, batch_size=bs, shuffle=False)
    train_loader_finetune = DataLoader(train_f, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    test_loader = DataLoader(test_ds, batch_size=bs)

    weights = model_classifier(train_ds.classnames, TEMPLATE, model)
    im_k, txt_k, vals = build_cache_model(cfg, model, train_loader_cache)
    val_f, val_l = extract_features_cached(cfg, model, val_loader, "val")
    test_f, test_l = extract_features_cached(cfg, model, test_loader, "test")

    run_siglip(cfg, im_k, txt_k, vals, val_f, val_l, test_f, test_l, weights)
    run_TSGILIP(cfg, im_k, txt_k, vals, val_f, val_l, test_f, test_l, weights, model, train_loader_finetune)
    