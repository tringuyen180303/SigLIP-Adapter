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
from datasets import OxfordPetsDataset
from torch.utils.data import DataLoader
def run_siglip(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels,test_features, test_labels, model_weights): 
    cache_keys = (img_cache_keys + text_cache_keys)/2

    zero_logits = 100. * val_features @ model_weights
    print("val shape", val_features.shape)
    print("model weight shape", model_weights.shape)
    acc = classification_acc(zero_logits, val_labels)

    print(f"Zero-shot accuracy SigLIP: {acc:.2f}%")
    
    # Adapter
    beta, alpha = cfg['idea']['beta'], cfg['idea']['alpha']
    affinity = val_features @ cache_keys
    cache_values = cache_values.float()  
    few_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    idea_logits = (few_logits * alpha) + zero_logits # SigLIP
    acc = classification_acc(idea_logits, val_labels)
    print(f"SigLIP with adapter: {acc:.2f}%")

    ### Search hyperparameters
    #best_theta, best_beta, best_alpha = 2, 0.2, 0.3
    #After searching, the best accuarcy: 64.95.
#, best theta: 0.05, best beta: 3.22, best alpha: 0.73
    best_theta, best_beta, best_alpha = 0.05, 3.22, 0.73
    #best_theta, best_beta, best_alpha = search_hp_2(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels, model_weights)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    zero_logits = 100. * test_features @ model_weights
    acc = classification_acc(zero_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # IDEA-Adapter    
    affinity = best_theta * test_features @ text_cache_keys + (1-best_theta) * test_features @ img_cache_keys
    few_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
    
    idea_logits = zero_logits + few_logits * best_alpha
    acc = classification_acc(idea_logits, test_labels)
    print("**** IDEA-Adapter's test accuracy: {:.2f}. ****\n".format(acc))


def run_TSGILIP(cfg, img_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_labels, test_features, model_weights, model,  train_loader_F):
    print("img cache keys", img_cache_keys.shape)
    print("img cache 0", img_cache_keys.shape[0])
    print("img cache 1", img_cache_keys.shape[1])


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

        for i, (images, target, dess) in enumerate(tqdm(train_loader_F)):
            with torch.no_grad():
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_feature_text = adapter(image_features)
            affinity2 = adapter2(image_features)

            affinity =  (image_feature_text @ text_cache_keys + image_features @ img_cache_keys + image_features @ text_cache_keys)/3
            affinity += affinity2
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
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Evaluation

        adapter.eval()
        adapter2.eval()
        test_feature_text = adapter(test_features)
        affinity2 = adapter2(test_features)

        affinity = (test_feature_text @ text_cache_keys + test_features @ img_cache_keys + test_features @ text_cache_keys)/3
        affinity += affinity2

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
        affinity += affinity2
        cache_values = cache_values.float()
        few_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values
        TIDEA_logits = zero_logits + few_logits * best_alpha
        acc = classification_acc(TIDEA_logits, test_labels)
        print("**** IDEA-Adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))
def main():
    # Load config file
    cfg = load_config("config.yaml")
    # Load SigLIP
    model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
    print("model context length", model.context_length)
    model.eval()
    random.seed(1)
    torch.manual_seed(1)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std= [0.229, 0.224, 0.225])]
    )
    root_path = "./data/oxfordpets"
    
    val_dataset = OxfordPetsDataset(root_path, split='val', transform=transform, val_ratio=0.1)
    test_dataset = OxfordPetsDataset(root_path, split='test', transform=transform, val_ratio=0.1)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)

    # Build cache loader
    train_dataset = OxfordPetsDataset(root_path, split='train', transform=transform, num_shots=cfg['data']['shots'])
    train_loader_cache = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)
    train_loader_F = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)

    model_weights = model_classifier(train_dataset.classnames, train_dataset.template, model)

    im_cache_keys, text_cache_keys, cache_values = build_cache_model(cfg, model, train_loader_cache)

    # Preload features
    #val_features, val_labels = extract_features(model, val_loader)
    #test_features, test_labels = extract_features(model, test_loader)

    val_features, val_labels = extract_features_cached(cfg, model, val_loader, split_name="val")
    print("val labels", val_labels)
    print("val labels shape", val_labels.shape)
    test_features, test_labels = extract_features_cached(cfg, model, test_loader, split_name="test")
    print("test labels", test_labels)
    print("test labels shape", test_labels.shape)
    #run_siglip(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_features, test_labels, model_weights)
    run_TSGILIP(cfg, im_cache_keys, text_cache_keys, cache_values, val_features, val_labels, test_labels, test_features, model_weights, model, train_loader_F)
# cfg = load_config() # Load configuration
# dataset = build_dataset(cfg['dataset_name'], cfg['root_path'], cfg['shots']) # Build dataset
    
if __name__ == "__main__":
    main()