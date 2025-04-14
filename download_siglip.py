import os
import torch
from open_clip import create_model_from_pretrained

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
        torch.save(model.state_dict(), model_file)
    
    model.eval()
    return model, preprocess

# Example usage:
if __name__ == "__main__":
    model, preprocess = load_local_model("SigLIP")
