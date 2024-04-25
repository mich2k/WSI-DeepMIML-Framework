import os
import torch
import urllib.request
import gzip, shutil
from utils import get_device

def extract_data(path, output_path):
    with gzip.open(path, 'rb') as f_in:
      with open(output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def load_pretrained_linear_weights(linear_classifier, model_name, patch_size):
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_linearweights.pth"
    if url is not None:
        print("We load the reference pretrained linear weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url, map_location=get_device(), progress=True)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        linear_classifier.load_state_dict(state_dict, strict=True)
    else:
        print("We use random linear weights.")
        
def load_custom_weights(model, checkpoint_path, checkpoint_key, dataset_name, scale_level):
    
    if dataset_name == "camelyon16" and scale_level == 5: 
        url = "https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x5/checkpoint.pth.gz"
    if dataset_name == "camelyon16" and scale_level == 10:    
        url= "https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x10/checkpoint.pth.gz"
    if dataset_name == "camelyon16" and scale_level == 20:    
        url = "https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x20/checkpoint.pth.gz"
    
    if dataset_name == "lung" and scale_level == 5:
        url = "https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x5/checkpoint.pth.gz"
    if dataset_name == "lung" and scale_level == 10:    
        url = "https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x10/checkpoint.pth.gz"
    if dataset_name == "lung" and scale_level == 20:    
        url = "https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x20/checkpoint.pth.gz"
    
    weights = f"dino_weights_{dataset_name}_{scale_level}"
    
    output_path = checkpoint_path + weights + ".pth"
    print(f"Load {weights}")
    
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(url, filename = weights + ".gz")
        extract_data(weights + ".gz", output_path)
        os.remove(weights + '.gz')
    
    state_dict = torch.load(checkpoint_path + weights + ".pth")[checkpoint_key]
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("head.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    