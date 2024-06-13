# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %pip install git+https://github.com/MarcusLoppe/meshgpt-pytorch.git
# %pip install matplotlib

from pathlib import Path 
import gc    
import torch
import os
import torch  
from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer,MeshDataset
)
from meshgpt_pytorch.data import ( 
    derive_face_edges_from_faces
)   


# %cd /root/text-to-mesh

autoencoder = MeshAutoencoder( 
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        dim_codebook = 192,  
        dim_area_embed = 16,
        dim_coor_embed = 16, 
        dim_normal_embed = 16,
        dim_angle_embed = 8,
    
    attn_decoder_depth  = 4,
    attn_encoder_depth = 2
    ).to("cuda")     

# +
dataset = MeshDataset.load("./mesh-transformer-datasets/objverse_250f_490.7M_all_17561_labels_568425_5_min_x5_aug.npz")  
dataset2 = MeshDataset.load("./mesh-transformer-datasets/objverse_250f_98.1M_all_17561_labels_113685_5_min_x1_aug.npz")
dataset.data.extend(dataset2.data)  
dataset2 = MeshDataset.load("./mesh-transformer-datasets/shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")  
dataset.data.extend(dataset2.data)  
dataset2 = MeshDataset.load("./mesh-transformer-datasets/shapenet_250f_21.9M_84_labels_21560_10_min_x10_aug.npz")  
dataset.data.extend(dataset2.data) 
dataset.sort_dataset_keys()
print("length", len(dataset.data))

def format_value(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    else:
        return f"{value}"

tokens = 0
for item in dataset.data:
    tokens += len(item['faces']) * 6 
total_tokens = format_value(tokens)
print("Tokens:", total_tokens)
# -

pkg = torch.load("./2k_autoencoder_490M__recon_0.3665_commit_-0.7556.pt") 
autoencoder.load_state_dict(pkg['model'])
for param in autoencoder.parameters():
    param.requires_grad = True

# +
batch_size=32 # The batch size should be max 64.
grad_accum_every =2
# So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
learning_rate = 1e-3 # Start with 1e-3 then at stagnation around 0.35, you can lower it to 1e-4.

autoencoder.commit_loss_weight = 0.2 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.a
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                             batch_size=batch_size,
                                             grad_accum_every = grad_accum_every,
                                             learning_rate = learning_rate,
                                             checkpoint_every_epoch=1)
 
loss = autoencoder_trainer.train(480, diplay_graph= True)   
# -

pkg = dict( model = autoencoder.state_dict(), ) 
import datetime
# now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_").replace("+", "_") 
torch.save(pkg, f"./16k_autoencoder_229M_0.338.pt")
