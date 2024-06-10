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
dataset = MeshDataset.load("./objverse_250f_45.9M_3086_labels_53730_10_min_x1_aug.npz")  
dataset2 = MeshDataset.load("./objverse_250f_229.7M_3086_labels_268650_10_min_x5_aug.npz")
dataset.data.extend(dataset2.data)  
dataset2 = MeshDataset.load("./shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")  
dataset.data.extend(dataset2.data)  
dataset2 = MeshDataset.load("./shapenet_250f_21.9M_84_labels_21560_10_min_x10_aug.npz")  
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

# +
# pkg = torch.load("checkpoints/mesh-autoencoder.ckpt.epoch_5_avg_loss_0.17483_recon_0.3439_commit_-0.6763.pt") 
# autoencoder.load_state_dict(pkg['model'])
# for param in autoencoder.parameters():
#     param.requires_grad = True

# +
batch_size=16 # The batch size should be max 64.
grad_accum_every =4
# So set the maximal batch size (max 64) that your VRAM can handle and then use grad_accum_every to create a effective batch size of 64, e.g  16 * 4 = 64
learning_rate = 1e-3 # Start with 1e-3 then at stagnation around 0.35, you can lower it to 1e-4.

autoencoder.commit_loss_weight = 0.2 # Set dependant on the dataset size, on smaller datasets, 0.1 is fine, otherwise try from 0.25 to 0.4.
autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                             batch_size=batch_size,
                                             grad_accum_every = grad_accum_every,
                                             learning_rate = learning_rate,
                                             checkpoint_every_epoch=5)
 
loss = autoencoder_trainer.train(480, diplay_graph= True)   

# +
import datetime
import torch
now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
now_safe = now_utc.replace(":", "_")
pkg = dict( model = autoencoder.state_dict(), )
filename = f"./MeshGPT-autoencoder_{now_safe}.pt"
torch.save(pkg, filename)
import torch
import random
from tqdm import tqdm 
from meshgpt_pytorch import mesh_render 

min_mse, max_mse = float('inf'), float('-inf')
min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
random_samples, random_samples_pred, all_random_samples = [], [], []
total_mse, sample_size = 0.0, 200

random.shuffle(dataset.data)
autoencoder = autoencoder.to("cuda")
for item in tqdm(dataset.data[:sample_size]):
    item['faces'] = item['faces'].to("cuda")
    item['vertices'] = item['vertices'].to("cuda")
    item['face_edges'] = item['face_edges'].to("cuda")
    codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges']) 
    
    codes = codes.flatten().unsqueeze(0)
    codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers] 
 
    coords, mask = autoencoder.decode_from_codes_to_faces(codes)
    orgs = item['vertices'][item['faces']].unsqueeze(0)

    mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu())**2)
    total_mse += mse 

    if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
    if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
 
    if len(random_samples) <= 30:
        random_samples.append((coords, mask)) 
    else:
        all_random_samples.extend([ random_samples])
        random_samples, random_samples_pred = [], []

print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')    
mesh_render.save_rendering(f'./mse_rows.obj', all_random_samples)

 

# +
transformer.conditioner.text_models[0].model.to("cuda")
transformer = transformer.to("cuda")

dataset.embed_texts(transformer,1)

# +
 
torch.cuda.empty_cache()
gc.collect()  
 
batch_size = 2
grad_accum_every =4     
trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=grad_accum_every,num_train_steps=100, dataset = dataset, 
                                  accelerator_kwargs = {"mixed_precision" : "fp16"}, optimizer_kwargs = { "eps": 1e-7} , 
                                 learning_rate = 1e-3, batch_size=batch_size ,checkpoint_every_epoch = 25) 
loss = trainer.train(35, stop_at_loss = 0.00005)   
 

# +
import torch
import random
from tqdm import tqdm 
from meshgpt_pytorch import mesh_render 

min_mse, max_mse = float('inf'), float('-inf')
min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
random_samples, random_samples_pred, all_random_samples = [], [], []
total_mse, sample_size = 0.0, 200

random.shuffle(dataset.data)
autoencoder = autoencoder.to("cuda")
for item in tqdm(dataset.data[:sample_size]):
    item['faces'] = item['faces'].to("cuda")
    item['vertices'] = item['vertices'].to("cuda")
    item['face_edges'] = item['face_edges'].to("cuda")
    codes = autoencoder.tokenize(vertices=item['vertices'], faces=item['faces'], face_edges=item['face_edges']) 

    codes = codes.flatten().unsqueeze(0)
    codes = codes[:, :codes.shape[-1] // autoencoder.num_quantizers * autoencoder.num_quantizers] 
 
    coords, mask = autoencoder.decode_from_codes_to_faces(codes)
    orgs = item['vertices'][item['faces']].unsqueeze(0)

    mse = torch.mean((orgs.view(-1, 3).cpu() - coords.view(-1, 3).cpu())**2)
    total_mse += mse 

    if mse < min_mse: min_mse, min_coords, min_orgs = mse, coords, orgs
    if mse > max_mse: max_mse, max_coords, max_orgs = mse, coords, orgs
 
    if len(random_samples) <= 30:
        random_samples.append((coords, mask)) 
    else:
        all_random_samples.extend([ random_samples])
        random_samples, random_samples_pred = [], []

print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
mesh_render.save_rendering(f'.\mse_rows.obj', all_random_samples)
# -

now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
now_safe = now_utc.replace(":", "_")
filename = f"./MeshGPT-autoencoder_{now_safe}.pt"
pkg = dict( model = autoencoder.state_dict(), ) 
torch.save(pkg, filename)
