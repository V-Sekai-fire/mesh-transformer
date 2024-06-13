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
# %pip install accelerate

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

# %cd /root/mesh-transformer

# +
from accelerate import notebook_launcher
def training_function(num_processes=2):
    autoencoder = MeshAutoencoder(
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,   
        dim_codebook = 192,  
        dim_area_embed = 16,
        dim_coor_embed = 16, 
        dim_normal_embed = 16,
        dim_angle_embed = 8, 
        attn_decoder_depth  = 4,
        attn_encoder_depth = 2) 
    
    pkg = torch.load("./mesh-autoencoder.ckpt.epoch_0_avg_loss_-0.06490_recon_0.3518_commit_-0.8334.pt")  
    autoencoder.load_state_dict(pkg['model'], strict = False) 
  
    dataset = MeshDataset.load("./mesh-transformer-datasets/objverse_250f_490.7M_all_17561_labels_568425_5_min_x5_aug.npz")  
    dataset2 = MeshDataset.load("./mesh-transformer-datasets/objverse_250f_98.1M_all_17561_labels_113685_5_min_x1_aug.npz")
    dataset.data.extend(dataset2.data)  
    dataset2 = MeshDataset.load("./mesh-transformer-datasets/shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")  
    dataset.data.extend(dataset2.data)  
    dataset2 = MeshDataset.load("./mesh-transformer-datasets/shapenet_250f_21.9M_84_labels_21560_10_min_x10_aug.npz")  
    dataset.data.extend(dataset2.data) 
    dataset.sort_dataset_keys() 
         
    autoencoder.commit_loss_weight = 0.5
    autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                                 batch_size=16,
                                                 grad_accum_every =4,
                                                 learning_rate = 1e-4,
                                                 checkpoint_every_epoch=1)  
    _loss1 = autoencoder_trainer.train(14445,  diplay_graph= False)        
 
args = ()
notebook_launcher(training_function, args, num_processes=1)
