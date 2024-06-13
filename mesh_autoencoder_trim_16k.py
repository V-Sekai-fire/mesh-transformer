# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="jQrv0eJn58b0"
# %pip install git+https://github.com/MarcusLoppe/meshgpt-pytorch.git
# %pip install matplotlib
from pathlib import Path
import gc
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

autoencoder = MeshAutoencoder(
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        dim_codebook = 192,
        dim_area_embed = 16,
        dim_coor_embed = 16,
        dim_normal_embed = 16,
        dim_angle_embed = 8,
        attn_decoder_depth  = 4,
        attn_encoder_depth = 2)

pkg = torch.load("./mesh-autoencoder.ckpt.epoch_0_avg_loss_-0.09213_recon_0.3429_commit_-0.8701.pt")
autoencoder.load_state_dict(pkg['model'], strict = False)

pkg = dict( model = autoencoder.state_dict(), )
import datetime
torch.save(pkg, f"./16k_mesh-autoencoder.pt")
