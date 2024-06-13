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

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18947, "status": "ok", "timestamp": 1718259681310, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="GPb71T_g3EwE" outputId="f4e1532f-b0d8-49c8-cf23-0a9a9a9bf467"
# %pip install git+https://github.com/MarcusLoppe/meshgpt-pytorch.git
# %pip install matplotlib
# %pip install accelerate

# + executionInfo={"elapsed": 4190, "status": "ok", "timestamp": 1718259687710, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="i4fP1VEP3EwF"
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

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 893201, "status": "ok", "timestamp": 1718260580906, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="oja1-j9v3EwF" outputId="32f7c29b-ff06-4c17-dd15-ce25f018524b"
from accelerate import notebook_launcher
autoencoder = MeshAutoencoder(
    decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
    dim_codebook = 192,
    dim_area_embed = 16,
    dim_coor_embed = 16,
    dim_normal_embed = 16,
    dim_angle_embed = 8,
    attn_decoder_depth  = 4,
    attn_encoder_depth = 2)

pkg = torch.load("../datasets/16k_autoencoder_229M_0.338.pt")
autoencoder.load_state_dict(pkg['model'], strict = False)

dataset = MeshDataset.load("../datasets/objverse_250f_490.7M_all_17561_labels_568425_5_min_x5_aug.npz")

dataset2 = MeshDataset.load("../datasets/objverse_250f_98.1M_all_17561_labels_113685_5_min_x1_aug.npz")
dataset.data.extend(dataset2.data)
dataset2 = MeshDataset.load("../datasets/shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")
dataset.data.extend(dataset2.data)
dataset2 = MeshDataset.load("../datasets/shapenet_250f_21.9M_84_labels_21560_10_min_x10_aug.npz")
dataset.data.extend(dataset2.data)

dataset.sort_dataset_keys()

# +
# autoencoder.commit_loss_weight = 0.5
# autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
#                                                 batch_size=32,
#                                                 grad_accum_every =2,
#                                                 learning_rate = 1e-4,
#                                                 checkpoint_every_epoch=1)
# _loss1 = autoencoder_trainer.train(14445,  diplay_graph= False)

# +
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer
)

transformer = MeshTransformer(
    autoencoder,
    dim = 768,
    coarse_pre_gateloop_depth =2,
    fine_pre_gateloop_depth= 2,
    attn_depth = 12,
    attn_heads = 12,
    cross_attn_num_mem_kv = 4,
    fine_cross_attend_text = True,
    text_cond_with_film = False,
    num_sos_tokens = 1,
    dropout  = 0.0,
    max_seq_len = 1500,
    fine_attn_depth = 2,
    condition_on_text = True,
    gateloop_use_heinsen = False,
    text_condition_model_types = "bge",
    text_condition_cond_drop_prob = 0.0,
).cuda()


def generate_codes(self, autoencoder : MeshAutoencoder, batch_size = 25):
    total_batches = (len(self.data) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(self.data), batch_size), total=total_batches):
        batch_data = self.data[i:i+batch_size]

        padded_batch_vertices = pad_sequence([item['vertices'] for item in batch_data], batch_first=True, padding_value=autoencoder.pad_id).cuda()
        padded_batch_faces = pad_sequence([item['faces'] for item in batch_data], batch_first=True, padding_value=autoencoder.pad_id).cuda()
        padded_batch_face_edges = pad_sequence([item['face_edges'] for item in batch_data], batch_first=True, padding_value=autoencoder.pad_id).cuda()

        batch_codes = autoencoder.tokenize(
            vertices=padded_batch_vertices,
            faces=padded_batch_faces,
            face_edges=padded_batch_face_edges
        )


        mask = (batch_codes != autoencoder.pad_id).all(dim=-1)
        for item_idx, (item_codes, item_mask) in enumerate(zip(batch_codes, mask)):
            item_codes_masked = item_codes[item_mask]
            item = batch_data[item_idx]
            item['codes'] = item_codes_masked.to("cpu")

    self.sort_dataset_keys()
    print(f"[MeshDataset] Generated codes for {len(self.data)} entries")

generate_codes(dataset, autoencoder, 350)
dataset.embed_texts(transformer, 1)

# + colab={"base_uri": "https://localhost:8080/"} id="jKNepVuVM0Pv" outputId="acd9940a-7d4c-424a-b352-36436243816c"
from accelerate import Accelerator

accelerator = Accelerator()

batch_size = 64
grad_accum_every = 64 // batch_size
rate = 1e-3

transformer = transformer.to(accelerator.device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=rate)

trainer = MeshTransformerTrainer(model=transformer, warmup_steps=100, grad_accum_every=grad_accum_every,
    num_train_steps=100, dataset=dataset, batch_size=batch_size, learning_rate=rate, checkpoint_every_epoch=1)

model, optimizer = accelerator.prepare(trainer.model, trainer.optimizer)

loss = trainer.train(503)  

# pkg = dict( model = transformer.state_dict(), )
# torch.save(pkg, str("./MeshGPT-transformer_trained.pt"))
