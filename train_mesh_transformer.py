# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2024 MarcusLoppe & K. S. Ernest (iFire) Lee

from dagster import op, execute_job, job, reconstructable, DagsterInstance
import datetime
import torch
import random
from tqdm import tqdm 
from meshgpt_pytorch import mesh_render 

from dagster import execute_job, reconstructable, DagsterInstance
from train_autoencoder import train_autoencoder_job, autoencoder_asset, train_autoencoder, save_model, evaluate_model

from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer,
    MeshDataset
)

AUTOENCODER_EPOCHS = 1 # 480
TRANSFORMER_TOTAL_EPOCHS = 1 # 740
TRANSFORMER_EPOCHS_PER_SAVE = 25

@op
def load_datasets(context):
    dataset = MeshDataset.load("./objverse_250f_45.9M_3086_labels_53730_10_min_x1_aug.npz")  
    dataset2 = MeshDataset.load("./objverse_250f_229.7M_3086_labels_268650_10_min_x5_aug.npz")
    dataset.data.extend(dataset2.data)  
    dataset2 = MeshDataset.load("./shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")  
    dataset.data.extend(dataset2.data)  
    dataset2 = MeshDataset.load("./shapenet_250f_21.9M_84_labels_21560_10_min_x10_aug.npz")  
    dataset.data.extend(dataset2.data) 
    dataset.sort_dataset_keys()
    return dataset

@op
def create_transformer(context, autoencoder):
    transformer = MeshTransformer(
        autoencoder,
        dim = 768,
        coarse_pre_gateloop_depth =2,
        fine_pre_gateloop_depth= 2,
        attn_depth = 12,
        attn_heads = 12,
        fine_cross_attend_text = True,
        text_cond_with_film = False,
        cross_attn_num_mem_kv = 4,
        num_sos_tokens = 1,
        dropout  = 0.0,
        max_seq_len = 1500,
        fine_attn_depth = 2,
        condition_on_text = True,
        gateloop_use_heinsen = False,
        text_condition_model_types = "bge",
        text_condition_cond_drop_prob = 0.25,
    ).to("cuda")

    pkg = torch.load("./MeshGPT-transformer_trained_base.pt")
    transformer.load_state_dict(pkg['model'],strict=False)
    
    return transformer

@op
def train_transformer(context, transformer, dataset):
    batch_size =16
    grad_accum_every =4
    trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=grad_accum_every,num_train_steps=100, dataset = dataset,
                                     learning_rate = 1e-4, batch_size=batch_size, checkpoint_every_epoch = 25)

    for i in range(TRANSFORMER_EPOCHS_PER_SAVE, TRANSFORMER_TOTAL_EPOCHS + 1, TRANSFORMER_EPOCHS_PER_SAVE):
        trainer.train(i, stop_at_loss = 0.00005)
        pkg = dict( model = transformer.state_dict(), )
        torch.save(pkg, str(f"./MeshGPT-transformer_trained_{i}.pt"))

@job
def training_pipeline():
    autoencoder = autoencoder_asset()
    dataset = load_datasets()
    trained_autoencoder = train_autoencoder(autoencoder, dataset)
    saved_model = save_model(trained_autoencoder)
    transformer = create_transformer(trained_autoencoder)
    train_transformer(transformer, dataset)
    evaluate_model(trained_autoencoder, dataset)

if __name__ == "__main__":
    instance = DagsterInstance.get()
    result = execute_job(reconstructable(training_pipeline), instance=instance)