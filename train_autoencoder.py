# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2024 MarcusLoppe & K. S. Ernest (iFire) Lee

from dagster import op, job
import torch
import datetime
import random
import tqdm
import os
from meshgpt_pytorch import MeshAutoencoderTrainer, MeshAutoencoder, MeshDataset, mesh_render
from dagster import execute_job, reconstructable, DagsterInstance, In, Out, DynamicOut, DynamicOutput, graph_asset, asset, DagsterType
from typing import Tuple

@asset
def autoencoder_asset():
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
    return autoencoder

@asset
def datasets_asset():
    dataset = MeshDataset.load("./shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")  
    # dataset2 = MeshDataset.load("./objverse_250f_45.9M_3086_labels_53730_10_min_x1_aug.npz")
    # dataset.data.extend(dataset2.data)  
    # dataset2 = MeshDataset.load("./objverse_250f_229.7M_3086_labels_268650_10_min_x5_aug.npz")
    # dataset.data.extend(dataset2.data) 
    dataset.sort_dataset_keys()
    return dataset

@op(
    ins={"dataset": In(),
         "autoencoder": In(metadata={
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_"),
        })},
    out={
        "autoencoder": Out(is_required=True),
        "mse_obj": Out(is_required=True),
    },
)
def evaluate_model_op(context, autoencoder, dataset) -> Tuple[MeshAutoencoder, str]:
    min_mse, max_mse = float('inf'), float('-inf')
    min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
    random_samples, random_samples_pred, all_random_samples = [], [], []
    total_mse, sample_size = 0.0, 200

    random.shuffle(dataset.data)
    autoencoder = autoencoder.to("cuda")
    for item in tqdm.tqdm(dataset.data[:sample_size]):
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

    context.log.info(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
    mesh_render.save_rendering('./mse_rows.obj', all_random_samples)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_")
    with open(f"./mse_rows_{timestamp}.obj", "r") as file:
        mse_obj = file.read()
        
    return autoencoder, mse_obj


@op(tags={"dagster/concurrency_key": "train"})
def train_autoencoder(autoencoder, dataset) -> Tuple[MeshAutoencoder, float]:
    batch_size=16
    grad_accum_every =4
    learning_rate = 1e-3
    autoencoder.commit_loss_weight = 0.2
    autoencoder_trainer = MeshAutoencoderTrainer(model =autoencoder ,warmup_steps = 10, dataset = dataset, num_train_steps=100,
                                            batch_size=batch_size,
                                            grad_accum_every = grad_accum_every,
                                            learning_rate = learning_rate,
                                            checkpoint_every_epoch=5)
    loss = autoencoder_trainer.train(1, diplay_graph= False)   
    return (autoencoder, loss)

@op(
    ins={"autoencoder": In()},
    out={
        "autoencoder": Out()
    },
)
def save_model_op(autoencoder, loss) -> MeshAutoencoder:
    pkg = dict(model=autoencoder.state_dict())
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_")
    filename = f"./MeshGPT_autoencoder_{timestamp}.pt"
    torch.save(pkg, filename)
    return Out(
        autoencoder,
        metadata={
            "time": timestamp,
            "loss": str(loss)
        }
    )

@op
def train_autoencoder_asset(model, datasets) -> Tuple[MeshAutoencoder, float]:
    model, loss = train_autoencoder(model, datasets)
    model = save_model_op(model, loss)
    # model, _mse_obj = evaluate_model_op(model, datasets)
    return (model, loss)

@op
def train_autoencoder_twice(model, datasets) -> Tuple[MeshAutoencoder, float]:
    model, loss = train_autoencoder_asset(model, datasets)
    model, loss = train_autoencoder_asset(model, datasets)
    return model

@graph_asset
def autoencoder_01() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_asset(autoencoder_asset(), datasets_asset())

@graph_asset
def autoencoder_02() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_01(), datasets_asset())

@graph_asset
def autoencoder_04() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_02(), datasets_asset())

@graph_asset
def autoencoder_08() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_04(), datasets_asset())

@graph_asset
def autoencoder_16() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_08(), datasets_asset())

@graph_asset
def autoencoder_32() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_16(), datasets_asset())

@graph_asset
def autoencoder_64() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_32(), datasets_asset())

@graph_asset
def autoencoder_128() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_64(), datasets_asset())

@graph_asset
def autoencoder_256() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_128(), datasets_asset())

@graph_asset
def autoencoder_512() -> Tuple[MeshAutoencoder, float]:
    return train_autoencoder_twice(autoencoder_256(), datasets_asset())
