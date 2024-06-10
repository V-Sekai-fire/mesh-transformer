# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2024 MarcusLoppe & K. S. Ernest (iFire) Lee

from dagster import op, job
import torch
import datetime
import random
import tqdm
import os
from meshgpt_pytorch import MeshAutoencoderTrainer, MeshAutoencoder, MeshDataset, mesh_render
from dagster import execute_job, reconstructable, DagsterInstance, In, Out, DynamicOut, DynamicOutput

@op
def create_autoencoder(context):
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

@op
def load_datasets(context):
    dataset = MeshDataset.load("./shapenet_250f_2.2M_84_labels_2156_10_min_x1_aug.npz")  
    # dataset2 = MeshDataset.load("./objverse_250f_45.9M_3086_labels_53730_10_min_x1_aug.npz")
    # dataset.data.extend(dataset2.data)  
    # dataset2 = MeshDataset.load("./shapenet_250f_21.9M_84_labels_21560_10_min_x10_aug.npz")  
    # dataset.data.extend(dataset2.data)
    # dataset2 = MeshDataset.load("./objverse_250f_229.7M_3086_labels_268650_10_min_x5_aug.npz")
    # dataset.data.extend(dataset2.data) 
    # dataset.sort_dataset_keys()
    return dataset


@op(
    ins={"autoencoder": In(metadata={
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_"),
            "model_size_bytes": str(os.path.getsize("./MeshGPT-autoencoder.pt")),
        })},
    out={
        "trained_autoencoder": Out(is_required=False)
    },
)
def save_model(context, autoencoder_tuple):
    autoencoder, loss = autoencoder_tuple
    pkg = dict( model = autoencoder.state_dict(), )
    filename = "./MeshGPT-autoencoder.pt"
    torch.save(pkg, filename)
    context.log.info(f'Saved model with loss {loss}')

@op(
    ins={"dataset": In(),
         "autoencoder": In(metadata={
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_"),
            "model_size_bytes": str(os.path.getsize("./MeshGPT-autoencoder.pt")),
        })},
    out={
        "autoencoder": Out(),
        "mse_obj": Out()
    },
)
def evaluate_model(context, autoencoder, dataset):
    min_mse, max_mse = float('inf'), float('-inf')
    min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
    random_samples, random_samples_pred, all_random_samples = [], [], []
    total_mse, sample_size = 0.0, 200

    random.shuffle(dataset.data)
    autoencoder = autoencoder.to("cuda")
    for item in tqdm.tqdm(dataset.data[:sample_size]):  # Use tqdm.tqdm here
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
    mesh_render.save_rendering('./mse_rows.obj', all_random_samples)
    with open("./mse_rows.obj", "r") as file:
        mse_obj = file.read()
        
    return autoencoder, mse_obj

@op(ins={"autoencoder": In(), "dataset": In()}, out={"autoencoder": Out(metadata={"time": datetime.datetime.now(datetime.timezone.utc).isoformat().replace(":", "_"),"model_size_bytes": str(os.path.getsize("./MeshGPT-autoencoder.pt")),}),        "loss": Out()},)
def train_autoencoder(autoencoder, dataset) -> tuple[MeshAutoencoder, float]:
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


@op(ins={"autoencoder": In(), "dataset": In(), "max_epochs": In(), "min_loss": In()}, out=DynamicOut())
def train_epochs(context, autoencoder, dataset, max_epochs, min_loss):
    for epoch in range(max_epochs):
        trained_autoencoder = train_autoencoder(autoencoder, dataset)
        new_autoencoder, new_loss = trained_autoencoder 
        
        if new_loss < min_loss:
            min_loss = new_loss
            context.log.info(f'New minimum loss {min_loss} at epoch {epoch}')
            
        yield DynamicOutput((new_autoencoder, new_loss), mapping_key=str(epoch))
        
        if new_loss <= 0.01:
            break

@op
def get_max_epochs():
    return 1 # 740

@op
def get_min_loss():
    return float('inf')

@op
def save_and_evaluate_model(autoencoder):
    save_model(autoencoder)
    evaluate_model(autoencoder)

@job
def train_autoencoder_job():
    autoencoder = create_autoencoder()
    dataset = load_datasets()
    max_epochs = get_max_epochs()
    min_loss = get_min_loss()
    epochs = train_epochs(autoencoder, dataset, max_epochs, min_loss)
    epochs.map(save_model).map(save_and_evaluate_model)

if __name__ == "__main__":
    instance = DagsterInstance.get()
    result = execute_job(reconstructable(train_autoencoder_job), instance=instance)