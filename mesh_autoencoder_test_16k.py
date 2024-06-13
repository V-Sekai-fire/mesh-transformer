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
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13096, "status": "ok", "timestamp": 1718255361660, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="b2e9bbe1" outputId="f4aae7de-2006-4fc2-a174-ebcba19ae865"
# %pip install git+https://github.com/MarcusLoppe/meshgpt-pytorch.git
# %pip install matplotlib

# + executionInfo={"elapsed": 7, "status": "ok", "timestamp": 1718255363914, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="7baeb840"
from pathlib import Path
import gc
import torch
import os
from meshgpt_pytorch import (
    MeshTransformerTrainer,
    MeshAutoencoderTrainer,
    MeshAutoencoder,
    MeshTransformer,MeshDataset
)
from meshgpt_pytorch.data import (
    derive_face_edges_from_faces
)


# + executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1718255363914, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="655619af"
device = "cuda" if torch.cuda.is_available() else "cpu"

# + executionInfo={"elapsed": 6, "status": "ok", "timestamp": 1718255363914, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="ce60e2b3"
autoencoder = MeshAutoencoder(
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,
        dim_codebook = 192,
        dim_area_embed = 16,
        dim_coor_embed = 16,
        dim_normal_embed = 16,
        dim_angle_embed = 8,
        attn_decoder_depth  = 4,
        attn_encoder_depth = 2)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 53515, "status": "ok", "timestamp": 1718255417425, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="37ecc8ed" outputId="55caaae5-c4b4-4739-9ab3-d185ed2542c3"
dataset = MeshDataset.load("./mesh-transformer-datasets/objverse_250f_229.7M_3086_labels_268650_10_min_x5_aug.npz")
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

# + executionInfo={"elapsed": 16640, "status": "ok", "timestamp": 1718255434061, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="b3a9cefa"
pkg = torch.load("./mesh-transformer-datasets/16k_autoencoder_229M_0.338.pt", map_location=torch.device(device))
autoencoder.load_state_dict(pkg['model'])
for param in autoencoder.parameters():
    param.requires_grad = True

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27946, "status": "ok", "timestamp": 1718255462001, "user": {"displayName": "Ernest Lee", "userId": "17643879017059299399"}, "user_tz": 420} id="61c2cc14" outputId="a423d436-4b7b-467d-eae5-5e4de042b84e"
import torch

def combined_mesh_with_rows(path, meshes):
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    translation_distance = 0.5
    obj_file_content = ""

    for row, mesh in enumerate(meshes):
        for r, faces_coordinates in enumerate(mesh):
            numpy_data = faces_coordinates[0].cpu().numpy().reshape(-1, 3)
            numpy_data[:, 0] += translation_distance * (r / 0.2 - 1)
            numpy_data[:, 2] += translation_distance * (row / 0.2 - 1)

            for vertex in numpy_data:
                all_vertices.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            for i in range(1, len(numpy_data), 3):
                all_faces.append(f"f {i + vertex_offset} {i + 1 + vertex_offset} {i + 2 + vertex_offset}\n")

            vertex_offset += len(numpy_data)

        obj_file_content = "".join(all_vertices) + "".join(all_faces)

    with open(path , "w") as file:
        file.write(obj_file_content)

from meshgpt_pytorch import MeshAutoencoderTrainer, MeshAutoencoder, MeshDataset, mesh_render
import tqdm
import datetime
min_mse, max_mse = float('inf'), float('-inf')
min_coords, min_orgs, max_coords, max_orgs = None, None, None, None
random_samples, random_samples_pred, all_random_samples = [], [], []
total_mse, sample_size = 0.0, 200

autoencoder = autoencoder.to(device)

for item in tqdm.tqdm(dataset.data[:sample_size]):  # Use tqdm.tqdm here
    item['faces'] = item['faces'].to(device)
    item['vertices'] = item['vertices'].to(device)
    item['face_edges'] = item['face_edges'].to(device)
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
        random_samples.append(coords)
    else:
        all_random_samples.extend([ random_samples])
        random_samples, random_samples_pred = [], []

print(f'MSE AVG: {total_mse / sample_size:.10f}, Min: {min_mse:.10f}, Max: {max_mse:.10f}')
combined_mesh_with_rows(f'./mse_rows.obj', all_random_samples)

