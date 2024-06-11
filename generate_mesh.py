# SPDX-License-Identifier: MIT
# Copyright (c) 2024 MarcusLoppe & K. S. Ernest (iFire) Lee

import torch  
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
)
import igl
from dagster import op, execute_job, job, reconstructable, DagsterInstance

device = "cuda" if torch.cuda.is_available() else "cpu"

autoencoder = MeshAutoencoder( 
        decoder_dims_through_depth =  (128,) * 6 + (192,) * 12 + (256,) * 24 + (384,) * 6,   
        codebook_size =  2048, 
        dim_codebook = 192,  
        dim_area_embed = 16,
        dim_coor_embed = 16, 
        dim_normal_embed = 16,
        dim_angle_embed = 8,
    
    attn_decoder_depth  = 4,
    attn_encoder_depth = 2
    ).to(device)

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
).to(device) 

pkg = torch.load("./mesh-transformer.ckpt.epoch_25_avg_loss_0.220.pt") 
transformer.load_state_dict(pkg['model'],strict=False)

from meshgpt_pytorch import mesh_render

def save_as_obj(file_path):
    v, f = igl.read_triangle_mesh(file_path)
    v, f, _, _ = igl.remove_unreferenced(v, f)
    c, _ = igl.orientable_patches(f)
    f, _ = igl.orient_outward(v, f, c)
    igl.write_triangle_mesh(file_path, v, f)
    return file_path

@op
def predict(context, text: str, num_input: int, num_temp: float):
    transformer.eval()
    labels = [label.strip() for label in text.split(',')]
    output = []
    if num_input > 1:
        for label in labels:
            output.append((transformer.generate(texts = [label ] * num_input, temperature = num_temp)))
    else:
        output.append((transformer.generate(texts = labels  , temperature = num_temp)))
    mesh_render.save_rendering('./render.obj', output)
    return save_as_obj('./render.obj')

@job
def prediction_pipeline():
    predict()
    
if __name__ == "__main__":
    run_config = {
        "ops": {
            "predict": {
                "inputs": {
                    "text": {"value": "hat, chair"},  # Enter labels, separated by commas.
                    "num_input": {"value": 1},
                    "num_temp": {"value": 0}
                }
            }
        }
    }
    instance = DagsterInstance.get()
    result = execute_job(reconstructable(prediction_pipeline), run_config=run_config, instance=instance)
