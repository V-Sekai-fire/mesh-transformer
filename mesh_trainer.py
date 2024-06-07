# # %%
# !pip install git+https://github.com/MarcusLoppe/meshgpt-pytorch.git
# !pip install matplotlib
# %cd /root/text_to_mesh

# %%
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


# %%
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
    ).to("cuda")     

# %%

torch.cuda.empty_cache()
gc.collect()  
 
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
pkg = torch.load("./MeshGPT-transformer_trained_01.pt") 
transformer.load_state_dict(pkg['model'],strict=False)
 

# %%
dataset = MeshDataset.load("./labels_885_10x5_21720_mod.npz") 
labels = list(set(item['texts'] for item in dataset.data))
dataset.generate_codes(autoencoder,150)
dataset.data[0].keys() 
dataset.embed_texts(transformer,1)
 

# %%
 
torch.cuda.empty_cache()
gc.collect()  
 
batch_size =16
grad_accum_every =4      
trainer = MeshTransformerTrainer(model = transformer,warmup_steps = 10,grad_accum_every=grad_accum_every,num_train_steps=100, dataset = dataset, 
                                 learning_rate = 1e-4, batch_size=batch_size ,checkpoint_every_epoch = 25) 
loss = trainer.train(740, stop_at_loss = 0.00005)   

# %%
pkg = dict( model = transformer.state_dict(), ) 
torch.save(pkg, str("./MeshGPT-transformer_trained.pt"))

# %%
# We use this to load independently of training.
# pkg = torch.load("./MeshGPT-transformer_trained_01.pt") 
# transformer.load_state_dict(pkg['model'],strict=False)

from meshgpt_pytorch import mesh_render
text_coords = []
rows = []
transformer.eval() 

for text in [ 'desk', 'couch', 'screen', 'stool', 'armchair', 'dining table' ]:
    print("Generating ", text) 
    face_coords = transformer.generate(texts = [text] ,  temperature = 0.0)
    text_coords.append(face_coords)

rows.append(text_coords)
mesh_render.combind_mesh_with_rows(f'./rows.obj', rows)


