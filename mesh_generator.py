import torch  
from meshgpt_pytorch import (
    MeshAutoencoder,
    MeshTransformer,
)

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
pkg = torch.load("./mesh-transformer.ckpt.epoch_25_avg_loss_0.220.pt") 
transformer.load_state_dict(pkg['model'],strict=False)
 
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


