# +
import math

# calculate vit tflops
def vit_encoder_tflops_calculator(W,H,P, in_channels, L, hidden_size, intermediate_size=None):
    N = ((W+P-1)//P) * ((H+P-1)//P)
    seq_len = N
    if intermediate_size is None:
        tflops_one_transformer_layer_forward=24*seq_len*(hidden_size**2) + 4*hidden_size*(seq_len**2)
    else:
        tflops_one_transformer_layer_forward=8*seq_len*(hidden_size**2) + 4*hidden_size*(seq_len**2) + 4*seq_len*hidden_size*intermediate_size
    
    tflops_conv_forward = 2*N*hidden_size*in_channels*(P**2)
#     tflops_transformer_forward = (24*N*(hidden_size**2) + 4*hidden_size*(N**2))*L
    tflops_transformer_forward = tflops_one_transformer_layer_forward * L
    tflops=3*(tflops_conv_forward + tflops_transformer_forward)
    return tflops/1e12, N
    

# calculate linear adaptor tflops
def adaptor_tflops_calculator(s, hidden_size_vit, hidden_size_llm):
    tflops_forward=2*s*hidden_size_vit*hidden_size_llm
    tflops = 3*tflops_forward
    return tflops/1e12



# calculate llm decoder tflops
def llm_encoder_tflops_calculator(hidden_size, L, seq_len, intermediate_size=None):
    if intermediate_size is None:
        tflops_one_transformer_layer_forward=24*seq_len*(hidden_size**2) + 4*hidden_size*(seq_len**2)
    else:
        tflops_one_transformer_layer_forward=8*seq_len*(hidden_size**2) + 4*hidden_size*(seq_len**2) + 4*seq_len*hidden_size*intermediate_size
#     tflops_transformer_forward = (24*seq_len*(hidden_size**2) + 4*hidden_size*(seq_len**2))*L
    tflops_transformer_forward = tflops_one_transformer_layer_forward * L
    tflops = tflops_transformer_forward*3/1e12
    tflops_one_layer = tflops_one_transformer_layer_forward*3/1e12
    return tflops, tflops_one_layer


# 
def uneven_pp_parameters_generator(image_w, image_h, image_in_channels, patch_size, vit_num_layers, vit_hidden_size, vit_intermediate_size, 
              decoder_seq_len, llm_num_layers, llm_hidden_size, llm_intermediate_size, pp_size):
    vit_encoder_tflops, images_tokens_len = vit_encoder_tflops_calculator(image_w, image_h, patch_size, image_in_channels, vit_num_layers, vit_hidden_size, vit_intermediate_size)
    
    mlp_adaptor_tflops = adaptor_tflops_calculator(images_tokens_len, vit_hidden_size, llm_hidden_size)
    
    llm_decoder_tflops, llm_decoder_one_transformer_layer_tflops = llm_encoder_tflops_calculator(llm_hidden_size, llm_num_layers, decoder_seq_len, llm_intermediate_size)
    
    total_tflops = vit_encoder_tflops+mlp_adaptor_tflops + llm_decoder_tflops
    
    tflops_per_rank = total_tflops/pp_size
    
    llm_decoder_num_layers_per_rank = math.ceil(tflops_per_rank/llm_decoder_one_transformer_layer_tflops)
    
    # case 1: vit_encoder_tflops + mlp_adaptor_tflops >= tflops_per_rank
    if vit_encoder_tflops + mlp_adaptor_tflops >= tflops_per_rank:
        num_layers_first_pp_rank = 0
        tflops_per_rank = total_tflops/(pp_size-1)
        llm_decoder_num_layers_per_rank = math.ceil(tflops_per_rank/llm_decoder_one_transformer_layer_tflops)
        num_layers_last_pp_rank = llm_num_layers - llm_decoder_num_layers_per_rank*(pp_size-2)
    else:
        tflops_per_rank = total_tflops/pp_size
        llm_decoder_num_layers_per_rank = math.ceil(tflops_per_rank/llm_decoder_one_transformer_layer_tflops)
        num_layers_first_pp_rank = llm_num_layers - llm_decoder_num_layers_per_rank*(pp_size-1)
        num_layers_last_pp_rank = llm_decoder_num_layers_per_rank
#     print('vit_encoder_tflops:', vit_encoder_tflops)
    return num_layers_first_pp_rank, num_layers_last_pp_rank


# -

if __name__ == '__main__':
    # *********************************************************************************************
    # case 1
    # vit encoder parameters
    image_w = 224
    image_h = 224
    image_in_channels = 3
    patch_size = 14
    vit_hidden_size = 1280
    vit_intermediate_size = vit_hidden_size*4
    vit_num_layers = 28
    
    # adaptor parameters
    # vit_hidden_size = 1280
    # llm_hidden_size = 3584
    
    # llm decoder parameters
    decoder_seq_len = 1024
    llm_hidden_size = 3584
    llm_intermediate_size = 18944
    llm_num_layers = 28
    
    pp_size = 2
    
    num_layers_first_pp_rank, num_layers_last_pp_rank = uneven_pp_parameters_generator(
        image_w, image_h, image_in_channels, patch_size, vit_num_layers, vit_hidden_size, vit_intermediate_size, 
        decoder_seq_len, llm_num_layers, llm_hidden_size, llm_intermediate_size, pp_size)
    
    print('\ncase1:')
    print('num_layers_first_pp_rank:', num_layers_first_pp_rank)
    print('num_layers_last_pp_rank:', num_layers_last_pp_rank)
    
    # *********************************************************************************************
    # case 2
    
    # vit encoder parameters
    image_w = 336
    image_h = 336
    image_in_channels = 3
    patch_size = 14
    vit_hidden_size = 4096
    vit_intermediate_size = vit_hidden_size*4
    vit_num_layers = 28
    
    # adaptor parameters
    # vit_hidden_size = 1280
    # llm_hidden_size = 3584
    
    # llm decoder parameters
    decoder_seq_len = 2048
    llm_hidden_size = 3584
    llm_intermediate_size = 18944
    llm_num_layers = 28
    
    pp_size = 2
    
    num_layers_first_pp_rank, num_layers_last_pp_rank = uneven_pp_parameters_generator(
        image_w, image_h, image_in_channels, patch_size, vit_num_layers, vit_hidden_size, vit_intermediate_size, 
        decoder_seq_len, llm_num_layers, llm_hidden_size, llm_intermediate_size, pp_size)
    
    print('\ncase2:')
    print('num_layers_first_pp_rank:', num_layers_first_pp_rank)
    print('num_layers_last_pp_rank:', num_layers_last_pp_rank) 
    
    # *********************************************************************************************
    # case 3
    # vit encoder parameters
    image_w = 224
    image_h = 224
    image_in_channels = 3
    patch_size = 14
    vit_hidden_size = 8000
    vit_intermediate_size = vit_hidden_size*4
    vit_num_layers = 28
    
    # adaptor parameters
    # vit_hidden_size = 1280
    # llm_hidden_size = 3584
    
    # llm decoder parameters
    decoder_seq_len = 1024
    llm_hidden_size = 3584
    llm_intermediate_size = 18944
    llm_num_layers = 28
    
    pp_size = 2
    
    num_layers_first_pp_rank, num_layers_last_pp_rank = uneven_pp_parameters_generator(
        image_w, image_h, image_in_channels, patch_size, vit_num_layers, vit_hidden_size, vit_intermediate_size, 
        decoder_seq_len, llm_num_layers, llm_hidden_size, llm_intermediate_size, pp_size)
    
    print('\ncase3:')
    print('num_layers_first_pp_rank:', num_layers_first_pp_rank)
    print('num_layers_last_pp_rank:', num_layers_last_pp_rank) 


