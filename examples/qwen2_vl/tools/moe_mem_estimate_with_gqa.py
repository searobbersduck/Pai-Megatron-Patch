# copy from https://gitlab-master.nvidia.com/arch_moe_exploration/megatron-moe-scripts/-/blob/master/misc/tools/moe_mem_estimate_with_gqa.py?ref_type=heads
def calc(name, seq_len,
         n_layers, n_embed, vocab_size,
         n_head, query_group_num,
         ff_factor, n_experts, n_activated_experts, ffn_hidden,moe_ffn_hidden, first_k_dense=0,
         shared_expert_num=0, mtp=0, gpus=0, pp=0, vpp=0, ep=0, tp=0, etp=0, layers_per_pp=0,
         fsdp=False, fp8=False, fp8_primary=False):
    total_params, total_flops = 0, 0
    head_dim = n_embed // n_head
    print(f'{name} (seq_len={seq_len}):')
    
    billion_to_gb = 1e9 * 2 / 1024**3

    # Embedding
    embedding_params = n_embed * vocab_size / 1e9
    total_params += embedding_params
    print(f' - Embedding params: {embedding_params} B')

    embedding_memory = embedding_params * billion_to_gb
    print(f' - Embedding memory size: {embedding_memory} GB')

    # Attention
    # attn_proj_params = n_layers * n_embed * n_embed * 2 / 1e9  # Q, O project
    # attn_proj_params += n_layers * n_embed * n_head_kv * head_dim * 2 / 1e9  # K, V project
    attn_proj_params = n_layers * n_embed * n_head * head_dim  # Q project
    attn_proj_params += n_layers * n_embed * query_group_num * head_dim  # KV
    attn_proj_params += n_layers * n_embed * n_head * head_dim  # O project
    attn_proj_params /= 1e9
    attn_proj_flops = attn_proj_params * seq_len * 2 * 1e9
    attn_proj_flops /= 1e12
    # qk_head_dim, v_head_dim = head_dim, head_dim
    attn_flops = n_layers * n_head * seq_len * head_dim * seq_len / 2 * 2  # QK^T
    attn_flops += n_layers * n_head * seq_len * seq_len * head_dim / 2 * 2  # (QK^T)V
    attn_flops /= 1e12
    attn_flops += attn_proj_flops
    total_params += attn_proj_params
    total_flops += attn_flops

    attn_proj_memory = attn_proj_params * billion_to_gb
    print(f' - Attention memory size: {attn_proj_memory} GB')
    print(f' - Attention params: {attn_proj_params} B')
    print(f' - Attention FLOPs (per {seq_len} training forward tokens): {attn_flops} TFLOPs')

    # MLP
    hidden = n_embed * ff_factor * 8 // 3
    hidden = (hidden + 127) // 128 * 128
    # mlp_params = (n_layers - first_k_dense) * n_experts * (n_embed * hidden * 2 + hidden * n_embed) / 1e9
    # mlp_params += first_k_dense * n_activated_experts * (n_embed * hidden * 2 + hidden * n_embed) / 1e9
    # mlp_act_params = n_layers * n_activated_experts * (n_embed * hidden * 2 + hidden * n_embed) / 1e9
    # mlp_act_flops = n_layers * seq_len * n_activated_experts * (n_embed * hidden * 2 + hidden * n_embed) * 2 / 1e12
    mlp_params = (n_layers - first_k_dense) * n_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
    mlp_params += first_k_dense * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9
    mlp_act_params = (n_layers - first_k_dense) * n_activated_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
    mlp_act_params += first_k_dense * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) / 1e9
    mlp_act_flops = (n_layers - first_k_dense) * seq_len * n_activated_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) * 2 / 1e12
    mlp_act_flops += first_k_dense * seq_len * (n_embed * ffn_hidden * 2 + ffn_hidden * n_embed) * 2 / 1e12
    total_params += mlp_params
    total_flops += mlp_act_flops
    mlp_memory = mlp_params * billion_to_gb
    print(f' - MLP hidden: {hidden}')
    print(f' - MLP params: {mlp_params} B')
    print(f' - MLP memory size: {mlp_memory} GB')
    print(f' - MLP activated params (per token): {mlp_act_params} B')
    print(f' - MLP activated FLOPs (per {seq_len} training forward tokens): {mlp_act_flops} TFLOPs')

    # Head
    head_params = n_embed * vocab_size / 1e9
    head_flops = seq_len * n_embed * vocab_size * 2 / 1e12
    total_params += head_params
    total_flops += head_flops
    head_memory = head_params * billion_to_gb
    total_memory = total_params * billion_to_gb
    print(f' - Head params: {head_params} B')
    print(f' - Head memory size: {head_memory} GB')
    print(f' - Head FLOPs (per {seq_len} training forward tokens): {head_flops} TFLOPs')

    # Gating
    gating_flops = (n_layers - first_k_dense) * n_experts * n_embed * seq_len * 2 / 1e12
    total_flops += gating_flops
    print(f' - Gating FLOPs (per {seq_len} training forward tokens): {gating_flops} TFLOPs')

    # Total
    print(f' - Total params: {total_params} B')
    print(f' - Total memory size: {total_memory} GB')
    print(f' - Total activated params (per token): {total_params + mlp_act_params - mlp_params - embedding_params} B')
    print(f' - Total FLOPs (per {seq_len} training forward tokens): {total_flops} TFLOPs')
    print(f' - Total FLOPs (per forward token): {total_flops / seq_len} TFLOPs')
    print(f' - Total FLOPs (fwd and bwdper {seq_len} training forward tokens): {total_flops * 3} TFLOPs')
    print()

    # MTP
    mtp_proj_params = mtp * n_embed * n_embed * 2 / 1e9
    mtp_attn_params = attn_proj_params / n_layers * mtp
    mtp_mlp_params = n_experts * (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9 * mtp
    mtp_params = mtp_proj_params + mtp_attn_params + mtp_mlp_params
    mtp_flops = (attn_flops + mlp_act_flops) / n_layers + gating_flops / (n_layers - first_k_dense) + head_flops + mtp_proj_params * seq_len * 2 / 1e3
    print(f' - MTP params: {mtp_params} B')
    print(f' - MTP FLOPs (per {seq_len} training forward tokens): {mtp_flops} TFLOPs')
    print()

    dense_dp = gpus // pp // tp
    moe_dp = gpus // pp // ep // etp
    print(f' - GPUs{gpus} PP{pp} VPP{vpp} EP{ep} TP{tp} ETP{etp} denseDP{dense_dp} EDP{moe_dp} FSDP{fsdp}')

    one_expert_params = (n_embed * moe_ffn_hidden * 2 + moe_ffn_hidden * n_embed) / 1e9
    moe_layer_dense_params = attn_proj_params / n_layers + one_expert_params * shared_expert_num
    moe_layer_moe_params = one_expert_params * (n_experts - shared_expert_num) / ep
    if fp8:
        # fp32 grads: 4, bf16 weights: 2, fp8 weights: 1, fp8 transposed weights: 1, total: 8
        # fp32 reminder master weights: 2, bf16 m and v: 2 + 2, total: 6
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * ((6 if fp8_primary else 8) + 6.0 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * ((6 if fp8_primary else 8) + 6.0 / moe_dp) * 1e9 / 1024**3
    else:
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (6 + 12.0 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (6 + 12.0 / moe_dp) * 1e9 / 1024**3
    if fsdp:
        assert not fp8
        rank_dense_mem = layers_per_pp * moe_layer_dense_params / tp * (18.0 / dense_dp) * 1e9 / 1024**3
        rank_moe_mem = layers_per_pp * moe_layer_moe_params / etp * (18.0 / moe_dp) * 1e9 / 1024**3 + moe_layer_moe_params / etp * 12.0 * 1e9 / 1024**3
    print(f' - Dense Param Mem per rank: {rank_dense_mem} GB')
    print(f' - MoE Param Mem per rank: {rank_moe_mem} GB')
    print(f' - Total Param Mem per rank: {rank_dense_mem + rank_moe_mem} GB')
    print()

    topk = n_activated_experts - shared_expert_num
    bf16_mb_coeff = 2 / 1024 / 1024
    fp8_mb_coeff = 1 / 1024 / 1024
    fp32_mb_coeff = 4 / 1024 / 1024
    int64_mb_coeff = 8 / 1024 / 1024

    if not fp8:
        input_mem = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        input_norm_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        q_out = seq_len * 1 * n_head * head_dim / tp * bf16_mb_coeff
        kv_out = seq_len * 1 * n_head * query_group_num / tp * bf16_mb_coeff * 2
        attn_out = seq_len * 1 * n_head * head_dim / tp * bf16_mb_coeff
        attn_ctx_tensor = 1 * n_head / tp * seq_len * 1 * fp32_mb_coeff
        proj_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        attn_bda_out = proj_out
        mlp_norm_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        shared_AG_out = seq_len * 1 * n_embed * bf16_mb_coeff
        router_probs = seq_len / tp * (n_experts - shared_expert_num) * bf16_mb_coeff
        permute_row_id_map = seq_len / tp * (n_experts - shared_expert_num) * int64_mb_coeff
        share_linear_1_out = seq_len * 1 * moe_ffn_hidden / tp * shared_expert_num * 2 * bf16_mb_coeff
        share_act_out = share_linear_1_out / 2
        share_linear_2_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        permute_out = seq_len / tp * etp * topk * 1 * n_embed * bf16_mb_coeff
        expert_linear_1_out = seq_len / tp * etp * topk * 1 * moe_ffn_hidden / etp * 2 * bf16_mb_coeff
        expert_act_out = expert_linear_1_out / 2
        expert_linear_2_out = seq_len / tp * etp * topk * 1 * n_embed * bf16_mb_coeff
        unpermute_alltoall_out = expert_linear_2_out / etp
        unpermute_out = unpermute_alltoall_out / topk
        mlp_bda_out = unpermute_out
        cached = input_mem + input_norm_out + q_out + kv_out + attn_out + attn_ctx_tensor + \
            attn_bda_out + shared_AG_out + \
            router_probs + permute_row_id_map + \
            share_linear_1_out + share_act_out + \
            permute_out + expert_linear_1_out + expert_act_out + unpermute_alltoall_out
        cached_layer_num = layers_per_pp * (pp - 1)
        if vpp > 1:
            cached_layer_num += (layers_per_pp // vpp) * (pp - 1)
        cached_t = cached * cached_layer_num
        print(f' -- input tensor: {input_mem} MB, cached by input norm            {input_mem / cached * 100:.2f}%')
        print(f' -- input norm output: {input_norm_out} MB cached by qkv            {input_norm_out / cached * 100:.2f}%')
        print(f' -- q_out: {q_out} MB cached by core_attn            {q_out / cached * 100:.2f}%')
        print(f' -- kv_out: {kv_out} MB cached by core_attn            {kv_out / cached * 100:.2f}%')
        print(f' -- attn_out: {attn_out} MB cached by proj_out and attn itself            {attn_out / cached * 100:.2f}%')
        print(f' -- attn_ctx_tensor: {attn_ctx_tensor} MB cached by attn itself            {attn_ctx_tensor / cached * 100:.2f}%')
        print(f' -- proj_out: {proj_out} MB not cached')
        print(f' -- attn_bda_out: {attn_bda_out} MB cached by mlp_norm            {attn_bda_out / cached * 100:.2f}%')
        print(f' -- mlp_norm_out: {mlp_norm_out} MB not cached')
        print(f' -- shared_AG_out: {shared_AG_out} MB cached by shared expert            {shared_AG_out / cached * 100:.2f}%')
        print(f' -- router_probs: {router_probs} MB cached by fused unpermute            {router_probs / cached * 100:.2f}%')
        print(f' -- permute_row_id_map: {permute_row_id_map} MB cached by fused (un)permute            {permute_row_id_map / cached * 100:.2f}%')
        print(f' -- share_linear_1_out: {share_linear_1_out} MB cached by share_act            {share_linear_1_out / cached * 100:.2f}%')
        print(f' -- share_act_out: {share_act_out} MB cached by share_linear_2            {share_act_out / cached * 100:.2f}%')
        print(f' -- share_linear_2_out {share_linear_2_out} MB not cached')
        print(f' -- permute_out {permute_out} MB cached by expert_linear_1            {permute_out / cached * 100:.2f}%')
        print(f' -- expert_linear_1_out: {expert_linear_1_out} MB cached by expert_act            {expert_linear_1_out / cached * 100:.2f}%')
        print(f' -- expert_act_out: {expert_act_out} MB cached by expert_linear_2            {expert_act_out / cached * 100:.2f}%')
        print(f' -- expert_linear_2_out: {expert_linear_2_out} MB not cached')
        print(f' -- unpermute_alltoall_out: {unpermute_alltoall_out} MB cached by unpermute            {unpermute_alltoall_out / cached * 100:.2f}%')
        print(f' -- unpermute_out: {unpermute_out} MB not cached')
        print(f' -- mlp_bda_out: {mlp_bda_out} MB not cached (sent to next layer)')

        print()
        print(f' -- cached micobatch layer num: {cached_layer_num}')
        print(f' -- total cached for 1 layer and 1 micobatch: {cached} MB')
        print(f' -- cached for all PP microbatches: {cached_t / 1024} GB')
        print(f' -- total usage {rank_dense_mem + rank_moe_mem + cached_t / 1024} GB')
        print()

        print(f' -- full recompute total cached for 1 layer and 1 micobatch: {input_mem} MB')
        print(f' -- full recompute cached for all PP microbatches: {input_mem * cached_layer_num / layers_per_pp / 1024} GB')
        print(f' -- full recompute total usage {rank_dense_mem + rank_moe_mem + input_mem * cached_layer_num / layers_per_pp / 1024} GB')
        print()

        probs2swiglu_save = unpermute_alltoall_out
        probs2swiglu_save_t = probs2swiglu_save * cached_layer_num
        print(f' --- By probs2swiglu, can save {probs2swiglu_save} MB for 1 layer and 1 micobatch')
        print(f' --- By probs2swiglu, can save {probs2swiglu_save_t / 1024} GB for all PP microbatches')
        cached_after_probs2swiglu = cached_t - probs2swiglu_save_t
        print(f' --- Cached size after probs2swiglu: {cached_after_probs2swiglu / 1024} GB')
        print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_probs2swiglu / 1024} GB')
        print()

        act_func_save = share_act_out + expert_act_out
        act_func_save_t = act_func_save * cached_layer_num
        print(f' --- By act_func recompute, can save {act_func_save} MB for 1 layer and 1 micobatch')
        print(f' --- By act_func recompute, can save {act_func_save_t / 1024} GB for all PP microbatches')
        norm_save = input_norm_out + mlp_norm_out
        norm_save_t = norm_save * cached_layer_num
        print(f' --- By norm recompute, can save {norm_save} MB')
        print(f' --- By norm recompute, can save {norm_save_t / 1024} GB for all PP microbatches')
        cached_after_recompute = cached_after_probs2swiglu - act_func_save_t - norm_save_t
        print(f' --- Cached size after the above recomputations: {cached_after_recompute / 1024} GB')
        print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_recompute / 1024} GB')
        print()

        fc1_offloading_save = permute_out
        fc1_offloading_save_t = fc1_offloading_save * cached_layer_num
        print(f' --- By fc1 offloading, can save {fc1_offloading_save} MB for 1 layer and 1 micobatch')
        print(f' --- By fc1 offloading, can save {fc1_offloading_save_t / 1024} GB for all PP microbatches')
        cached_after_offloading = cached_after_recompute - fc1_offloading_save_t
        print(f' --- Cached size after the above offloading: {cached_after_offloading / 1024} GB')
        print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_offloading / 1024} GB')
        print()

        shared_expert_save = share_linear_1_out + share_act_out
        shared_expert_save_t = shared_expert_save * cached_layer_num
        print(f' --- By shared expert recompute, can save {shared_expert_save} MB for 1 layer and 1 micobatch')
        print(f' --- By shared expert recompute, can save {shared_expert_save_t / 1024} GB for all PP microbatches')
        cached_after_shared_expert = cached_after_offloading - shared_expert_save_t
        print(f' --- Cached size after the above recomputations: {cached_after_shared_expert / 1024} GB')
        print(f' --- total usage {rank_dense_mem + rank_moe_mem + cached_after_shared_expert / 1024} GB')
        print()

    else:
        # fp8
        input_mem = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        input_norm_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        q_in = seq_len * 1 * n_embed / tp * fp8_mb_coeff
        kv_in = q_in
        q_out = seq_len * 1 * n_head * head_dim / tp * bf16_mb_coeff
        kv_out = seq_len * 1 * n_head * query_group_num / tp * bf16_mb_coeff * 2
        attn_out = seq_len * 1 * n_head * head_dim / tp * bf16_mb_coeff
        attn_ctx_tensor = 1 * n_head / tp * seq_len * 1 * fp32_mb_coeff
        proj_in = seq_len * 1 * n_head * head_dim / tp * fp8_mb_coeff
        proj_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        attn_bda_out = proj_out
        mlp_norm_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        router_probs = seq_len / tp * (n_experts - shared_expert_num) * fp32_mb_coeff
        router_others = router_probs * 2
        permute_row_id_map = seq_len / tp * (n_experts - shared_expert_num) / ep * topk * int64_mb_coeff
        expert_linear_1_in = seq_len / tp * etp * topk * 1 * n_embed * fp8_mb_coeff
        expert_linear_1_out = seq_len / tp * etp * topk * 1 * moe_ffn_hidden / etp * 2 * bf16_mb_coeff
        expert_act_out = expert_linear_1_out / 2
        expert_linear_2_in = expert_act_out / bf16_mb_coeff * fp8_mb_coeff
        expert_linear_2_out = seq_len / tp * etp * topk * 1 * n_embed * bf16_mb_coeff
        share_linear_1_in = seq_len * 1 * n_embed * fp8_mb_coeff
        share_linear_1_out = seq_len * 1 * moe_ffn_hidden / tp * shared_expert_num * 2 * bf16_mb_coeff
        share_act_out = share_linear_1_out / 2
        share_linear_2_in = share_act_out / bf16_mb_coeff * fp8_mb_coeff
        share_linear_2_out = seq_len * 1 * n_embed / tp * bf16_mb_coeff
        mlp_bda_out = share_linear_2_out
        cached = input_mem + q_in + kv_in + q_out + kv_out + attn_out + attn_ctx_tensor + \
            proj_in + attn_bda_out + mlp_norm_out + \
            router_probs + router_others + permute_row_id_map + \
            expert_linear_1_in + expert_linear_1_out + expert_linear_2_in + \
            share_linear_1_in + share_linear_1_out + share_linear_2_in
        cached_layer_num = layers_per_pp * (pp - 1)
        if vpp > 1:
            cached_layer_num += (layers_per_pp // vpp) * (pp - 1)
        cached_t = cached * cached_layer_num
        print(f' -- input tensor: {input_mem} MB, cached by input norm            {input_mem / cached * 100:.2f}%')
        print(f' -- input norm output: {input_norm_out} MB not cached')
        print(f' -- q_in: {q_in} MB cached by q            {q_in / cached * 100:.2f}%')
        print(f' -- kv_in: {kv_in} MB cached by kv            {kv_in / cached * 100:.2f}%')
        print(f' -- q_out: {q_out} MB cached by core_attn            {q_out / cached * 100:.2f}%')
        print(f' -- kv_out: {kv_out} MB cached by core_attn            {kv_out / cached * 100:.2f}%')
        print(f' -- attn_out: {attn_out} MB cached by attn itself            {attn_out / cached * 100:.2f}%')
        print(f' -- attn_ctx_tensor: {attn_ctx_tensor} MB cached by attn itself            {attn_ctx_tensor / cached * 100:.2f}%')
        print(f' -- proj_in: {proj_in} MB cached by proj            {proj_in / cached * 100:.2f}%')
        print(f' -- proj_out: {proj_out} MB not cached')
        print(f' -- attn_bda_out: {attn_bda_out} MB cached by mlp_norm            {attn_bda_out / cached * 100:.2f}%')
        print(f' -- mlp_norm_out: {mlp_norm_out} MB cached by router_gating            {mlp_norm_out / cached * 100:.2f}%')
        print(f' -- router_probs (fp32): {router_probs} MB cached by fused unpermute            {router_probs / cached * 100:.2f}%')
        print(f' -- router_others (not accurate): {router_others} MB cached by fused unpermute            {router_others / cached * 100:.2f}%')
        print(f' -- permute_row_id_map: {permute_row_id_map} MB cached by fused (un)permute            {permute_row_id_map / cached * 100:.2f}%')
        print(f' -- expert_linear_1_in: {expert_linear_1_in} MB cached by expert_linear_1            {expert_linear_1_in / cached * 100:.2f}%')
        print(f' -- expert_linear_1_out: {expert_linear_1_out} MB cached by expert_act            {expert_linear_1_out / cached * 100:.2f}%')
        print(f' -- expert_act_out: {expert_act_out} MB not cached')
        print(f' -- expert_linear_2_in {expert_linear_2_in} MB cached by expert_linear_2            {expert_linear_2_in / cached * 100:.2f}%')
        print(f' -- expert_linear_2_out: {expert_linear_2_out} MB not cached')
        print(f' -- share_linear_1_in: {share_linear_1_in} MB cached by share_linear_1            {share_linear_1_in / cached * 100:.2f}%')
        print(f' -- share_linear_1_out: {share_linear_1_out} MB cached by share_act            {share_linear_1_out / cached * 100:.2f}%')
        print(f' -- share_act_out: {share_act_out} MB not cached')
        print(f' -- share_linear_2_in: {share_linear_2_in} MB cached by share_linear_2            {share_linear_2_in / cached * 100:.2f}%')
        print(f' -- share_linear_2_out {share_linear_2_out} MB not cached')
        print(f' -- mlp_bda_out: {mlp_bda_out} MB not cached (sent to next layer)')

        print()
        print(f' -- cached micobatch layer num: {cached_layer_num}')
        print(f' -- total cached for 1 layer and 1 micobatch: {cached} MB')
        print(f' -- cached for all PP microbatches: {cached_t / 1024} GB')
        print(f' -- total usage {rank_dense_mem + rank_moe_mem + cached_t / 1024} GB')
        print()

        print(f' -- full recompute total cached for 1 layer and 1 micobatch: {input_mem} MB')
        print(f' -- full recompute cached for all PP microbatches: {input_mem * cached_layer_num / layers_per_pp / 1024} GB')
        print(f' -- full recompute total usage {rank_dense_mem + rank_moe_mem + input_mem * cached_layer_num / layers_per_pp / 1024} GB')
        print()


if __name__ == '__main__':

    calc('moe_671b_lora', seq_len=4096,
         n_layers=61, n_embed=7168, vocab_size=129280,
         n_head=128, query_group_num=8,
         ff_factor=0.1125, n_experts=257, n_activated_experts=9,
         ffn_hidden=18432, moe_ffn_hidden=2048,
         first_k_dense=3,
         shared_expert_num=1, mtp=1, gpus=2048, pp=8, vpp=4, ep=64, tp=1, etp=1, layers_per_pp=8, fsdp=False, fp8=True, fp8_primary=True)
