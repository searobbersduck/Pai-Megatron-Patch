from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core import InferenceParams
from megatron.core.models.vision.multimodal_projector import MultimodalProjector

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from contextlib import nullcontext
from megatron.core.utils import is_te_min_version, make_viewless_tensor


# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

# copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs.float()

# copied from 
# class Qwen2_5_VLVisionBlock(nn.Module):
#     def __init__(self, config, attn_implementation: str = "sdpa") -> None:
#         super().__init__()
#         self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
#         self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
#         self.attn = QWEN2_5_VL_VISION_ATTENTION_CLASSES[attn_implementation](
#             config.hidden_size, num_heads=config.num_heads
#         )
#         self.mlp = Qwen2_5_VLMLP(config, bias=True)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         cu_seqlens: torch.Tensor,
#         rotary_pos_emb: Optional[torch.Tensor] = None,
#         position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#     ) -> torch.Tensor:
#         hidden_states = hidden_states + self.attn(
#             self.norm1(hidden_states),
#             cu_seqlens=cu_seqlens,
#             rotary_pos_emb=rotary_pos_emb,
#             position_embeddings=position_embeddings,
#         )
#         hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
#         return hidden_states
    
class Qwen2_5_VLVisionBlock(TransformerBlock):
    def __init__(self, config, spec, post_layer_norm = True, pre_process = True, post_process = True):
        super().__init__(config, spec, post_layer_norm, pre_process, post_process)
        self.window_size = 112
        self.spatial_merge_size = 2
        self.patch_size = 14
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.fullatt_block_indexes = [8]
        self.gradient_checkpointing = False

    # copy from https://github.com/huggingface/transformers/blob/ed95493ce05688447d15d9a82d2d70695290ecff/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L454
    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens
    
    # def forward(
    #     self,
    #     hidden_states: Tensor,
    #     grid_thw: torch.Tensor,
    #     attention_mask: Tensor = None,
    #     context: Tensor = None,
    #     context_mask: Tensor = None,
    #     rotary_pos_emb: Tensor = None,
    #     rotary_pos_cos: Tensor = None,
    #     rotary_pos_sin: Tensor = None,
    #     attention_bias: Tensor = None,
    #     inference_params: InferenceParams = None,
    #     packed_seq_params: PackedSeqParams = None,
    #     sequence_len_offset: Tensor = None,        
    # ):
    #     window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    #     cu_window_seqlens = torch.tensor(
    #         cu_window_seqlens,
    #         device=hidden_states.device,
    #         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    #     )
    #     cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    #     # seq_len, _ = hidden_states.size()
    #     seq_len = hidden_states.size()[0]
    #     hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    #     hidden_states = hidden_states[window_index, :, :]
    #     hidden_states = hidden_states.reshape(seq_len, -1)
    #     rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    #     rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    #     rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    #     emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    #     position_embeddings = (emb.cos(), emb.sin())

    #     cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
    #         dim=0,
    #         # Select dtype based on the following factors:
    #         #  - FA2 requires that cu_seqlens_q must have dtype int32
    #         #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
    #         # See https://github.com/huggingface/transformers/pull/34852 for more information
    #         dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    #     )
    #     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
    #     for layer_num, blk in enumerate(self.layers):
    #         if layer_num in self.fullatt_block_indexes:
    #             cu_seqlens_now = cu_seqlens
    #         else:
    #             cu_seqlens_now = cu_window_seqlens
    #         if self.gradient_checkpointing and self.training:
    #             hidden_states = self._gradient_checkpointing_func(
    #                 blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
    #             )
    #         else:
    #             # hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
    #             # window attention
    #             # copied from https://github.com/huggingface/transformers/blob/ed95493ce05688447d15d9a82d2d70695290ecff/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L300
    #             xxx

    #     hidden_states = self.merger(hidden_states)
    #     reverse_indices = torch.argsort(window_index)
    #     hidden_states = hidden_states[reverse_indices, :]   
        
    #     return hidden_states
    
    def forward(
        self,
        hidden_states: Tensor,
        grid_thw: torch.Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,
    ):
        """
        Perform the forward pass through the transformer block.

        This method handles the core computation of the transformer, including
        self-attention, optional cross-attention, and feed-forward operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is the
                sequence length, b is the batch size, and h is the hidden size.
            attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
                self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask for cross-attention context
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor): Bias tensor for Q * K.T of shape in shape broadcastable
                to [b, num_head, sq, skv], e.g. [1, 1, sq, skv].
                Used as an alternative to apply attention mask for TE cuDNN attention.
            inference_params (InferenceParams, optional): Parameters for inference-time
                optimizations.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence
                processing.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: The output hidden states tensor of shape
            [s, b, h], and optionally the updated context tensor if cross-attention is used.
        """

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Update the inference parameters with the current batch size in case it is variable
        if inference_params and not self.training:
            inference_params.current_batch_size = hidden_states.size(1)

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            import transformer_engine  # To keep out TE dependency when not training in fp8

            if self.config.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif self.config.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

            fp8_recipe = TEDelayedScaling(
                config=self.config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not self.config.fp8_wgrad),
            )
            fp8_group = None
            if parallel_state.model_parallel_is_initialized():
                fp8_group = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True, tp_only_amax_red=self.tp_only_amax_red
                )
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            fp8_context = nullcontext()

        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # seq_len, _ = hidden_states.size()
        seq_len,bs,_ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, bs, -1)
        hidden_states = hidden_states[window_index, :, :, :]
        hidden_states = hidden_states.reshape(seq_len, bs, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, bs, 1, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, bs, 1, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        
        # seq_length = seq_len
        # attention_mask = torch.zeros([1, seq_length, seq_length], device=hidden_states.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True        

        with rng_context, fp8_context:
            # Forward pass.
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    with self.offload_context:
                        layer.use_cudagraph = True
                        if (len(self.cuda_graphs) == 0) or (not self.training):
                            # generate window attention mask
                            seq_length = seq_len
                            if l_no in self.fullatt_block_indexes:
                                cu_seqlens_now = cu_seqlens
                            else:
                                cu_seqlens_now = cu_window_seqlens                            
                            attention_mask = torch.zeros([1, seq_length, seq_length], device=hidden_states.device, dtype=torch.bool)
                            for i in range(1, len(cu_seqlens_now)):
                                attention_mask[..., cu_seqlens_now[i - 1] : cu_seqlens_now[i], cu_seqlens_now[i - 1] : cu_seqlens_now[i]] = True                              
                            hidden_states, context = layer(
                                hidden_states=hidden_states,
                                attention_mask=attention_mask,
                                context=context,
                                context_mask=context_mask,
                                rotary_pos_emb=rotary_pos_emb,
                                rotary_pos_cos=rotary_pos_cos,
                                rotary_pos_sin=rotary_pos_sin,
                                attention_bias=attention_bias,
                                inference_params=inference_params,
                                packed_seq_params=packed_seq_params,
                                sequence_len_offset=sequence_len_offset,
                            )
                        else:
                            # CUDA graph replay for layer `l_no` and microbatch
                            # `self.current_microbatch`. TransformerEngine versions>=1.10
                            # allow keyword arguments with CUDA graph. However, CUDA graph
                            # acccepts only Tensor inputs and Tensor outputs. Hence,
                            # `inference_params` and `packed_seq_params` are excluded from
                            # input list while output is limited to `hidden_states`.
                            cg_index = self.current_microbatch % len(self.cuda_graphs[l_no])
                            assert not any(
                                [inference_params, packed_seq_params]
                            ), "CUDA graph accepts only Tensor inputs."
                            optional_inputs = self.get_cuda_graph_optional_args(
                                attention_mask,
                                context,
                                context_mask,
                                rotary_pos_emb,
                                attention_bias,
                                inference_params,
                                packed_seq_params,
                            )
                            hidden_states = self.cuda_graphs[l_no][cg_index](
                                hidden_states, **optional_inputs
                            )

                    if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                    ):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        return hidden_states
        
    
class Qwen2VisionModel(VisionModule):
    """Qwen2 ViT vision model, adapted from CLIPViTModel to support Naive Dynamic Resolution.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        projection_config: TransformerConfig,
        projection_layer_spec: ModuleSpec,
        projection_type: str = "mlp",

        pre_process: bool = True, 
        post_process: bool = False
    ) -> None:
        super().__init__(config=transformer_config)

        self.spatial_merge_size = transformer_config.spatial_merge_size

        embed_dim = transformer_config.hidden_size
        num_heads = transformer_config.num_attention_heads
        temporal_patch_size = transformer_config.temporal_patch_size
        patch_size = transformer_config.patch_size
        in_channels = transformer_config.in_channels

        self.max_sequence_length = transformer_config.seq_length
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        head_dim = embed_dim // num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.model_type = ModelType.encoder_or_decoder
        self.pre_process = pre_process
        self.post_process = post_process

        # Transformer layers.
        # TODO: Follow-up changes will make pre and post_process configurable. They are needed for supporting pipeline parallelism.
        # NOTE: a final layer norm and/or linear layer present in some implementations are omitted here. 
        self.decoder = Qwen2_5_VLVisionBlock(
            config=transformer_config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            post_layer_norm=True
        )

        self.merge_hidden_size = projection_config.ffn_hidden_size
        self.square_merge_size = self.merge_hidden_size // embed_dim

        if self.post_process:
            self.projection = MultimodalProjector(
                projection_config,
                projection_layer_spec,
                projection_type,
                projection_config.ffn_hidden_size
            )
        else:
            self.projection = None
        
        self.input_tensor = None

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Sets input tensor to the model.

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        if self.pre_process: # always True
            self.input_tensor = input_tensor
        else:
            raise NotImplementedError()

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0).to(grid_thw.device)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size).to(grid_thw.device)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
      
    def forward(
        self, 
        vision_data: Optional[torch.Tensor], 
        grid_thw: torch.Tensor,
        inference_params: Optional[InferenceParams] = None,
        extra_block_kwargs: dict = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen2 Vision Model. This function passes the input tensors
        through the embedding layer and then the transformer.

        Args:
            x (torch.Tensor): input image/video data of shape [n_tokens, n_dims]
            grid_thw (torch.Tensor): the size tensor indicates grid size of each image/frame
            packed_seq_params (PackedSeqParams): parameters to build attention mask in the backend

        Returns:
            x (torch.Tensor): output after final transformer block of shape [b, s, h].
        """
        # NOTE: each epp stage should have thw_grids to build PackedSedParams
        assert grid_thw is not None
        if self.input_tensor is not None:
            vision_data = self.input_tensor[:, None]
            self.input_tensor = None
        # otherwise, either vision_data is not None or in EPP intermediate stage
        if inference_params is not None:
            raise NotImplementedError()

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.pre_process:
            vision_data = self.patch_embed(vision_data)[:, None]
            rotary_pos_emb = self.rot_pos_emb(grid_thw)[:, None, None, :].repeat(1, 1, 1, 2)

        # window attention

        hidden_states = self.decoder(
            hidden_states = vision_data, 
            grid_thw = grid_thw,
            attention_mask = None,
            inference_params = inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=self.build_packed_seq_params(grid_thw),
            **(extra_block_kwargs or {}),
        )

        if not self.post_process:
            return hidden_states
        
        return self.projection(hidden_states.view(-1, self.merge_hidden_size))

    def build_packed_seq_params(self, grid_thw: torch.Tensor) -> PackedSeqParams:
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).int()
        return PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            qkv_format='thd'
        )