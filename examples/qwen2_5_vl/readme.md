# Qwen2.5-vl

## task

- [ ] Features
    - [x] window attention
      - [ ] test window attention performance
    - [x] merger
      - [x] Same as qwen2-vl, MultimodalProjector Module.
    - [x] m-rope
    - [x] other modules, RMSNorm, SwiGLU.
      - [x] In vit modules, switch to swiglu by modifying [transformer_config.py](../../megatron_patch/model/qwen2_5_vl/transformer_config.py)
      - [x] In vit modules, switch to RMSNorm by modifying [transformer_config.py](../../megatron_patch/model/qwen2_5_vl/transformer_config.py)
      - [x] decoder layer, switch to swiglu by modifying [transformer_config.py](../../megatron_patch/model/qwen2_5_vl/transformer_config.py) by modifying [transformer_config.py](../../megatron_patch/model/qwen2_5_vl/transformer_config.py)
      - [x] decoder layer, switch to RMSNorm
    - [x] add new parameters in configuration file for qwen2.5-vl
    - [x] load params from configuration file
- [x] examples - pretrain qwen2.5 vl
  - [x] `examples/qwen2_vl/pretrain_qwen_2_5_vl.py`
  - [x] pretrain script: [run_mcore_qwen_h800.sh](./run_mcore_qwen_h800.sh)
  - [x] invoke script: [run_mcore_qwen_h800_args.sh](./run_mcore_qwen_h800_args.sh)
- [ ] readme for pretrain
- [x] convergence verification for no parallelism
  - [x] loss curve can be aligned: [qwen2.5-vl-convergence](https://wandb.ai/searobbersandduck/qwen2.5-vl-convergence?nw=nwusersearobbersandduck)
