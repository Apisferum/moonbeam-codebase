# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import json
import dataclasses
import fire
import random
from transformers.data.data_collator import torch_default_data_collator
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM_Conditional_Generation,
    LlamaConfig,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import ddp_config as DDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.data.concatenator import ConcatDataset, ConcatDataset_dummy_padding
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.model_checkpointing import load_model_checkpoint_ddp

from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.fsdp_utils import hsdp_device_mesh
from llama_recipes.utils.train_utils import (
    train_con_gen,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from accelerate.utils import is_xpu_available

# --- SIMD & RUST JSON OPTIMIZATION ---
import orjson
try:
    import simdjson
    _SIMD_PARSER = simdjson.Parser()
except ImportError:
    _SIMD_PARSER = None # Fallback to orjson if pysimdjson fails to install
# -------------------------------------

def setup_wandb(train_config, fsdp_config, llama_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    # 🚀 FIX: Allow W&B to update the config when resuming with a new num_epochs
    run.config.update(train_config, allow_val_change=True)
    run.config.update(fsdp_config, allow_val_change=True)

    # Convert the llama_config to a dictionary and then to a JSON string
    config_dict = llama_config.to_dict()
    
    # CRITICAL FIX: Revert to standard json for the config file.
    # HuggingFace configs sometimes contain non-string keys (like integers/tuples).
    # orjson strictly rejects them, but standard json auto-casts them to strings.
    # Since this only runs ONCE at startup, the SIMD speedup is irrelevant here.
    config_json = json.dumps(config_dict, indent=4, default=str)
    
    # Get the wandb run directory
    from pathlib import Path
    folder_name = (train_config.dist_checkpoint_root_folder+ "/"+ train_config.dist_checkpoint_folder+ "-"+ train_config.model_name)
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)
    config_file_path = os.path.join(save_dir, 'llama_config.json')

    # Write the JSON string to the file (Text mode 'w' instead of binary 'wb')
    with open(config_file_path, 'w') as f:
        f.write(config_json)
        print(f"config file saved to {config_file_path}!")
        
    return run

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, ddp_config = TRAIN_CONFIG(), FSDP_CONFIG(), DDP_CONFIG()
    model_config_path = "src/llama_recipes/configs/model_config_multi_task.json"
    update_config((train_config, fsdp_config, ddp_config), **kwargs)
    print("updated training config", train_config)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or train_config.enable_ddp: 
        setup() #enable nccl / ccl
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained( 
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        else:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            llama_config.use_cache = use_cache
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)

    else: #DDP and non-distributed training
        llama_config = LlamaConfig.from_pretrained(model_config_path)
        llama_config.use_cache = use_cache
        print(f"model_config:{llama_config}")
        model = LlamaForCausalLM_Conditional_Generation(llama_config) 

        model_checkpoint = torch.load(train_config.trained_checkpoint_path, map_location="cpu", weights_only=False)
        
        checkpoint = model_checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'): 
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        # Load the state_dict into the model, ignoring unmatched keys
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"when loading checkpoint, encounter missing keys: {missing_keys}; unexpected_keys:{unexpected_keys}")

    # 🚀 DUAL-RUN MODE: Let BOTH GPUs initialize their own W&B run
    wandb_run = None
    if train_config.use_wandb:
        wandb_run = setup_wandb(train_config, fsdp_config, llama_config, **kwargs)


    # Load the tokenizer and add special tokens
    tokenizer = MusicTokenizer(timeshift_vocab_size = llama_config.onset_vocab_size, dur_vocab_size = llama_config.dur_vocab_size, octave_vocab_size = llama_config.octave_vocab_size, pitch_class_vocab_size = llama_config.pitch_class_vocab_size, instrument_vocab_size = llama_config.instrument_vocab_size, velocity_vocab_size = llama_config.velocity_vocab_size, sos_token = llama_config.sos_token, eos_token = llama_config.eos_token, pad_token = llama_config.pad_token)

    dataset_config = generate_dataset_config(train_config, kwargs)
    
    # Safely get the dictionary path from either standard or multi-task config
    dict_path = getattr(dataset_config, 'additional_token_dict_path', None) or getattr(dataset_config, 'commu_additional_token_dict_path', None)
    
    if dict_path:
        # Read as raw bytes for zero-copy SIMD parsing
        with open(dict_path, "rb") as f:
            raw_bytes = f.read()
            if _SIMD_PARSER is not None:
                additional_token_dict = _SIMD_PARSER.parse(raw_bytes).as_dict()
            else:
                additional_token_dict = orjson.loads(raw_bytes)
                
        for key, value in additional_token_dict.items():
            tokenizer.add_new_tokens(token_name = key, token_val = value)
            print(f"added {key} to tokenizer")

    print_model_size(model, train_config, rank if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # --- FIX FOR PEFT & GRADIENT CHECKPOINTING ---
    import types
    
    def custom_enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        
        if hasattr(self.model, 'embed_tokens') and isinstance(self.model.embed_tokens, torch.nn.Module):
            self.model.embed_tokens.register_forward_hook(make_inputs_require_grads)
        elif hasattr(self.model, 'layers'):
            self.model.layers[0].register_forward_hook(make_inputs_require_grads)
            
    model.enable_input_require_grads = types.MethodType(custom_enable_input_require_grads, model)
    # ---------------------------------------------

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.enable_ddp and ddp_config.pure_bf16:
        model.to(torch.bfloat16)

    if train_config.use_peft:
        from peft import LoraConfig
        
        # --- LAYER-WISE ROUTING: CoMMU gets Lower Layers (0-7), EMOPIA/SLakh get Upper Layers (8-14) ---
        commu_target_modules = []
        for i in range(0, 8):  
            commu_target_modules.extend([
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj",
            ])
        
        style_target_modules = []
        for i in range(8, 15):  
            style_target_modules.extend([
                f"model.layers.{i}.self_attn.q_proj",
                f"model.layers.{i}.self_attn.k_proj",
                f"model.layers.{i}.self_attn.v_proj",
                f"model.layers.{i}.self_attn.o_proj",
            ])
        
        # 🚀 UPGRADED LORA RANK: Doubled capacity for deeper emotional/orchestral learning
        commu_config = LoraConfig(r=32, lora_alpha=64, target_modules=commu_target_modules)
        style_config = LoraConfig(r=32, lora_alpha=64, target_modules=style_target_modules)
        
        model = get_peft_model(model, commu_config, adapter_name="commu_lora")
        model.add_adapter("emopia_lora", style_config)
        model.add_adapter("slakh_lora", style_config)
        model.set_adapter("commu_lora")  
        
        model.print_trainable_parameters()
        # -------------------------------------------------------------------------------------------------
        
    hsdp_device_mesh = None 
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer) 

        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model) 
    elif train_config.enable_ddp: #wrap ddp code
        mixed_precision_policy, wrapping_policy = get_policies(ddp_config, rank)
        model.to(local_rank)
        model = DDP(model,
                    mixed_precision=mixed_precision_policy if not ddp_config.pure_bf16 else None, 
                    device_mesh=hsdp_device_mesh,
                    device_ids=[local_rank],
                    find_unused_parameters=True,
                    )
    elif not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to(torch.bfloat16).to("cuda:0")


    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = None
    if train_config.run_validation:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
        
    # Disable packing for multi-task to prevent mixing tasks in a single sequence
    if train_config.batching_strategy == "packing" and train_config.dataset != "multi_task_dataset":
        dataset_train = ConcatDataset_dummy_padding(dataset_train, chunk_size=train_config.context_length, split="train", data_dir=getattr(dataset_config, 'data_dir', ''))

    # --- CREATE DATALOADERS (OPTIMIZED MULTI-TASK BYPASS) ---
    num_workers = getattr(train_config, 'num_workers_dataloader', 0)
    worker_kwargs = {"num_workers": num_workers}
    if num_workers > 0:
        worker_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})

    if train_config.dataset == "multi_task_dataset":
        from llama_recipes.datasets.multi_task_dataset import DDPSyncedTaskSampler
        from transformers.data.data_collator import torch_default_data_collator
        
        # Train Loader
        train_sampler = DDPSyncedTaskSampler(dataset_train, batch_size=train_config.batch_size_training)
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=train_sampler, pin_memory=True,
            collate_fn=torch_default_data_collator, **worker_kwargs
        )
        
        # Eval Loader
        eval_dataloader = None
        if train_config.run_validation and dataset_val is not None:
            val_sampler = DDPSyncedTaskSampler(dataset_val, batch_size=train_config.val_batch_size, shuffle=False)
            eval_dataloader = torch.utils.data.DataLoader(
                dataset_val, batch_sampler=val_sampler, pin_memory=True,
                collate_fn=torch_default_data_collator, **worker_kwargs
            )
    else:
        # Fallback for standard single-task datasets
        train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
        train_dl_kwargs.pop('collate_fn', None)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, pin_memory=True, collate_fn=torch_default_data_collator, **worker_kwargs, **train_dl_kwargs)
        
        eval_dataloader = None
        if train_config.run_validation and dataset_val is not None:
            val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
            val_dl_kwargs.pop('collate_fn', None)
            eval_dataloader = torch.utils.data.DataLoader(dataset_val, pin_memory=True, collate_fn=torch_default_data_collator, **worker_kwargs, **val_dl_kwargs)

    starting_epoch, starting_step = 0, 0
    print("check model trainable parameters")
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name} | Shape: {param.shape} | Parameters: {param.numel()}")
            total_trainable += param.numel()
        else:
            print(f"Frozen: {name} | Shape: {param.shape} | Parameters: {param.numel()}")
    print(f"\nTotal Trainable Parameters: {total_trainable}")

    # --- AUTO-RESUME LOGIC (TRUE CHECKPOINTING) ---
    output_dir = train_config.output_dir
    _resume_optimizer_state = None
    _resume_scheduler_state = None
    
    if os.path.exists(output_dir):
        saved_epochs = [d for d in os.listdir(output_dir) if d.startswith("epoch_") and "_step_" not in d]
        if saved_epochs:
            # Find the latest saved epoch folder
            latest_epoch_dir = os.path.join(output_dir, max(saved_epochs, key=lambda x: int(x.split('_')[1])))
            state_path = os.path.join(latest_epoch_dir, "training_state.pt")
            
            if os.path.exists(state_path):
                print(f"\n--> Found True Checkpoint! Resuming from {latest_epoch_dir}...")
                
                # 1. Load the trained LoRA weights back into the 3 adapters
                if train_config.use_peft:
                    # 🚨 CRITICAL: Unwrap DDP to access PEFT methods
                    unwrapped_model = model.module if hasattr(model, "module") else model
                    
                    for adapter_name in ["commu_lora", "emopia_lora", "slakh_lora"]:
                        # 🚨 FIX: Handle PEFT's nested folder structure
                        nested_dir = os.path.join(latest_epoch_dir, adapter_name, adapter_name)
                        flat_dir = os.path.join(latest_epoch_dir, adapter_name)
                        
                        if os.path.exists(nested_dir):
                            adapter_dir = nested_dir
                        elif os.path.exists(flat_dir):
                            adapter_dir = flat_dir
                        else:
                            print(f"⚠️ Warning: Could not find adapter {adapter_name}")
                            continue
                            
                        unwrapped_model.delete_adapter(adapter_name) 
                        unwrapped_model.load_adapter(adapter_dir, adapter_name) 
                        print(f"✅ Successfully loaded {adapter_name} from {adapter_dir}")
                    
                    unwrapped_model.set_adapter("commu_lora") 
                
                # 2. Load Epoch counter and stash Optimizer/Scheduler states for later
                resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
                starting_epoch = resume_state['epoch'] + 1 
                _resume_optimizer_state = resume_state['optimizer_state_dict']
                _resume_scheduler_state = resume_state['scheduler_state_dict']
                
                print(f"--> Adapters loaded! Will initialize optimizer and resume from Epoch {starting_epoch}.")
    # ----------------------------------------------

    # 🚀 CRITICAL FIX: Initialize Optimizer and Scheduler AFTER adapters are fully loaded!
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Now safely load the optimizer and scheduler states if resuming
    if _resume_optimizer_state is not None:
        optimizer.load_state_dict(_resume_optimizer_state)
        scheduler.load_state_dict(_resume_scheduler_state)
        print("✅ Optimizer and Scheduler states loaded successfully.")

    # Start the training process
    results = train_con_gen( 
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        starting_epoch,
        starting_step, 
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        ddp_config if train_config.enable_ddp else None,
        local_rank if (train_config.enable_fsdp or train_config.enable_ddp) else None, 
        rank if (train_config.enable_fsdp or train_config.enable_ddp) else None,
        wandb_run,
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb and wandb_run is not None:
            for k,v in results.items():
                wandb_run.summary[k] = v

if __name__ == "__main__":
    fire.Fire(main)