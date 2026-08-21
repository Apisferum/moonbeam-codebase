# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
import packaging
from datetime import datetime
import contextlib


import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import math


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_model_checkpoint_ddp, save_peft_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_xccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, starting_epoch, starting_step,gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(starting_epoch, train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    if step < starting_step and epoch == starting_epoch:  #skip until the starting step in the first continuing epoch
                        continue
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    # --- MULTI-TASK LORA ROUTING ---
                    if "task_id" in batch:
                        task_ids = batch.pop("task_id") 
                        current_task = task_ids[0].item() if hasattr(task_ids[0], "item") else task_ids[0]
                        
                        unwrapped_model = model.module if hasattr(model, "module") else model
                        if hasattr(unwrapped_model, "set_adapter"):
                            if current_task == 0: unwrapped_model.set_adapter("commu_lora")
                            elif current_task == 1: unwrapped_model.set_adapter("emopia_lora")
                            elif current_task == 2: unwrapped_model.set_adapter("slakh_lora")

                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0 and train_config.run_validation:
                        
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                model.save_pretrained(train_config.output_dir)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))     

                        """IMPORTANT"""         
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp)):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def train_overfit(model, batch, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        eval_dataloader: same as train_dataloader
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:

                for step, batch_unused in enumerate(train_dataloader):
                    # print("batch train: ", batch['input_ids'])
                    """
                    save data as npy file for visualization
                    """

                    # Save data as npy files for the first few steps for visualization
                    if step < 5:
                        import numpy as np
                        for key in batch.keys():
                            # Convert the tensor to a NumPy array (move to CPU if needed)
                            data_array = batch[key].cpu().numpy()
                            
                            # Save the NumPy array to a file with a unique name per key and step
                            np.save(f'/data/home/acw753/musicllama/dataset_analysis/{key}_step_{step}.npy', data_array)

                    if step > 1000:
                        break

                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:

                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step%train_config.validation_interval==0 and train_config.run_validation:
                        
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, generation_logits, generation_hidden_state, logits_shrinked = evaluation_overfit(model, train_config, batch, eval_dataloader, local_rank, tokenizer, wandb_run)

                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                model.save_pretrained(train_config.output_dir)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        torch.save(generation_logits, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_logits_epoch_{epoch}_step_{step}.pt')
                                        torch.save(generation_hidden_state, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_hidden_state_epoch_{epoch}_step_{step}.pt')
                                        torch.save(logits_shrinked, f'/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/logits_shrinked_epoch_{epoch}_step_{step}.pt')
                                        print(f"generation logits and hidden states saved to /data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/generation_logits_epoch_{epoch}_step_{step}.pt")
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))     

                        """IMPORTANT"""         
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp)):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def train_con_gen(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, starting_epoch, starting_step,gradient_accumulation_steps, train_config, fsdp_config=None, ddp_config=None, local_rank=None, rank=None, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])



    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []
        val_task_losses_history = [] # <--- ADDED FOR PER-TASK PLOTTING
        train_task_history = []      # <--- ADDED FOR JSON TRACKING
        val_task_ppls_history = []   # <--- ADDED FOR JSON TRACKING

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(starting_epoch, train_config.num_epochs):
        # 🚀 CRITICAL: Tell the sampler what epoch we are on for proper shuffling
        if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
            train_dataloader.batch_sampler.set_epoch(epoch)
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    if step < starting_step and epoch == starting_epoch:  #skip until the starting step in the first continuing epoch
                        continue
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    current_task = -1 # <--- SAFETY NET: Prevents UnboundLocalError if task_id is missing
                    # --- MULTI-TASK LORA ROUTING & TASK_ID REMOVAL ---
                    if "task_id" in batch:
                        task_ids = batch.pop("task_id") 
                        current_task = task_ids[0].item() if hasattr(task_ids[0], "item") else task_ids[0]
                        
                        # Unwrap DDP to access PEFT methods
                        unwrapped_model = model.module if hasattr(model, "module") else model
                        if hasattr(unwrapped_model, "set_adapter"):
                            if current_task == 0:
                                unwrapped_model.set_adapter("commu_lora")
                            elif current_task == 1:
                                unwrapped_model.set_adapter("emopia_lora")
                            elif current_task == 2:
                                unwrapped_model.set_adapter("slakh_lora")

                    # --- REMOVE EMPTY ATTENTION MASK ---
                    if "attention_mask" in batch:
                        mask = batch["attention_mask"]
                        if (isinstance(mask, torch.Tensor) and mask.numel() == 0) or \
                           (isinstance(mask, list) and len(mask) == 0):
                            batch.pop("attention_mask")
                    
                    # --- BRUTE-FORCE GLOBAL INJECTION OF ADDITIONAL TOKEN MAP ---
                    if not hasattr(model, "_cached_token_map"):
                        import json
                        try:
                            with open("src/llama_recipes/configs/model_config_multi_task.json", "r") as f:
                                cfg = json.load(f)
                            # Map negative token IDs to their 0-based index in the embedding matrix
                            model._cached_token_map = {int(k): idx for idx, k in enumerate(cfg["metadata_tokens"])}
                        except Exception as e:
                            print(f"WARNING: Failed to load token map: {e}")
                            model._cached_token_map = {}
                            
                    if "additional_token_map" in batch:
                        batch.pop("additional_token_map")
                        
                    # Brute-force inject into EVERY loaded module that has this global variable
                    import sys
                    for name, mod in list(sys.modules.items()):
                        if hasattr(mod, 'additional_token_map'):
                            mod.additional_token_map = model._cached_token_map
                    # ----------------------------------------------------------------

                    for key in list(batch.keys()):
                        # Skip non-tensor items
                        if not isinstance(batch[key], torch.Tensor):
                            continue
                            
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            else:
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        outputs = model(**batch)
                        loss = outputs.loss

                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item()) # TRUE LOSS BEFORE DIVIDING
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                        
                        # 🚀 JSON TASK TRACKING (Sparse Step-Log)
                        step_log = {"step": step}
                        current_loss = loss.detach().float().item()
                        current_ppl = float(torch.exp(loss.detach().float()))
                        if current_task == 0: step_log["commu_loss"] = current_loss; step_log["commu_ppl"] = current_ppl
                        elif current_task == 1: step_log["emopia_loss"] = current_loss; step_log["emopia_ppl"] = current_ppl
                        elif current_task == 2: step_log["slakh_loss"] = current_loss; step_log["slakh_ppl"] = current_ppl
                        train_task_history.append(step_log)
                        
                    # --- 2. NOW DIVIDE FOR GRADIENT ACCUMULATION ---
                    loss = loss / gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    
                    if train_config.use_fp16:
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            # 🚀 CRITICAL: Must unscale BEFORE measuring, otherwise norm is artificially massive
                            scaler.unscale_(optimizer)
                            
                            # 🛡️ NATIVE CLIPPING & RAW NORM CAPTURE
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    total_norm = model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            else:
                                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

                            total_norm = total_norm.item() if hasattr(total_norm, 'item') else float(total_norm)

                            # 📊 LOG RAW EXPLOSIONS TO CONSOLE & W&B
                            if rank == 0 and step % 50 == 0:
                                print(f"\n🔍 Step {step} | Raw Gradient Norm (FP16): {total_norm:.4f}", flush=True)
                                if wandb_run:
                                    wandb_run.log({'train/raw_grad_norm': total_norm, 'train/step': epoch * len(train_dataloader) + step})
                                if train_config.gradient_clipping and total_norm > train_config.gradient_clipping_threshold:
                                    print(f"   ✂️ Safety Net Triggered! Clipped {total_norm:.2f} -> {train_config.gradient_clipping_threshold}", flush=True)

                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            
                            # 🛡️ NATIVE CLIPPING & RAW NORM CAPTURE
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    total_norm = model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            else:
                                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

                            total_norm = total_norm.item() if hasattr(total_norm, 'item') else float(total_norm)

                            # 📊 LOG RAW EXPLOSIONS TO CONSOLE & W&B
                            if rank == 0 and step % 50 == 0:
                                print(f"\n🔍 Step {step} | Raw Gradient Norm: {total_norm:.4f}", flush=True)
                                if wandb_run:
                                    wandb_run.log({'train/raw_grad_norm': total_norm, 'train/step': epoch * len(train_dataloader) + step})
                                if train_config.gradient_clipping and total_norm > train_config.gradient_clipping_threshold:
                                    print(f"   ✂️ Safety Net Triggered! Clipped {total_norm:.2f} -> {train_config.gradient_clipping_threshold}", flush=True)

                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        # 🚀 DUAL-RUN TELEMETRY: Each GPU logs to its own W&B run
                        current_loss = loss.detach().float().item()
                        current_ppl = math.exp(current_loss)
                        
                        log_dict = {
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': current_loss,
                            'train/ppl': current_ppl
                        }
                        
                        if current_task == 0: 
                            log_dict['train/loss_commu'] = current_loss
                            log_dict['train/ppl_commu'] = current_ppl
                        elif current_task == 1: 
                            log_dict['train/loss_emopia'] = current_loss
                            log_dict['train/ppl_emopia'] = current_ppl
                        elif current_task == 2: 
                            log_dict['train/loss_slakh'] = current_loss
                            log_dict['train/ppl_slakh'] = current_ppl
                            
                        wandb_run.log(log_dict)

                    task_map = {0: "CoMMU", 1: "EMOPIA", 2: "SLakh", -1: "Unknown"}
                    t_name = task_map.get(current_task, "?")
                    pbar.set_description(f"Epoch {epoch} | Step {step}/{len(train_dataloader)} | 🎵 {t_name} | Loss: {loss.detach().float():.4f}")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep, val_task_losses=val_task_losses_history, train_task_history=train_task_history, val_task_ppls=val_task_ppls_history)
                
                
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    # --- MID-EPOCH WEIGHT CHECKPOINT (EVERY 2000 STEPS) ---
                    if step > 0 and step % 327 == 0:
                        unwrapped_model = model.module if hasattr(model, "module") else model
                        
                        # ALL ranks must switch adapters together to prevent DDP desync
                        for adapter_name in ["commu_lora", "emopia_lora", "slakh_lora"]:
                            unwrapped_model.set_adapter(adapter_name)
                            
                            if dist.is_initialized():
                                dist.barrier()  # Ensure all ranks have switched before Rank 0 writes
                                
                            if rank == 0:
                                step_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}_step_{step}")
                                os.makedirs(step_dir, exist_ok=True)
                                unwrapped_model.save_pretrained(os.path.join(step_dir, adapter_name))
                                
                            if dist.is_initialized():
                                dist.barrier()  # Ensure Rank 0 finishes writing before ranks switch to the next adapter
                        
                        # ALL ranks reset to default together
                        unwrapped_model.set_adapter("commu_lora")
                        if rank == 0:
                            print(f"\n💾 Mid-Epoch Checkpoint saved at step {step}!")
                    # -------------------------------------------------------

                    #TODO: More frequent evaluation; Remember to switch on model.train again
                    if step > 0 and step%train_config.validation_interval==0 and train_config.run_validation:
                        # --- FIX TQDM GHOST TEXT ---
                        pbar.clear()
                        print("\r" + " " * 100 + "\r", end="")
                        # ---------------------------
                        
                        # Unpack all 8 items returned by the new evaluation function
                        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, temp_val_acc, eval_acc, task_avg_losses, task_avg_ppls = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                        
                        if train_config.save_metrics:
                            val_step_loss.extend(temp_val_loss)
                            val_step_perplexity.extend(temp_step_perplexity)
                            val_task_losses_history.append(task_avg_losses)
                            val_task_ppls_history.append(task_avg_ppls) # <--- ADDED
                            
                        pbar.refresh() # Redraw training bar after eval

                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model:
                            if train_config.enable_fsdp:
                                dist.barrier()
                            if train_config.use_peft:
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"we are about to save the PEFT modules")
                                else:
                                    print(f"we are about to save the PEFT modules")
                                # model.save_pretrained(train_config.output_dir)
                                save_peft_checkpoint(model, train_config.output_dir, epoch=epoch, step = step)
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                                else:
                                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

                            else: #since we are training a smaller model, we are not using FDSP and PEFT
                                if train_config.enable_fsdp:
                                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:

                                        save_model_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                        print("=====================================================")

                                        save_model_and_optimizer_sharded(model, rank, train_config)
                                        if train_config.save_optimizer:
                                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                            print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                            print("=====================================================")

                                    if not train_config.use_peft and  train_config.save_optimizer:
                                        save_optimizer_checkpoint(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                        print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                elif train_config.enable_ddp: 
                                    if not train_config.use_peft:
                                        save_model_checkpoint_ddp(
                                            model, optimizer, rank, train_config, epoch=epoch, step=step
                                        )
                                        print(" Saving the DDP model checkpoints and optimizer using FULL_STATE_DICT")
                                        print("=====================================================")
                                    else:
                                        print("Warning! Model Checkpoints are not saved properly")
                                        print("=====================================================")
                            if train_config.enable_fsdp:
                                dist.barrier()
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch} is {best_val_loss}")
                        val_loss.append(float(best_val_loss))
                        val_prep.append(float(eval_ppl))    

                        #IMPORTANT        
                        model.train()
                
                
                
                
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp)):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            
        train_epoch_loss = total_loss / len(train_dataloader)
        # 🚀 FIX: Divide by world_size for BOTH FSDP and DDP
        if train_config.enable_fsdp or train_config.enable_ddp: 
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss)) # Appends average train loss for this epoch

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # --- END-OF-EPOCH FULL EVALUATION (The Pragmatic Strategy) ---
        if train_config.run_validation:
            print(f"\n--- Running End-of-Epoch Full Evaluation (Epoch {epoch}) ---")
            eval_ppl, eval_epoch_loss, _, _, _, _, task_avg_losses, task_avg_ppls = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            
            # Update Best Loss & Checkpoint
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.save_model:
                    if train_config.enable_fsdp: dist.barrier()
                    if train_config.use_peft:
                        save_peft_checkpoint(model, train_config.output_dir, epoch=epoch, step=step)
                    if train_config.enable_fsdp: dist.barrier()
            
            # CRITICAL: Append the CURRENT epoch's validation loss (not just the best)
            # This allows the plot to show divergence (overfitting) if it happens.
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
            val_task_losses_history.append(task_avg_losses)
            val_task_ppls_history.append(task_avg_ppls) # <--- ADDED

        # Update the learning rate as needed
        lr_scheduler.step()

        # 🚀 LOG EPOCH AVERAGES TO W&B
        if wandb_run:
            wandb_run.log({
                'epoch/train_loss': float(train_epoch_loss),
                'epoch/train_ppl': float(train_perplexity),
                'epoch/number': epoch + 1
            })

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                print(f"Epoch {epoch}: train_perplexity={train_perplexity:.6f}, train_epoch_loss={train_epoch_loss:.6f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch}: train_perplexity={train_perplexity:.6f}, train_epoch_loss={train_epoch_loss:.6f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, 
                         val_step_loss, val_loss, val_step_perplexity, val_prep, 
                         val_task_losses=val_task_losses_history, 
                         train_task_history=train_task_history, 
                         val_task_ppls=val_task_ppls_history)
        # --- EPOCH-LEVEL TRUE CHECKPOINTING (PAUSE/RESUME) ---
        unwrapped_model = model.module if hasattr(model, "module") else model
        
        # ALL ranks must switch adapters together to prevent DDP desync
        for adapter_name in ["commu_lora", "emopia_lora", "slakh_lora"]:
            unwrapped_model.set_adapter(adapter_name)
            if dist.is_initialized(): dist.barrier()
            
            if not train_config.enable_fsdp or rank == 0:
                epoch_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                unwrapped_model.save_pretrained(os.path.join(epoch_dir, adapter_name))
                
            if dist.is_initialized(): dist.barrier()
            
        # ALL ranks reset to default together
        unwrapped_model.set_adapter("commu_lora")
        
        if not train_config.enable_fsdp or rank == 0:
            epoch_dir = os.path.join(train_config.output_dir, f"epoch_{epoch}")
            # 2. Save Optimizer, Scheduler, and Epoch counter
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, os.path.join(epoch_dir, "training_state.pt"))
            print(f"\n--> True Checkpoint for Epoch {epoch} saved to {epoch_dir}!")
        # ------------------------------------------------------

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if (train_config.enable_fsdp or train_config.enable_ddp) and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results


import os
import json

# --- HIGH-PERFORMANCE JSON SETUP ---
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

try:
    import simdjson
    USE_SIMDJSON = True
except ImportError:
    USE_SIMDJSON = False

# Helper function for reading (if you ever need to read JSON inside train_utils.py)
def load_json_fast(json_file_path):
    if USE_SIMDJSON:
        parser = simdjson.Parser()
        with open(json_file_path, 'rb') as f:
            proxy = parser.parse(f.read())
            return {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v for k, v in proxy.items()}
    else:
        with open(json_file_path, 'r') as f:
            return json.load(f)
# -----------------------------------

def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1
        
    model.eval()
    val_step_loss = []
    eval_loss = 0.0
    total_eval_steps = 0
    
    # --- PER-TASK TRACKING ---
    task_losses = {0: 0.0, 1: 0.0, 2: 0.0}
    task_counts = {0: 0, 1: 0, 2: 0}
    task_names = {0: "CoMMU (Structure)", 1: "EMOPIA (Emotion)", 2: "SLakh (Orchestration)"}
    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation.")
                break
                
            # --- MULTI-TASK EVALUATION ROUTING ---
            t_id = -1 
            if "task_id" in batch:
                task_id_batch = batch.pop("task_id")
                t_id = task_id_batch[0].item() if hasattr(task_id_batch[0], 'item') else task_id_batch[0]
                
                unwrapped_model = model.module if hasattr(model, "module") else model
                if hasattr(unwrapped_model, "set_adapter"):
                    if t_id == 0: unwrapped_model.set_adapter("commu_lora")
                    elif t_id == 1: unwrapped_model.set_adapter("emopia_lora")
                    elif t_id == 2: unwrapped_model.set_adapter("slakh_lora")
            
            if "attention_mask" in batch:
                mask = batch["attention_mask"]
                if (isinstance(mask, torch.Tensor) and mask.numel() == 0) or (isinstance(mask, list) and len(mask) == 0):
                    batch.pop("attention_mask")
                    
            if "additional_token_map" in batch:
                batch.pop("additional_token_map")

            for key in list(batch.keys()):
                if not isinstance(batch[key], torch.Tensor): continue
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    batch[key] = batch[key].to('xpu:0' if is_xpu_available() else 'cuda:0')
                    
            with torch.no_grad():
                outputs = model(**batch)
                
                # 🚀 PURE NATIVE LOSS
                loss = outputs.loss.float()

                # --- TRACK PER-TASK LOSS ---
                if t_id in task_losses:
                    task_losses[t_id] += loss.detach().float().item()
                    task_counts[t_id] += 1

                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())

                eval_loss += loss.detach().float()

    # 🚀 CRITICAL DDP SYNCHRONIZATION FIX (BULLETPROOF) 🚀
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert eval_loss to a standard Python float to avoid 0-d tensor DDP quirks
    local_eval_loss = eval_loss.item() if isinstance(eval_loss, torch.Tensor) else float(eval_loss)
    local_steps = float(total_eval_steps)
    
    if train_config.enable_fsdp or train_config.enable_ddp:
        # 1. Sync global eval loss and steps
        eval_loss_tensor = torch.tensor([local_eval_loss], device=device)
        steps_tensor = torch.tensor([local_steps], device=device)
        
        dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.SUM)
        
        eval_epoch_loss = (eval_loss_tensor / steps_tensor).item() if steps_tensor.item() > 0 else 0.0
        eval_ppl = math.exp(eval_epoch_loss)
        
        # 2. Sync per-task losses and counts across GPUs
        task_losses_tensor = torch.tensor([task_losses[0], task_losses[1], task_losses[2]], device=device)
        task_counts_tensor = torch.tensor([task_counts[0], task_counts[1], task_counts[2]], device=device, dtype=torch.float32)
        
        dist.all_reduce(task_losses_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(task_counts_tensor, op=dist.ReduceOp.SUM)
        
        for i in range(3):
            task_losses[i] = task_losses_tensor[i].item()
            task_counts[i] = int(task_counts_tensor[i].item())
    else:
        eval_epoch_loss = (eval_loss / total_eval_steps).item() if total_eval_steps > 0 else 0.0
        eval_ppl = math.exp(eval_epoch_loss)

    # --- PRINT HIGH-PRECISION METRICS (6 Decimals to kill the rounding illusion) ---
    if not train_config.enable_fsdp or local_rank == 0:
        print(f"\n🎯 EVALUATION RESULTS (Global Epoch Avg):")
        print(f"   -> Total Perplexity (PPL): {eval_ppl:.6f}")
        print(f"   -> Total Loss: {eval_epoch_loss:.6f}")
        
        print(f"\n--- Per-Task Validation Metrics (Global) ---")
        for t_id, name in task_names.items():
            if task_counts[t_id] > 0:
                avg_loss = task_losses[t_id] / task_counts[t_id]
                avg_ppl = math.exp(avg_loss)
                print(f"   {name}:")
                print(f"      Loss : {avg_loss:.6f}")
                print(f"      PPL  : {avg_ppl:.6f}")
            else:
                print(f"   {name}: (No data)")
        print(f"-----------------------------------\n")

    if wandb_run:
        log_dict = {
            'eval/perplexity': float(eval_ppl),
            'eval/loss': float(eval_epoch_loss),
        }
        # 🚀 THESIS GOLD: Log Per-Task Eval Loss & Perplexity to W&B
        for t_id, name in task_names.items():
            if task_counts[t_id] > 0:
                clean_name = name.split(" ")[0].lower() # Extracts "commu", "emopia", or "slakh"
                avg_loss = task_losses[t_id] / task_counts[t_id]
                log_dict[f'eval/loss_{clean_name}'] = float(avg_loss)
                log_dict[f'eval/ppl_{clean_name}'] = float(math.exp(avg_loss))
                
        wandb_run.log(log_dict, commit=False)

    task_avg_losses = {}
    task_avg_ppls = {} 
    for t_id, name in task_names.items():
        if task_counts[t_id] > 0:
            avg_loss = task_losses[t_id] / task_counts[t_id]
            task_avg_losses[name] = avg_loss
            task_avg_ppls[name] = math.exp(avg_loss) 
            
    # Return exactly 8 items now
    return eval_ppl, eval_epoch_loss, val_step_loss, [], [], 0.0, task_avg_losses, task_avg_ppls

def evaluation_overfit(model,train_config, batch, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    model.eval()
    # eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch_unused in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            if step > 1:
                break
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                """ check generation logits and targets  """

                generation_logits = outputs.generation_logits #batch * len_x, decoder_vocab_size

                batch_size = batch['input_ids'].shape[0]
                length = batch['input_ids'].shape[1]-1 
                no_attributes = 6


                generation_logits_reshaped = torch.reshape(generation_logits, (batch_size, length, no_attributes, -1))

                # print(f"generation_logits:{generation_logits_reshaped.shape}")
                max_values, max_indices = torch.max(generation_logits_reshaped, dim=-1)
                # print(f"max_indices:{max_indices.shape}, {max_indices}")
                torch.save(generation_logits_reshaped, "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits.pth")
                torch.save(max_indices, "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits_max.pth")

                
                try:
                    decoded_tokens = tokenizer.convert_from_language_tokens(torch.max(generation_logits, dim=-1))
                    torch.save(torch.tensor(decoded_tokens), "/data/scratch/acw753/MusicLlama/ddp-MusicLlama-decoder_overfitting/batch_data_train_logits_max_decoded_tokens.pth")
                    print(f"decoded_tokens:{decoded_tokens}")
                except:
                    print(f"failed to decode tokens")

                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp)):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp or train_config.enable_ddp:
        eval_epoch_loss = eval_epoch_loss / world_size

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, outputs.generation_logits, outputs.generation_hidden_state, outputs.logits


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_xccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params / 1e6:.2f} Million")
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")
        print(f"Trainable %: {(trainable_params / total_params) * 100:.2f}%\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

# --- HIGH-PERFORMANCE JSON SERIALIZATION ---
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    import json
    USE_ORJSON = False
    print("⚠️ orjson not found. Falling back to standard json. (Install via: pip install orjson)")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, 
                 val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl, val_task_losses=None,
                 train_task_history=None, val_task_ppls=None):
    
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
        "val_task_losses": val_task_losses or [],
        "train_task_history": train_task_history or [],
        "val_task_ppls": val_task_ppls or []
    }
    
    if USE_ORJSON:
        # orjson serializes to bytes, so we MUST use "wb" (write binary)
        # OPT_SERIALIZE_NUMPY prevents crashes if any numpy floats sneak into the lists
        with open(output_filename, "wb") as f:
            f.write(orjson.dumps(metrics_data, option=orjson.OPT_SERIALIZE_NUMPY))
    else:
        # Standard fallback
        with open(output_filename, "w") as f:
            json.dump(metrics_data, f)