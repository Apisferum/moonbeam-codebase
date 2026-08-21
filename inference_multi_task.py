import torch
import json
import argparse
import os
import numpy as np
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM_Conditional_Generation
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
from peft import PeftModel

def load_multi_task_model(base_model_path, lora_dir, active_adapter, config_path):
    print(f"--> Loading Base Model from {base_model_path}...")
    config = LlamaConfig.from_pretrained(config_path)
    model = LlamaForCausalLM_Conditional_Generation(config)
    
    # Load base weights
    checkpoint = torch.load(base_model_path, map_location="cpu")
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    
    print(f"--> Loading LoRA Adapters from {lora_dir}...")
    # Initialize PEFT model with the first adapter, then add the others
    model = PeftModel.from_pretrained(model, os.path.join(lora_dir, "commu_lora"), adapter_name="commu_lora")
    model.load_adapter(os.path.join(lora_dir, "emopia_lora"), adapter_name="emopia_lora")
    model.load_adapter(os.path.join(lora_dir, "slakh_lora"), adapter_name="slakh_lora")
    
    # Set the active adapter based on user choice
    model.set_adapter(active_adapter)
    print(f"--> Active Adapter set to: {active_adapter}")
    
    model.eval()
    if torch.cuda.is_available():
        model.to(torch.bfloat16).to("cuda")
        
    return model

def build_prompt(task_type, prompt_args, master_dict_path):
    """
    Translates human-readable prompts into the negative token IDs from your Master Dictionary.
    """
    with open(master_dict_path, 'r') as f:
        master_dict = json.load(f)
        
    metadata_tokens = []
    
    if task_type == "emopia":
        # Example: prompt_args = {"mood": "happy"}
        mood_map = {"happy": "<emo_q2>", "sad": "<emo_q1>", "angry": "<emo_q3>", "calm": "<emo_q4>"}
        token_name = mood_map.get(prompt_args.get("mood", "happy"), "<emo_q2>")
        metadata_tokens.append(master_dict[token_name])
        
    elif task_type == "slakh":
        # Example: prompt_args = {"orchestra": "full", "sections": ["Strings", "Brass_Winds"]}
        orch_map = {"full": "<slakh_orch_full>", "chamber": "<slakh_orch_chamber>"}
        metadata_tokens.append(master_dict[orch_map.get(prompt_args.get("orchestra", "full"), "<slakh_orch_full>")])
        
        for section in prompt_args.get("sections", ["Strings"]):
            metadata_tokens.append(master_dict[f"<slakh_sec_{section}>"])
            
    elif task_type == "commu":
        # Example: prompt_args = {"genre": "cinematic", "bpm": 120}
        if "genre" in prompt_args: metadata_tokens.append(master_dict[f"genre_{prompt_args['genre']}"])
        if "bpm" in prompt_args: metadata_tokens.append(master_dict[f"bpm_{prompt_args['bpm']}"])
        if "key" in prompt_args: metadata_tokens.append(master_dict[f"audio_key_{prompt_args['key']}"])

    return metadata_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to moonbeam_839M.pt")
    parser.add_argument("--lora_dir", type=str, required=True, help="Path to the epoch_X folder containing the 3 adapters")
    parser.add_argument("--config_path", type=str, default="src/llama_recipes/configs/model_config_multi_task.json")
    parser.add_argument("--master_dict", type=str, default="processed/ComMU/indexed_tokens_dict.json")
    parser.add_argument("--active_adapter", type=str, choices=["commu_lora", "emopia_lora", "slakh_lora"], required=True)
    parser.add_argument("--task_type", type=str, choices=["emopia", "slakh", "commu"], required=True)
    parser.add_argument("--output_midi", type=str, default="generated_multitask.mid")
    args = parser.parse_args()

    # 1. Load Model
    model = load_multi_task_model(args.base_model_path, args.lora_dir, args.active_adapter, args.config_path)
    
    # 2. Load Tokenizer
    tokenizer = MusicTokenizer(
        timeshift_vocab_size=model.config.onset_vocab_size, 
        dur_vocab_size=model.config.dur_vocab_size, 
        octave_vocab_size=model.config.octave_vocab_size, 
        pitch_class_vocab_size=model.config.pitch_class_vocab_size, 
        instrument_vocab_size=model.config.instrument_vocab_size, 
        velocity_vocab_size=model.config.velocity_vocab_size, 
        sos_token=model.config.sos_token, 
        eos_token=model.config.eos_token, 
        pad_token=model.config.pad_token
    )

    # 3. Build Prompt
    # --- CUSTOMIZE YOUR PROMPT HERE ---
    if args.task_type == "slakh":
        prompt_args = {"orchestra": "full", "sections": ["Strings", "Brass_Winds"]}
    elif args.task_type == "emopia":
        prompt_args = {"mood": "happy"}
    elif args.task_type == "commu":
        prompt_args = {"genre": "cinematic", "bpm": 120, "key": "cmajor"}
        
    metadata_ids = build_prompt(args.task_type, prompt_args, args.master_dict)
    print(f"--> Generated Metadata IDs: {metadata_ids}")

    # 4. Generate (Autoregressive Loop)
    print("--> Starting Generation...")
    # NOTE: You will need to copy the author's exact autoregressive sampling loop 
    # from their original inference/generate script here. 
    # Because Moonbeam uses a custom GRU decoder and 6D compound tokens, 
    # standard HuggingFace `model.generate()` will not work out-of-the-box.
    # 
    # PSEUDOCODE FOR THE LOOP:
    # input_ids = format_metadata_and_sos(metadata_ids, tokenizer)
    # with torch.no_grad():
    #     outputs = model(input_ids=input_ids)
    #     hidden_state = outputs.hidden_states
    #     for step in range(max_steps):
    #         next_token_logits, hidden_state = model(decoded_language_tokens=prev_token, decoded_hidden_state=hidden_state)
    #         next_token = sample_from_6D_logits(next_token_logits)
    #         generated_tokens.append(next_token)
            
    # 5. Decode and Save
    # midi_obj = tokenizer.decode_to_midi(generated_tokens)
    # midi_obj.write(args.output_midi)
    print(f"--> Generation Complete! Saved to {args.output_midi}")

if __name__ == "__main__":
    main()