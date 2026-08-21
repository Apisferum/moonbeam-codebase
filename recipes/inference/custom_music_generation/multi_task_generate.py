import fire
import json
import os
import torch
from generation import MusicLlama

def main(
    ckpt_dir: str,
    lora_dir: str,
    active_adapter: str, # "commu_lora", "emopia_lora", "slakh_lora"
    task_type: str,      # "commu", "emopia", "slakh"
    model_config_path: str,
    additional_token_dict_path: str,
    output_midi: str = "generated_multitask.mid",
    max_gen_len: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)

    with open(additional_token_dict_path, "r") as f:
        master_dict = json.load(f)

    # 1. Load Model (Loads base + 3 adapters)
    generator = MusicLlama.build_multi_task(
        ckpt_dir=ckpt_dir, lora_dir=lora_dir, active_adapter="commu_lora", # Placeholder, we will merge next
        model_config_path=model_config_path, max_seq_len=max_gen_len, max_batch_size=1,
        additional_token_dict=master_dict
    )

    # --- METHOD 1: HEURISTIC TIES MERGING ---
    def heuristic_router(t_type):
        if t_type == "commu": return [1.0, 0.0, 0.0]
        elif t_type == "emopia": return [0.2, 0.8, 0.0]   # Keep 20% CoMMU for basic rhythm
        elif t_type == "slakh": return [0.2, 0.0, 0.8]
        elif t_type == "hybrid": return [0.5, 0.25, 0.25] # The Ultimate Blend
        return [0.33, 0.33, 0.34]

    weights = heuristic_router(task_type)
    
    # TIES merging prevents the Frankenstein interference
    generator.model.add_weighted_adapter(
        adapters=["commu_lora", "emopia_lora", "slakh_lora"],
        weights=weights,
        adapter_name="merged_super_lora",
        combination_type="ties"  # <-- THE MAGIC BULLET
    )
    generator.model.set_adapter("merged_super_lora")
    print(f"--> TIES-merged adapter activated with weights: CoMMU={weights[0]}, EMOPIA={weights[1]}, SLakh={weights[2]}")
    # ----------------------------------------

    # 2. Build Prompt Metadata
    metadata_ids = []
    if task_type == "emopia":
        # Example: Happy = <emo_q2>
        metadata_ids = [master_dict["<emo_q2>"]] 
    elif task_type == "slakh":
        # Example: Full Orchestra + Strings + Brass
        metadata_ids = [
            master_dict["<slakh_orch_full>"],
            master_dict["<slakh_sec_Strings>"],
            master_dict["<slakh_sec_Brass_Winds>"]
        ]
    elif task_type == "commu":
        # Standard 11 CoMMU tokens
        metadata_ids = [
            master_dict["audio_key_cmajor"], master_dict["pitch_range_mid"], master_dict["num_measures_8"],
            master_dict["bpm_120"], master_dict["genre_cinematic"], master_dict["track_role_main_melody"],
            master_dict["inst_string_ensemble"], master_dict["sample_rhythm_standard"], master_dict["time_signature_4/4"],
            master_dict["min_velocity_50"], master_dict["max_velocity_100"]
        ]

    # The engine expects metadata as a list of lists: [[id1, id2, ...]]
    metadata_condition_decoder = [metadata_ids]
    
    # Format the prompt (Just the SOS token to start generation)
    # We pass an empty chord condition because EMOPIA/SLakh don't use the CoMMU chord decoder
    prompts = [[[generator.tokenizer.sos_token_compound for _ in range(6)]]] 

    print(f"--> Generating {task_type} music with {active_adapter}...")
    
    # 3. Generate!
    results = generator.music_completion(
        prompts,
        bpm_condition=[120], # Dummy values for EMOPIA/SLakh
        time_signature_condition=["4/4"],
        num_measures_condition=[8],
        metadata_condition=metadata_condition_decoder,
        chord_condition=None, # CRITICAL: No chords for EMOPIA/SLakh
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        condition_token_lengths=[1],
        chord_dict=None,
        if_return_chords=False
    )

    # 4. Save MIDI
    midi_obj = results[0]['generation']['content'][0]
    midi_obj.save(output_midi)
    print(f"--> Success! Saved to {output_midi}")

if __name__ == "__main__":
    fire.Fire(main)