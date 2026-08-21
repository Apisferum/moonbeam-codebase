# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import pandas as pd
import numpy as np
import os
import re
# from llama import Dialog, Llama
from generation import MusicLlama
import random
import ast
import json
def find_chord_string(filename, csv_file):
    df = pd.read_csv(csv_file)
    chord_str = df[df['id'] == filename.split("_")[-1][:-4]]['chord_progressions'].values[0]
    return chord_str
def metadata_id2label(metadata_id, additional_token_dict_inv):
    label = [additional_token_dict_inv[x].replace("/",".") for x in metadata_id]
    return "-".join(label)
def count_notes_in_range(tokens, min_val, max_val):
    correct = sum(1 for token in tokens if min_val <= token <= max_val)
    return correct

def find_mapping_vel(vel_token, additional_token_dict_inv):
    return int(additional_token_dict_inv[vel_token].split("_")[-1])

def find_mapping_pitch(pitch_token, additional_token_dict_inv):
    range_str = "_".join(additional_token_dict_inv[pitch_token].split("_")[2:])
    RANGE_MAP = {
        "very_low": (12, 23),       # C-2 to B0
        "low": (24, 35),           # C1 to B1
        "mid_low": (36, 47),       # C2 to B2
        "mid": (48, 59),           # C3 to B3
        "mid_high": (60, 71),      # C4 to B4
        "high": (72, 83),          # C5 to B5
        "very_high": (84, 107)     # C6 to G8
    }

    return RANGE_MAP[range_str]


def main(
    ckpt_dir: str,
    csv_file: str,
    tokenizer_path: str,
    model_config_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    num_test_data: int = 50,
    if_add_chords_in_transformer: bool = True,
    if_add_metadata_in_transformer: bool = False,
    max_gen_len: Optional[int] = None,
    finetuned_PEFT_weight_path: Optional[str] = None,
    additional_token_dict_path: Optional[str] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    # Set the random seed for CPU and GPU
    seed = 42
    import torch
    torch.manual_seed(seed)
    random.seed(seed)  # You can choose any seed value, 42 is commonly used
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.    

    with open(additional_token_dict_path, "r") as f:
        additional_token_dict = json.load(f)

    additional_token_dict_inv = {v: k for k, v in additional_token_dict.items()}
    generator = MusicLlama.build_commu_con_gen(
        ckpt_dir=ckpt_dir,
        model_config_path = model_config_path,  #TODO: check where to add conditions
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        finetuned_PEFT_weight_path = finetuned_PEFT_weight_path,
        additional_token_dict = additional_token_dict
    ) 

    df = pd.read_csv(csv_file)

    # 2. Find the row where 'split' is 'test' and get the corresponding .npy file path
    split = "val"
    test_filenames = df[df['split'] == split]['file_base_name'].tolist() #change from test to train, since there's really no test data in LLM
    test_metadata_ids = df[df['split'] == split]['label'].tolist()
    test_metadata_ids = [list(ast.literal_eval(l)) for l in test_metadata_ids]
    test_filenames_chord = df[df['split'] == split]['chord_file_base_name'].tolist()
    
    #temporary fix: find chord symbols from original csv file
    test_filenames_chord_str = [find_chord_string(x, "/data/scratch/acw753/datasets/commu/commu_meta.csv") for x in test_filenames]

    test_files = list(zip(test_filenames, test_metadata_ids, test_filenames_chord, test_filenames_chord_str))
    test_files_sampled = random.sample(test_files, num_test_data)
    prompts = []
    metadata_condition_decoder = []
    chord_condition_decoder = []
    bpm_condition = []
    time_signature_condition = []
    num_measures_condition = []
    if_add_chords_in_decoder = generator.model.if_add_chord_in_decoder
    if_add_metadata_in_decoder = generator.model.if_add_metadata_in_decoder
    print(f"if_add_chords_in_transformer: {if_add_chords_in_transformer}, if_add_metadata_in_transformer: {if_add_metadata_in_transformer}")
    print(f"if_add_chords_in_decoder: {if_add_chords_in_decoder}, if_add_metadata_in_decoder: {if_add_metadata_in_decoder}")
    for filename, metadata_id, chord_file_name, chord_str in test_files_sampled:
        print(f"Loading filename: {filename}, chord_file_name: {chord_file_name}")
        raw_tokens = np.load(os.path.join(os.path.dirname(csv_file), "processed",filename))
        raw_tokens_chord = np.load(os.path.join(os.path.dirname(csv_file), "processed",chord_file_name))
        metadata_condition_decoder.append(metadata_id)
        bpm_condition.append(int(additional_token_dict_inv[metadata_id[3]].split("_")[-1]))
        time_signature_condition.append(additional_token_dict_inv[metadata_id[-3]].split("_")[-1])
        num_measures_condition.append(int(additional_token_dict_inv[metadata_id[2]].split("_")[-1]))
        chord_condition_decoder.append(ast.literal_eval(chord_str)[0]) #TODO: think of what to add
        metadata_id = [[x for _ in range(6)] for x in metadata_id] #TODO: think of a better way to do this
        test_data_with_sos = generator.tokenizer.encode_series_con_gen_commu(raw_tokens, raw_tokens_chord, metadata_tokens = metadata_id, if_only_keep_condition_tokens = True, if_add_chords_in_transformer = if_add_chords_in_transformer, if_add_metadata_in_transformer = if_add_metadata_in_transformer)
        prompts.append(test_data_with_sos)
    
    if not if_add_metadata_in_decoder: 
        metadata_condition_decoder = None
    if not if_add_chords_in_decoder:
        chord_condition_decoder = None

    condition_token_lengths = [len(x) for x in prompts]
    if if_add_chords_in_transformer:
        chord_token_indices = [[x.index(generator.tokenizer.soc_token_compound), x.index(generator.tokenizer.eoc_token_compound)] for x in prompts]
    else:
        chord_token_indices = None
    
    results = generator.music_completion(
        prompts, #this controls whether condition exists in the transformer
        bpm_condition = bpm_condition,
        time_signature_condition = time_signature_condition,
        num_measures_condition = num_measures_condition,
        metadata_condition = metadata_condition_decoder, #this controls whether condition exists in the decoder
        chord_condition = chord_condition_decoder, 
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        condition_token_lengths = condition_token_lengths, #remove sos token and emotion token
        chord_token_indices = chord_token_indices,
        chord_dict_path = "/data/scratch/acw753/finetune/commu_con_gen/chord_dictionary.json" #temporary fix: add chord dict path
    )
    
    save_folder = os.path.join(finetuned_PEFT_weight_path, os.path.basename(ckpt_dir), f"temperature_{temperature}_top_p_{top_p}")
    os.makedirs(save_folder, exist_ok=True)

    total_correct_vel = 0 
    total_vel = 0 
    total_correct_pitch = 0 
    total_pitch = 0 
    difference_chord_music_ending_time = []
    for i, (dialog, result, metadata_id) in enumerate(zip(prompts, results, [x[1] for x in test_files_sampled])):

        epoch_step = re.search(r'(\d+-\d+)\.pt$', ckpt_dir).group(1)
        for j, (sub_mid, chord_mid, tokens, chord_tokens) in enumerate(zip(result['generation']['content'], result['generation']['chord'], result['generation']['tokens'], result['generation']['chord_tokens'])):
            save_path = f'{save_folder}/{epoch_step}_{str(i)}_{str(j)}_meta_{metadata_id2label(metadata_id, additional_token_dict_inv)}.mid'
            try:
                sub_mid.save(save_path)
                chord_mid.save(save_path.replace(".mid", "_chord.mid"))
                print(f"midis saved at {save_path}")
                print(f"tokens: {tokens}")

                #count how many notes in the right velocity range 
                correct_vel = count_notes_in_range([x[-1] for x in tokens], find_mapping_vel(metadata_id[-2], additional_token_dict_inv), find_mapping_vel(metadata_id[-1], additional_token_dict_inv)) #TODO: in the future, change this! 
                total_correct_vel += correct_vel
                total_vel += len(tokens)

                # #count how many notes in the right pitch range 
                print(f"mean pitch: {np.mean([x[2]*12 + x[3] for x in tokens])}")
                correct_pitch = count_notes_in_range([np.mean([x[2]*12 + x[3] for x in tokens])], *find_mapping_pitch(metadata_id[1], additional_token_dict_inv))
                total_correct_pitch += correct_pitch
                total_pitch += 1

                #the differece (mean and variance) between the chord ending time and the last note of the generated midi 
                chord_ending_time = chord_tokens[-1][0] + chord_tokens[-1][1]
                note_ending_time = tokens[-1][0] + tokens[-1][1]
                difference_chord_music_ending_time.append(chord_ending_time - note_ending_time)
                #the differece (mean and variance) between the chord ending time and the last note of the generated midi    
            except Exception as e:
                print(f"Error: {e}, sub_mid: {sub_mid}")
                continue
        #Also save the prompt 
        print("\n==================================\n")

    print("percentage of correct velocity", total_correct_vel/total_vel)
    print("percentage of correct pitch", total_correct_pitch/total_pitch)
    print("mean and variance of the difference between the chord ending time and the last note of the generated midi", np.mean(difference_chord_music_ending_time), np.var(difference_chord_music_ending_time))
if __name__ == "__main__":
    fire.Fire(main)
