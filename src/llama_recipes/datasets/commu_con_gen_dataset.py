import copy
import json

import torch
from torch.utils.data import Dataset
import glob
import mido
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
import pandas as pd
import os
import ast

class Commu_Con_Gen_Datasets(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        # Print the hidden value so we can see what the script is passing
        print(f"---> DEBUG: Commu dataset received partition split string: '{partition}' (Type: {type(partition)})")

        # If validation is disabled or string is empty, fall back to "val" to prevent crashes
        if partition in [False, "False", "none", "None", "", None]:
            partition = "val"

        # Check safely
        assert partition in ["train", "val", "validation"], f"Invalid partition name: {partition}"

        if partition == "validation":
            partition = "val"
        self.data_dir = dataset_config.data_dir
        split_data = pd.read_csv(dataset_config.csv_file)
        file_basenames = split_data['file_base_name'].values
        chord_file_base_name = split_data['chord_file_base_name'].values
        bar_beat_chord_file_basenames = split_data['beat_bar_base_name'].values
        splits = split_data['split'].values       
        labels = split_data['label'].values 
        self.file_basenames = [f for f, s in zip(file_basenames, splits) if s == partition]
        self.chord_file_basenames = [f for f, s in zip(chord_file_base_name, splits) if s == partition]
        self.bar_beat_chord_file_basenames = [f for f, s in zip(bar_beat_chord_file_basenames, splits) if s == partition]
        self.metadata_ids = [l for l, s in zip(labels, splits) if s == partition]
        self.metadata_ids = [list(ast.literal_eval(l)) for l in self.metadata_ids]
        # self.emotion_ids2token = {0:tokenizer.emotion_token_4Q1, 1:tokenizer.emotion_token_4Q2, 2:tokenizer.emotion_token_4Q3, 3:tokenizer.emotion_token_4Q4}
        self.tokenizer = tokenizer
        self.if_add_chords_in_transformer = dataset_config.if_add_chords_in_transformer
        self.if_add_metadata_in_transformer = dataset_config.if_add_metadata_in_transformer 
        
    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, index):
        raw_tokens = np.load(os.path.join(self.data_dir, "processed",self.file_basenames[index]))
        raw_tokens_chord = np.load(os.path.join(self.data_dir, "processed",self.chord_file_basenames[index]))
        raw_bar_beat_chord = np.load(os.path.join(self.data_dir, "processed",self.bar_beat_chord_file_basenames[index]))
        metadata_id = [[x for _ in range(6)] for x in self.metadata_ids[index]] #TODO: think of a better way to do this
        encoded_tokens = self.tokenizer.encode_series_con_gen_commu(raw_tokens, raw_tokens_chord, metadata_tokens = metadata_id, if_add_chords_in_transformer = self.if_add_chords_in_transformer, if_add_metadata_in_transformer = self.if_add_metadata_in_transformer)  #meta_data_tokens,<SOC> chords, <EOC>, <SOS> music_seq, <EOS>              
        encoded_tokens_label = self.tokenizer.encode_series_labels_con_gen_commu(encoded_tokens)

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "metadata_condition": [x[0] for x in metadata_id],
            "bar_beat_chord_condition": raw_bar_beat_chord.tolist(),
            "attention_mask":[] #Mask to be calculated dynamically during concatenation
        }

 