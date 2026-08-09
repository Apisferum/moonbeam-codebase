import json
import numpy as np
import pandas as pd
import ast
import os
from torch.utils.data import Dataset

class Slakh_Con_Gen_Datasets(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.data_dir = dataset_config.data_dir
        self.tokenizer = tokenizer
        
        df = pd.read_csv(dataset_config.csv_file)
        self.data = df[df['split'] == partition].reset_index(drop=True)
        
        # Load the 6D mapping from your multi-task config
        with open(dataset_config.model_config_path, 'r') as f:
            self.pos_map = json.load(f)['metadata_tokens_pos']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # The .npy files are inside the 'processed' subfolder
        npy_path = os.path.join(self.data_dir, "processed", row['file_base_name'])
        if not os.path.exists(npy_path):
            npy_path = os.path.join(self.data_dir, row['file_base_name'])
            
        raw_tokens = np.load(npy_path)
        
        # metadata_ids is a stringified list like "[-360, -363]"
        slakh_ids = ast.literal_eval(row['metadata_ids'])
        
        # Map each ID to its 6D array using the config
        metadata_tokens = [self.pos_map[str(sid)] for sid in slakh_ids]
        
        # Use the author's exact CoMMU pipeline function!
        encoded_tokens = self.tokenizer.encode_series_con_gen_commu(
            raw_token_series=raw_tokens,
            raw_chord_series=[],
            metadata_tokens=metadata_tokens,
            if_add_chords_in_transformer=False,
            if_add_metadata_in_transformer=True
        )
        
        encoded_tokens_label = self.tokenizer.encode_series_labels_con_gen_commu(encoded_tokens)

        return {
            "input_ids": encoded_tokens,
            "labels": encoded_tokens_label,
            "attention_mask": [],
            "task_id": 2  # 2 = SLakh
        }