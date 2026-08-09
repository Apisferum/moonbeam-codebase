# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

    
@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    
    
@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "/PATH/TO/CSV" 
    test_split: str = "/PATH/TO/CSV"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "/PATH/TO/DATADIR"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"

@dataclass
class lakhmidi_dataset:
    dataset: str = "lakhmidi_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATADIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class merge_dataset:
    dataset: str = "merge_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATADIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class emophia_con_gen_dataset:
    dataset: str = "emophia_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "/PATH/TO/DATADIR"
    csv_file: str = "/PATH/TO/CSV"

@dataclass
class commu_con_gen_dataset:
    dataset: str = "commu_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_dir: str = "/PATH/TO/DATADIR"
    csv_file: str = "/PATH/TO/CSV"
    additional_token_dict_path: str = "/PATH/TO/JSON"
    if_add_chords_in_transformer: bool=True
    if_add_metadata_in_transformer: bool=True

@dataclass
class slakh_con_gen_dataset:
    dataset: str = "slakh_con_gen_dataset"
    train_split: str = "train"
    test_split: str = "validation" # SLakh uses 'validation' instead of 'test'
    data_dir: str = "/PATH/TO/SLAKH/processed"
    csv_file: str = "/PATH/TO/SLAKH/slakh_split.csv"
    model_config_path: str = "/PATH/TO/model_config_multi_task.json"

@dataclass
class multi_task_dataset:
    dataset: str = "multi_task_dataset"
    train_split: str = "train"
    test_split: str = "test"
    batch_size: int = 1  # Required for HomogeneousTaskSampler
    
    # --- KAGGLE CLOUD PATHS ---
    
    # CoMMU paths
    commu_data_dir: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/ComMU"
    commu_csv_file: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/ComMU/train_test_split.csv"
    commu_additional_token_dict_path: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/ComMU/indexed_tokens_dict.json"
    
    # EMOPIA paths
    emopia_data_dir: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/EMOPIA2.2"
    emopia_csv_file: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/EMOPIA2.2/train_test_split.csv"
    
    # SLakh paths
    slakh_data_dir: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/SLAKH2100"
    slakh_csv_file: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/SLAKH2100/slakh2100_split.csv"
    
    # Shared config (Points to the codebase dataset)
    model_config_path: str = "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-codebase/src/llama_recipes/configs/model_config_multi_task.json"
