# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import glob
from dataclasses import dataclass

def _find_path(env_key: str, local_relative: str, kaggle_pattern: str, fallback_path: str) -> str:
    # 1. Environment Variable
    if os.environ.get(env_key):
        return os.environ[env_key]
    
    # 2. Local sibling directory
    codebase_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    workspace_root = os.path.abspath(os.path.join(codebase_dir, ".."))
    
    # Try case-insensitive search for directories under workspace_root
    if os.path.exists(workspace_root):
        for folder in os.listdir(workspace_root):
            if folder.lower() in ["moonbeam multi-task data", "moonbeam-multi-task-data", "moonbeam_multi_task_data"]:
                full_path = os.path.join(workspace_root, folder, local_relative)
                if os.path.exists(full_path):
                    return full_path
                
        # Sibling direct fallback
        sibling_guess = os.path.abspath(os.path.join(workspace_root, "Moonbeam Multi-Task Data", local_relative))
        if os.path.exists(sibling_guess):
            return sibling_guess

    # 3. Kaggle search
    for root in ["/kaggle/input", "/kaggle/working"]:
        if os.path.isdir(root):
            matches = glob.glob(os.path.join(root, "**", kaggle_pattern), recursive=True)
            if matches:
                return matches[0]
                
    # 4. Fallback
    return fallback_path
    
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
    
    # --- DYNAMIC RESOLUTION OF PATHS ---
    
    # CoMMU paths
    commu_data_dir: str = _find_path("COMMU_DATA_DIR", "ComMU", "ComMU", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/ComMU")
    commu_csv_file: str = _find_path("COMMU_CSV_FILE", "ComMU/train_test_split.csv", "train_test_split.csv", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/ComMU/train_test_split.csv")
    commu_additional_token_dict_path: str = _find_path("COMMU_DICT_PATH", "ComMU/indexed_tokens_dict.json", "indexed_tokens_dict.json", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/ComMU/indexed_tokens_dict.json")
    
    # EMOPIA paths
    emopia_data_dir: str = _find_path("EMOPIA_DATA_DIR", "EMOPIA2.2", "EMOPIA2.2", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/EMOPIA2.2")
    emopia_csv_file: str = _find_path("EMOPIA_CSV_FILE", "EMOPIA2.2/train_test_split.csv", "EMOPIA2.2/train_test_split.csv", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/EMOPIA2.2/train_test_split.csv")
    
    # SLakh paths
    slakh_data_dir: str = _find_path("SLAKH_DATA_DIR", "SLAKH2100", "SLAKH2100", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/SLAKH2100")
    slakh_csv_file: str = _find_path("SLAKH_CSV_FILE", "SLAKH2100/slakh2100_split.csv", "slakh2100_split.csv", "/home/aashishbishow/ProjectX/moonbeam-codebasemoonbeam-multi-task-data/SLAKH2100/slakh2100_split.csv")
    
    # Shared config (Points to the codebase dataset)
    model_config_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "model_config_multi_task.json"))
