# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from llama_recipes.datasets.grammar_dataset.grammar_dataset import get_dataset as get_grammar_dataset
from llama_recipes.datasets.alpaca_dataset import InstructionDataset as get_alpaca_dataset
from llama_recipes.datasets.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from llama_recipes.datasets.lakh_dataset import LakhDataset as get_lakhmidi_dataset
# from llama_recipes.datasets.merge_dataset import MergeDataset as get_merge_dataset
from llama_recipes.datasets.emophia_con_gen_dataset import Emophia_Con_Gen_Datasets as get_emophia_con_gen_dataset
from llama_recipes.datasets.commu_con_gen_dataset import Commu_Con_Gen_Datasets as get_commu_con_gen_dataset
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

# --- NEW ADDITIONS FOR SLAKH AND MULTI-TASK ---
from llama_recipes.datasets.slakh_con_gen_dataset import Slakh_Con_Gen_Datasets
from llama_recipes.datasets.multi_task_dataset import MultiTaskDataset

def get_slakh_con_gen_dataset(dataset_config, tokenizer, split):
    return Slakh_Con_Gen_Datasets(dataset_config, tokenizer, split)

def get_multi_task_dataset(dataset_config, tokenizer, split):
    # INJECT EMOPIA MASTER DICTIONARY IDs HERE!
    tokenizer.emotion_token_4Q1 = [-350 for _ in range(6)]
    tokenizer.emotion_token_4Q2 = [-351 for _ in range(6)]
    tokenizer.emotion_token_4Q3 = [-352 for _ in range(6)]
    tokenizer.emotion_token_4Q4 = [-353 for _ in range(6)]

    from llama_recipes.datasets.commu_con_gen_dataset import Commu_Con_Gen_Datasets
    from llama_recipes.datasets.emophia_con_gen_dataset import Emophia_Con_Gen_Datasets
    from llama_recipes.datasets.slakh_con_gen_dataset import Slakh_Con_Gen_Datasets
    from types import SimpleNamespace

    # Create sub-configs for each dataset
    commu_config = SimpleNamespace(
        data_dir=dataset_config.commu_data_dir,
        csv_file=dataset_config.commu_csv_file,
        additional_token_dict_path=dataset_config.commu_additional_token_dict_path,
        if_add_chords_in_transformer=True,
        if_add_metadata_in_transformer=True
    )
    
    emopia_config = SimpleNamespace(
        data_dir=dataset_config.emopia_data_dir,
        csv_file=dataset_config.emopia_csv_file,
        model_config_path=dataset_config.model_config_path
    )
    
    slakh_config = SimpleNamespace(
        data_dir=dataset_config.slakh_data_dir,
        csv_file=dataset_config.slakh_csv_file,
        model_config_path=dataset_config.model_config_path
    )
    
    # --- FIX: Translate the split names for each specific dataset ---
    # CoMMU expects "val", while EMOPIA and SLakh expect "test"
    commu_split = "val" if split in ["test", "val", "validation"] else split
    emopia_split = "test" if split in ["test", "val", "validation"] else split
    slakh_split = "test" if split in ["test", "val", "validation"] else split

    # Initialize the 3 datasets with their correct specific splits
    commu_ds = Commu_Con_Gen_Datasets(commu_config, tokenizer, commu_split)
    emopia_ds = Emophia_Con_Gen_Datasets(emopia_config, tokenizer, emopia_split)
    slakh_ds = Slakh_Con_Gen_Datasets(slakh_config, tokenizer, slakh_split)
    
    return MultiTaskDataset(commu_ds, emopia_ds, slakh_ds)