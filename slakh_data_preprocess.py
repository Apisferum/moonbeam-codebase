import os
import yaml
import json
import csv
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

num_cores = multiprocessing.cpu_count()

ORCHESTRAL_FAMILIES = {
    "Rhythm": ["Bass", "Drums", "Chromatic Percussion", "Percussive"],
    "Keys_Synths": ["Piano", "Organ", "Synth Pad", "Synth Lead"],
    "Guitars": ["Guitar"], 
    "Strings": ["Strings", "Strings (continued)"],
    "Brass_Winds": ["Brass", "Reed", "Pipe", "Ethnic"]
}

def chunk_compounds(compounds, silence_threshold=1024, max_tokens=1000):
    if not compounds: return []
    
    # --- CRITICAL FIX: Strictly sort notes by onset time to prevent negative timeshifts ---
    compounds = sorted(compounds, key=lambda x: x[0])
    # ---------------------------------------------------------------------------------------

    # Step 1: Split by long silences
    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshifts = [onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]
    
    cur_pos = 0
    silence_chunks = []
    for pointer in range(len(onsets)):
        if timeshifts[pointer] > silence_threshold:
            silence_chunks.append(compounds[cur_pos:pointer])
            cur_pos = pointer
    silence_chunks.append(compounds[cur_pos:])
    
    # Step 2: Enforce max_tokens limit
    final_chunks = []
    for chunk in silence_chunks:
        if not chunk: continue
        for i in range(0, len(chunk), max_tokens):
            sub_chunk = chunk[i:i+max_tokens]
            if not sub_chunk: continue
            
            # Reset the onset of the sub_chunk to 0 to preserve relative timing
            first_onset = sub_chunk[0][0]
            reset_sub_chunk = [[comps[0] - first_onset] + comps[1:] for comps in sub_chunk]
            final_chunks.append(reset_sub_chunk)
            
    return [c for c in final_chunks if c]

def process_single_track(args):
    track_dir, split, output_folder, master_dict, onset_vocab_size, dur_vocab_size = args
    track_id = os.path.basename(track_dir)
    meta_path = os.path.join(track_dir, 'metadata.yaml')
    midi_path = os.path.join(track_dir, 'all_src.mid')
    
    if not os.path.exists(midi_path) or not os.path.exists(meta_path):
        return None

    try:
        with open(meta_path, 'r') as f:
            meta = yaml.safe_load(f)
            
        active_stems = [s for s in meta.get('stems', {}).values() if s.get('midi_saved', False)]
        if not active_stems:
            return None
            
        ids = set()
        # 1. Ensemble Size
        if len(active_stems) > 8: 
            ids.add(master_dict["<slakh_orch_full>"])
        else: 
            ids.add(master_dict["<slakh_orch_chamber>"])
        
        # 2. Orchestral Sections
        for stem in active_stems:
            inst_class = stem.get('inst_class', '')
            for family, members in ORCHESTRAL_FAMILIES.items():
                if inst_class in members:
                    ids.add(master_dict[f"<slakh_sec_{family}>"])
                    break
        
        # Initialize tokenizer
        tokenizer = MusicTokenizer(
            timeshift_vocab_size=onset_vocab_size, 
            dur_vocab_size=dur_vocab_size, 
            octave_vocab_size=13, 
            pitch_class_vocab_size=14, 
            instrument_vocab_size=131, 
            velocity_vocab_size=130
        )
        
        compounds = tokenizer.midi_to_compound(midi_path, calibate_to_default_tempo=True)
        if not compounds:
            return None
            
        chunks = chunk_compounds(compounds, silence_threshold=1024, max_tokens=1000)
        
        # Define the 'processed' subfolder
        processed_dir = os.path.join(output_folder, "processed")
        
        saved_files = []
        for i, chunk in enumerate(chunks):
            out_name = f"{track_id}_{i}.npy"
            # Save .npy inside the 'processed' subfolder
            out_path = os.path.join(processed_dir, out_name)
            np.save(out_path, np.array(chunk))
            saved_files.append([out_name, split, str(list(ids))])
            
        return saved_files

    except Exception as e:
        print(f"Failed to process {track_id}: {e}")
        return None

def main(args):
    with open(args.master_dict, 'r') as f:
        master_dict = json.load(f)
        
    with open(args.model_config, 'r') as f:
        config = json.load(f)
        onset_vocab_size = config.get("onset_vocab_size", 4099)
        dur_vocab_size = config.get("dur_vocab_size", 4099)
        
    # Create the root output folder and the 'processed' subfolder
    os.makedirs(args.output_folder, exist_ok=True)
    processed_dir = os.path.join(args.output_folder, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    csv_rows = []
    tasks = []
    
    # SLakh is structured as: slakh_root/train/Track00001, slakh_root/validation/..., etc.
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(args.dataset_folder, split)
        if not os.path.exists(split_dir):
            continue
        for track_id in os.listdir(split_dir):
            track_dir = os.path.join(split_dir, track_id)
            if os.path.isdir(track_dir):
                tasks.append((track_dir, split, args.output_folder, master_dict, onset_vocab_size, dur_vocab_size))
                
    print(f"Found {len(tasks)} SLakh tracks to process using {num_cores} CPU cores.")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_single_track, tasks), total=len(tasks)))
        
    for res in results:
        if res is not None:
            csv_rows.extend(res)
            
    # Save the CSV in the root output folder (SLAKH2100/)
    csv_path = os.path.join(args.output_folder, 'slakh2100_split.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_base_name', 'split', 'metadata_ids'])
        writer.writerows(csv_rows)
        
    print(f"Done! Saved {len(csv_rows)} chunks.")
    print(f".npy files saved to: {processed_dir}")
    print(f"CSV file saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--master_dict', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    main(parser.parse_args())