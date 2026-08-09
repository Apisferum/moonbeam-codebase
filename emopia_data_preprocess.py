import os
import matplotlib.pyplot as plt
from llama_recipes.datasets.music_tokenizer import MusicTokenizer
import traceback
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from collections import Counter
import numpy as np
from collections import defaultdict
import argparse
import csv
from sklearn.model_selection import train_test_split
import pandas as pd
num_cores = multiprocessing.cpu_count()

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

def find_midi_files(folder):
    midi_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.midi') or file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    return midi_files

def find_midi_files_from_file(dataset_name, split_file, dataset_folder):
    if dataset_name == "pijama30":
        df = pd.read_csv(split_file)
        df = df[df['album_split_0'] != 'val']
        midi_files = df['midi_filepath'].tolist()
        midi_files = [os.path.join(dataset_folder, f) for f in midi_files]
        splits = df['album_split_0'].tolist()
        labels = df['player_id'].tolist()
    elif dataset_name == "pianist8":
        import pickle
        train_files = pickle.load(open(dataset_folder+'/pianist8_train.pkl','rb'))
        test_files = pickle.load(open(dataset_folder+'/pianist8_test.pkl','rb'))
        train_labels_raw = [x.split("/")[0] for x in train_files]
        test_labels_raw = [x.split("/")[0] for x in test_files]

        # Step 1: Get unique labels
        unique_labels = sorted(set(train_labels_raw + test_labels_raw))
        # Step 2: Create a mapping from label to index
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        train_labels = [label_to_index[label] for label in train_labels_raw]
        test_labels = [label_to_index[label] for label in test_labels_raw]

        train_files = [os.path.join(dataset_folder,'midi',x) for x in train_files]
        test_files = [os.path.join(dataset_folder,'midi',x) for x in test_files]
        midi_files = train_files+test_files
        splits = ['train']*len(train_files) + ['test']*len(test_files)
        labels = train_labels + test_labels
    elif dataset_name == "emopia":
        train_files = pd.read_csv(f"{split_file}/train_clip.csv")['clip_name'].tolist()
        test_files = pd.read_csv(f"{split_file}/test_clip.csv")['clip_name'].tolist()
        train_files = [os.path.join(dataset_folder, f) for f in train_files]
        test_files = [os.path.join(dataset_folder, f) for f in test_files]
        train_labels_raw = [x.split("/")[-1].split("_")[0] for x in train_files]
        test_labels_raw = [x.split("/")[-1].split("_")[0] for x in test_files]

        label_to_index = {"Q1":0, "Q2":1, "Q3":2, "Q4":3}
        train_labels = [label_to_index[label] for label in train_labels_raw]
        test_labels = [label_to_index[label] for label in test_labels_raw]

        midi_files = train_files+test_files
        splits = ['train']*len(train_files) + ['test']*len(test_files)
        labels = train_labels + test_labels
    elif dataset_name == "Giant_Piano_MIDI":
        df = pd.read_csv(split_file)
        midi_files = df['midi_file'].tolist()
        midi_files = [os.path.join(dataset_folder, f) for f in midi_files]
        splits = df['split'].tolist()
        labels_raw = df['composer'].tolist()

        # Step 1: Get unique labels
        unique_labels = sorted(set(labels_raw))
        # Step 2: Create a mapping from label to index
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        print("label dictionary", label_to_index)
        labels = [label_to_index[label] for label in labels_raw]

    assert len(midi_files) == len(splits) == len(labels)
    return midi_files, splits, labels

def process_midi_file_safe_v3(midi_file, split, label, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    """
    input: midi_file, split, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold

    output: a list of processed chunked dictionaries [{file: file_path, split: split, onsets: onset_counter, durations: duration_counter, length: length}, ...] or None
    """
    try:
        out = process_midi_file_v3(midi_file, split, label, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold)
        if out is not None:
            for out_chunk in out:
                if out_chunk is not None:
                    np.save(out_chunk['file'], out_chunk['compounds'])
        return out
    except Exception as e:
        # Log failure with full traceback
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {midi_file}:\n')
            log.write(traceback.format_exc())
            log.write('\n')
        return [None]

def detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size):
    #check if file contains large timeshits and durations
    onsets = [c[0] for c in compounds]
    onsets_padded = [0] + onsets
    timeshift_counter = Counter([onsets_padded[i+1] - onsets_padded[i] for i in range(len(onsets_padded) - 1)]) 
    duration_counter = Counter([c[1] for c in compounds])
    onsets_exceed_vocab_size = any(key > onset_vocab_size-3 for key in timeshift_counter.keys()) #why -3? onset_vocab_size -1, onset_vocab_size -2 are assigned to sos and eos, max value is onset_vocab_size - 3
    duration_exceed_vocab_size = any(key > dur_vocab_size-3 for key in duration_counter.keys()) #if dur_vocab_size=1026, 1024 and 1025 are reserved for sos and eos, the largest value becomes 1023
    return onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter

def filter_large_ts_dur(compounds, output_file_path, split, label, onset_vocab_size, dur_vocab_size, log_file):
    """detect large timeshifts and durations, return None if any of them exceed the vocab size else return the processed dictionary"""

    onsets_exceed_vocab_size, duration_exceed_vocab_size, timeshift_counter, duration_counter = detect_large_timeshifts_and_durations(compounds, onset_vocab_size, dur_vocab_size)

    if onsets_exceed_vocab_size or duration_exceed_vocab_size:
        with open(log_file, 'a') as log:
            log.write(f'Failed to process {output_file_path} but during finetuning these files will be kept:\n')
            log.write(f'{output_file_path} contains large onsets: {onsets_exceed_vocab_size}, large durations: {duration_exceed_vocab_size}\n')
            # Optionally, print the largest onset and duration exceeding the vocab size
            if onsets_exceed_vocab_size:
                largest_onset = max(key for key in timeshift_counter.keys() if key > onset_vocab_size - 3)
                log.write(f'Largest onset: {largest_onset}\n')
            if duration_exceed_vocab_size:
                largest_duration = max(key for key in duration_counter.keys() if key > dur_vocab_size - 3)
                log.write(f'Largest duration: {largest_duration}\n')

        return {
            'file': output_file_path,
            'compounds': compounds,
            'split': split,
            'label':label,
            'timeshifts':dict(timeshift_counter),
            'durations':dict(duration_counter),           
            'length_token': len(compounds),
            'length_duration': compounds[-1][0]+compounds[-1][1],
        }

    else:

        return {
            'file': output_file_path,
            'compounds': compounds,
            'split': split,
            'label':label,
            'timeshifts':dict(timeshift_counter),
            'durations':dict(duration_counter),           
            'length_token': len(compounds),
            'length_duration': compounds[-1][0]+compounds[-1][1],
        }

def process_midi_file_v3(midi_file, split, label, onset_vocab_size, dur_vocab_size, output_folder, log_file, silence_threshold = None):
    #convert midi to compounds
    compounds = tokenizer.midi_to_compound(midi_file)

    output_file_name = midi_file.replace("/", "_").replace(".midi", ".npy").replace(".mid", ".npy")
    output_file_path = os.path.join(output_folder, output_file_name)
    # Always chunk by max_tokens to prevent DataLoader truncation, even if no silence threshold is provided
    s_thresh = silence_threshold if silence_threshold else 999999
    list_of_compounds = chunk_compounds(compounds, silence_threshold=s_thresh, max_tokens=1000)
    
    if len(list_of_compounds) == 1:
        return [filter_large_ts_dur(list_of_compounds[0], output_file_path, split, label, onset_vocab_size, dur_vocab_size, log_file)]
    else:
        list_of_output_file_path = [os.path.join(output_folder, 
                                                 midi_file.split("/")[-1].replace('.midi', f'_{i}.npy').replace('.mid', f'_{i}.npy')) 
                                                 for i in range(len(list_of_compounds))]
        return [filter_large_ts_dur(c, p, split, label, onset_vocab_size, dur_vocab_size, log_file) for (c, p) in zip(list_of_compounds, list_of_output_file_path)]

# Main script execution
if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process MIDI files for the MIDI dataset.')

    # Define and Parse the arguments
    parser.add_argument('--dataset_name', type=str, help='Dataset Name')
    parser.add_argument('--dataset_folder', type=str, help='Path to the dataset folder containing MIDI files.')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where processed files will be saved.')
    parser.add_argument('--model_config', type=str, help='Model configuration file that decides the vocab size')
    parser.add_argument('--train_test_split_file', type=lambda x: None if x == "None" else str(x), help='Path to the split file.')
    parser.add_argument('--train_ratio', type=float, help='Training/Total')
    parser.add_argument('--ts_threshold', type=lambda x: None if x == "None" else int(x), help='If Timeshift exceeds this value, chunk the file')

    args = parser.parse_args()

    #get file paths
    midi_output_folder = args.output_folder+"/processed"
    log_file = args.output_folder + "/failed_midi_files.log"
    csv_file_path = args.output_folder+"/train_test_split.csv"
    train_stats_file = args.output_folder+"/train_tokens_stats.json"
    test_stats_file = args.output_folder+"/test_tokens_stats.json"
    os.makedirs(midi_output_folder, exist_ok=True)
    
    #find all midi files 
    if not args.train_test_split_file:
        train_files, test_files = train_test_split(find_midi_files(args.dataset_folder), test_size=1-args.train_ratio, random_state=42)
        midi_files = train_files+test_files
        splits = ['train']*len(train_files) + ['test']*len(test_files)
    else:
        midi_files, splits, labels = find_midi_files_from_file(args.dataset_name, args.train_test_split_file, args.dataset_folder)
    
    print(f"{len(midi_files)} midi files found! {midi_files[:3]}")
    #determine vocab size
    with open(args.model_config, 'r') as file:
        data = json.load(file)
        onset_vocab_size = data.get("onset_vocab_size", None) #value - 2 = max value of timeshift
        dur_vocab_size = data.get("dur_vocab_size", None) #value - 2 = max value of duration
        octave_vocab_size = data.get("octave_vocab_size", None)
        pitch_class_vocab_size = data.get("pitch_class_vocab_size", None)
        instrument_vocab_size = data.get("instrument_vocab_size", None)
        velocity_vocab_size = data.get("velocity_vocab_size", None)
        assert onset_vocab_size and dur_vocab_size
    print(f"processing using {num_cores} cpus. tokenizer config: max timeshift allowed: {onset_vocab_size-3}, max duration allowed: {dur_vocab_size-3}")
    tokenizer = MusicTokenizer(timeshift_vocab_size = onset_vocab_size, dur_vocab_size = dur_vocab_size, octave_vocab_size = octave_vocab_size, pitch_class_vocab_size = pitch_class_vocab_size, instrument_vocab_size = instrument_vocab_size, velocity_vocab_size = velocity_vocab_size)  
    
    #process all midi files
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = list(tqdm(executor.map(process_midi_file_safe_v3, midi_files, splits, labels, [onset_vocab_size]*len(midi_files), [dur_vocab_size]*len(midi_files), [midi_output_folder]*len(midi_files), [log_file]*len(midi_files), [args.ts_threshold]*len(midi_files)), total=len(midi_files), desc='Processing MIDI files'))
    
    results = [item for sublist in results for item in sublist] #results contains of list of lists, flatten it

    csv_rows = []
    results_success = [result for result in results if result is not None]
    for result in results_success: 
        csv_rows.append([os.path.basename(result['file']), result['split'], result['label']])

    # Write CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['file_base_name', 'split', 'label'])  # Write header
        csv_writer.writerows(csv_rows)

    print(f'Processed {len(midi_files)} files including {len(results)} chunks with {len(results)-len(results_success)} failures. CSV file saved to {csv_file_path}')