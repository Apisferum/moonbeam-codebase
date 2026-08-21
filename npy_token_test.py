import os
import numpy as np
import glob

def scan_dataset(folder_name, folder_path):
    files = glob.glob(os.path.join(folder_path, "*.npy"))
    main_files = [f for f in files if "bar_beat_chord" not in f and "_chord.npy" not in f]
    
    if not main_files:
        print(f"--- {folder_name} SCAN ---")
        print("No main sequence files found in this folder!")
        print("-" * 40 + "\n")
        return

    max_file = ""
    max_len = 0
    
    min_file = ""
    min_len = float('inf') # Start at infinity to find the true minimum
    
    over_limit = 0
    
    for f in main_files:
        try:
            length = np.load(f).shape[0]
        except Exception:
            continue # Skip corrupted files
            
        # Track Maximum
        if length > max_len: 
            max_len = length
            max_file = os.path.basename(f)
            
        # Track Minimum
        if length < min_len:
            min_len = length
            min_file = os.path.basename(f)
            
        # Track Context Limit Violations
        if length > 1024: 
            over_limit += 1
            
    print(f"--- {folder_name} SCAN ---")
    print(f"Total main sequence files checked: {len(main_files)}")
    print(f"Shortest file: {min_file} ({min_len} tokens)")
    print(f"Longest file:  {max_file} ({max_len} tokens)")
    print(f"Max length: {max_len} | Over 1024 limit: {over_limit}")
    print("-" * 40 + "\n")

# Using relative paths as you provided, 
# but make sure you run this from the root Moonbeam folder!
scan_dataset("ComMU", "processed/ComMU/processed")
scan_dataset("SLakh", "processed/SLAKH2100/processed")
scan_dataset("EMOPIA", "processed/EMOPIA2.2/processed")