# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain
import multiprocessing as mp
import mmap
import concurrent.futures
from torch.utils.data import Dataset
import numpy as np
import os
import pickle

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096, split="train", data_dir = None): #TODO: add split
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.mmap_path = os.path.join(data_dir, f"dataset_chunksize_{str(chunk_size)}_{split}.mmap")
        self.offsets_cache_path = os.path.join(data_dir, f"dataset_chunksize_{str(chunk_size)}_offsets_{split}.pkl")
        self.sample_offsets = []
        
        self.buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        if os.path.isfile(self.mmap_path) and os.path.isfile(self.offsets_cache_path):
            print(f"using cached dataset: {self.mmap_path}")
            self.mmap_file = open(self.mmap_path, 'rb')
            self._load_offsets()
        else:
            # Initialize memory-mapped file
            self.mmap_file = open(self.mmap_path, 'wb+')
            
            # Preprocess dataset and store in memory-mapped file
            self._preprocess_and_store()
        self.mmap_obj = mmap.mmap(self.mmap_file.fileno(), 0, access=mmap.ACCESS_READ)


    def _preprocess_and_store(self): 
        sample_count = 0  # Initialize a counter for the samples
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            sample['attention_mask'] = [sample_count] * len(sample['input_ids'])  # In the model, each sample will only attend to itself according to the "sample_count" value
            self.buffer = {k: v + sample[k] for k, v in self.buffer.items()}
            while len(next(iter(self.buffer.values()))) > self.chunk_size:
                chunk = {k: v[:self.chunk_size] for k, v in self.buffer.items()}
                self.buffer = {k: v[self.chunk_size:] for k, v in self.buffer.items()}
                self._store_sample(chunk)
                
            sample_count += 1  # Increment the sample counter
        
        # Store any remaining data in the buffer as the last chunk
        # if any(len(v) > 0 for v in self.buffer.values()):
        #     self._store_sample(self.buffer)

        self.mmap_file.close()  # Close the file after preprocessing
        self.mmap_file = open(self.mmap_path, 'rb')  # Reopen in read mode for use in __getitem__
        self._save_offsets()
    """def _store_sample(self, sample):
        # Convert sample data to numpy arrays and then to binary format
        input_ids = np.array(sample['input_ids'], dtype=np.int32)
        attention_mask = np.array(sample['attention_mask'], dtype=np.int32)
        labels = np.array(sample['labels'], dtype=np.int32)
        
        # Write sample data to memory-mapped file
        offset = self.mmap_file.tell()
        self.sample_offsets.append(offset)
        
        for array in [input_ids, attention_mask, labels]:
            self.mmap_file.write(array.tobytes())
    
    def _read_sample(self, offset):
        self.mmap_file.seek(offset)
        
        input_ids = np.frombuffer(self.mmap_file.read(self.chunk_size * 4), dtype=np.int32) #32 bits = 4 bytes
        attention_mask = np.frombuffer(self.mmap_file.read(self.chunk_size * 4), dtype=np.int32)
        labels = np.frombuffer(self.mmap_file.read(self.chunk_size * 4), dtype=np.int32)
        
        return {
            'input_ids': input_ids.tolist(),
            'attention_mask': attention_mask.tolist(),
            'labels': labels.tolist(),
        }"""
    def _store_sample(self, sample):
        # Convert sample data to numpy arrays and then to binary format
        input_ids = np.array(sample['input_ids'], dtype=np.int32)
        attention_mask = np.array(sample['attention_mask'], dtype=np.int32)
        labels = np.array(sample['labels'], dtype=np.int32)
        
        # Store shapes
        input_ids_shape = np.array(input_ids.shape, dtype=np.int32)
        attention_mask_shape = np.array(attention_mask.shape, dtype=np.int32)
        labels_shape = np.array(labels.shape, dtype=np.int32)

        # Write sample data to memory-mapped file
        offset = self.mmap_file.tell()
        self.sample_offsets.append(offset)

        for array in [input_ids_shape, input_ids, attention_mask_shape, attention_mask, labels_shape, labels]:
            self.mmap_file.write(array.tobytes())
    
    def _read_sample(self, offset):
        self.mmap_obj.seek(offset)

        # Read shapes first
        input_ids_shape = np.frombuffer(self.mmap_obj.read(2 * 4), dtype=np.int32)  # 2 elements for shape of input_ids
        input_ids_size = np.prod(input_ids_shape)
        input_ids = np.frombuffer(self.mmap_obj.read(input_ids_size * 4), dtype=np.int32).reshape(input_ids_shape)
        
        attention_mask_shape = np.frombuffer(self.mmap_obj.read(1 * 4), dtype=np.int32)  # 1 element for shape of attention_mask
        attention_mask_size = np.prod(attention_mask_shape)
        attention_mask = np.frombuffer(self.mmap_obj.read(attention_mask_size * 4), dtype=np.int32).reshape(attention_mask_shape)

        labels_shape = np.frombuffer(self.mmap_obj.read(2 * 4), dtype=np.int32)  # 2 elements for shape of labels
        labels_size = np.prod(labels_shape)
        labels = np.frombuffer(self.mmap_obj.read(labels_size * 4), dtype=np.int32).reshape(labels_shape)

        return {
            'input_ids': input_ids.tolist(),
            'attention_mask': attention_mask.tolist(),
            'labels': labels.tolist(),
        }
    def __getitem__(self, idx):
        offset = self.sample_offsets[idx]
        return self._read_sample(offset)

    def __len__(self):
        return len(self.sample_offsets)

    def __del__(self):
        self.mmap_file.close()
    
    def _save_offsets(self):
        with open(self.offsets_cache_path, 'wb') as f:
            pickle.dump(self.sample_offsets, f)

    def _load_offsets(self):
        with open(self.offsets_cache_path, 'rb') as f:
            self.sample_offsets = pickle.load(f)
class ConcatDataset_vanilla(Dataset): 
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        sample_count = 0  # Initialize a counter for the samples
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            sample['attention_mask'] = [sample_count]*len(sample['attention_mask']) #In the model, each sample will only attend to itself according to the "sample_count" value
            buffer = {k: v + sample[k] for k, v in buffer.items()}            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
            sample_count += 1  # Increment the sample counter

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

class ConcatDataset_dummy_padding(Dataset): 
    def __init__(self, dataset, chunk_size=4096, split = "train",data_dir = None):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []
        
        sample_count = 1  # Initialize a counter for the samples
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True): 
            length = len(sample['input_ids'])
            #pad the input if it is less than chunk_size
            if length<self.chunk_size:
                pad_length = self.chunk_size - length
                pad_length_bar_beat_chord = self.chunk_size - len(sample['bar_beat_chord_condition'])
                sample['input_ids'] = sample['input_ids'] + [[0, 0, 0, 0, 0, 0]]*pad_length 
                sample['labels'] = sample['labels'] + [[0, 0, 0, 0, 0, 0, 0]]*pad_length
                sample['bar_beat_chord_condition'] = sample['bar_beat_chord_condition'] + [[0, 0, 0]]*pad_length_bar_beat_chord
                sample['attention_mask'] = [sample_count] * length + [sample_count*2] * pad_length
                self.samples.append(sample)
                sample_count+=1
            elif length == self.chunk_size:
                sample['attention_mask'] = [sample_count] * length
                self.samples.append(sample)
                sample_count+=1
            else:
                continue 

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

class ConcatDataset_serial(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []

        # Split the indices for the dataset into chunks for serial processing
        indices = list(range(len(self.dataset)))
        num_chunks = (len(indices) + chunk_size - 1) // chunk_size  # Calculate number of chunks
        chunks = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

        # Process each chunk serially
        for i, chunk_indices in enumerate(tqdm(chunks, desc="Processing chunks", dynamic_ncols=True)):
            self.samples.extend(self._process_chunk(chunk_indices, i))

    def _process_chunk(self, chunk_indices, start_idx):
        samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        sample_count = start_idx
        for idx in chunk_indices:
            sample = self.dataset[idx]
            sample['attention_mask'] = [sample_count] * len(sample['attention_mask'])
            buffer = {k: v + sample[k] for k, v in buffer.items()}
            while len(next(iter(buffer.values()))) > self.chunk_size:
                samples.append({k: v[:self.chunk_size] for k, v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}
            sample_count += 1

        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

class ConcatDataset_parallel(Dataset):
    def __init__(self, dataset, chunk_size=4096, num_workers=None):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.samples = []
        if num_workers is None:
            num_workers = mp.cpu_count()
        print(f"concat dataset using {num_workers} workers")
        # Split the indices for the dataset into chunks for parallel processing
        indices = list(range(len(self.dataset)))
        batch_size = 64
        num_chunks = (len(indices) + batch_size - 1) // batch_size  # Calculate number of chunks
        chunks = [indices[i*batch_size:(i+1)*batch_size] for i in range(num_chunks)]

        # Use concurrent.futures to process each chunk in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk_indices, i) for i, chunk_indices in enumerate(chunks)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing chunks", dynamic_ncols=True):
                try:
                    self.samples.extend(future.result())
                except Exception as e:
                    print(f"Error processing chunk: {e}")

    def _process_chunk(self, chunk_indices, start_idx):
        samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        sample_count = start_idx
        for idx in chunk_indices:
            sample = self.dataset[idx]
            sample['attention_mask'] = [sample_count] * len(sample['attention_mask'])
            buffer = {k: v + sample[k] for k, v in buffer.items()}
            while len(next(iter(buffer.values()))) > self.chunk_size:
                samples.append({k: v[:self.chunk_size] for k, v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}
            sample_count += 1

        return samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)