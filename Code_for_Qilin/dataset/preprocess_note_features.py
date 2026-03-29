import numpy as np
import pandas as pd
from datasets import load_dataset
import os
import torch

# Configuration parameters (consistent with training code)
all_feat= {
    'rec_view_time': 269531514.0,
    'video_width': 7680.0,
    'video_height': 10240.0,
    'full_view_times': 9889130.0,
    'search_follow_num': 8697.0,
    'valid_view_times': 11677727.0,
    'video_duration': 7777.0,
    'search_view_time': 9579437.0,
    'view_time': 270575311.0,
    'search_comment_num': 7375.0,
    'comment_num': 143961.0,
    'search_share_num': 7323.0,
    'share_num': 66777.0
}

# feature_max_values = {
#     'rec_view_time': 269531514.0,
#     'full_view_times': 9889130.0,
#     'search_follow_num': 8697.0,
#     'valid_view_times': 11677727.0,
#     'search_view_time': 9579437.0,
#     'view_time': 270575311.0,
#     'search_comment_num': 7375.0,
#     'comment_num': 143961.0,
#     'search_share_num': 7323.0,
#     'share_num': 66777.0
# }
feature_max_values= {
    'rec_view_time': 269531514.0,
    'video_width': 7680.0,
    'video_height': 10240.0,
    'full_view_times': 9889130.0,
    'search_follow_num': 8697.0,
    'valid_view_times': 11677727.0,
    'video_duration': 7777.0,
    'search_view_time': 9579437.0,
    'view_time': 270575311.0,
    'search_comment_num': 7375.0,
    'comment_num': 143961.0,
    'search_share_num': 7323.0,
    'share_num': 66777.0
}

# Apply logarithm to maximum values
log_feature_max_values = {
    key: np.ceil(np.log2(value)) if int(value) > 0 else value
    for key, value in feature_max_values.items()
}

feature_max_bits = {}
for feat, max_val in log_feature_max_values.items():
    # Calculate the number of binary bits
    max_bits = len(bin(int(max_val))) - 2  # bin(5) -> '0b101', actual bits: 3
    total_bits = max_bits
    feature_max_bits[feat] = total_bits

# Feature processing function
def process_feature(feat_name, value):
    """Process a single feature value"""
    # Handle special values: convert NaN or None to 0
    if pd.isna(value) or value is None:
        value = 0.0

    # Ensure the value does not exceed the maximum
    max_val = feature_max_values.get(feat_name, 0.0)
    if value > max_val:
        value = max_val

    # Apply log transformation to specific features (exclude features in the list)
    if feat_name in all_feat:
        value = np.ceil(np.log2(value)) if int(value) > 0 else value

    # Convert to integer
    value = int(value)
    
    # Get the number of binary bits
    total_bits = feature_max_bits.get(feat_name, 16)  # Default 16 bits
    
    # Convert number to binary string and pad with zeros
    binary_str = bin(value)[2:].zfill(total_bits)
    binary_str = [int(bit) for bit in binary_str]
    # Convert to 0/1 tensor
    # binary_tensor = torch.tensor([int(bit) for bit in binary_str], dtype=torch.float32)
    # Convert to integer list
    return binary_str

# Load note data
file_paths = [
    "dataset/PocessedQilin/notes/train-00000-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00001-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00002-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00003-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00004-of-00005.parquet"
]
# corpus = load_dataset("parquet", data_files=file_paths, split="train")


# Process each file in batches
for path in file_paths:
    print(f"Processing: {path}")
    
    # Load parquet data
    df = pd.read_parquet(path)
    
    # Process feature columns
    for feat in log_feature_max_values.keys():
        df[feat] = df[feat].apply(lambda val: process_feature(feat, val))
    
    # Save new file (prefix: log-)
    filename = os.path.basename(path)
    new_path = os.path.join(os.path.dirname(path), f"log-{filename}")
    df.to_parquet(new_path, index=False)
    print(f"Saved processed file: {new_path}")

print("All files processed and saved.")
# # For each file, read and print the first row before and after modification
# for path in file_paths:
#     df = pd.read_parquet(path)
#     first_row = df.head(1).squeeze().tolist()
#     print(f"Before modification:{first_row}")

#     # Construct log file path
#     dirname = os.path.dirname(path)
#     basename = os.path.basename(path)
#     log_path = os.path.join(dirname, f"log-{basename}")

#     df = pd.read_parquet(log_path)
#     first_row_log = df.head(1).squeeze().tolist()
#     print(f"After modification:{first_row_log}")