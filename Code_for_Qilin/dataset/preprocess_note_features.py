import numpy as np
import pandas as pd
from datasets import load_dataset
import os
import torch

# 配置参数（与训练代码保持一致）
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

# 最大值取log
log_feature_max_values = {
    key: np.ceil(np.log2(value)) if int(value) > 0 else value
    for key, value in feature_max_values.items()
}

feature_max_bits = {}
for feat, max_val in log_feature_max_values.items():
    # 计算二进制位数
    max_bits = len(bin(int(max_val))) - 2  # bin(5) -> '0b101'，实际位数为3
    total_bits = max_bits
    feature_max_bits[feat] = total_bits

# 特征处理函数
def process_feature(feat_name, value):
    """处理单个特征值"""
    # 处理特殊值：NaN或None转换为0
    if pd.isna(value) or value is None:
        value = 0.0

    # 确保值不超过最大值
    max_val = feature_max_values.get(feat_name, 0.0)
    if value > max_val:
        value = max_val

    # 对特定特征应用log转换（排除列表中的特征）
    if feat_name in all_feat:
        value = np.ceil(np.log2(value)) if int(value) > 0 else value

    # 转换为整数
    value = int(value)
    
    # 获取二进制位数
    total_bits = feature_max_bits.get(feat_name, 16)  # 默认16位
    
    # 数值转二进制字符串并补零
    binary_str = bin(value)[2:].zfill(total_bits)
    binary_str = [int(bit) for bit in binary_str]
    # 转换为0/1张量
    # binary_tensor = torch.tensor([int(bit) for bit in binary_str], dtype=torch.float32)
    # 转换为整数列表
    return binary_str

# 加载笔记数据
file_paths = [
    "dataset/PocessedQilin/notes/train-00000-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00001-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00002-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00003-of-00005.parquet",
    "dataset/PocessedQilin/notes/train-00004-of-00005.parquet"
]
# corpus = load_dataset("parquet", data_files=file_paths, split="train")


# 批量处理每个文件
for path in file_paths:
    print(f"Processing: {path}")
    
    # 加载 parquet 数据
    df = pd.read_parquet(path)
    
    # 处理特征列
    for feat in log_feature_max_values.keys():
        df[feat] = df[feat].apply(lambda val: process_feature(feat, val))
    
    # 保存新文件（前缀 log-）
    filename = os.path.basename(path)
    new_path = os.path.join(os.path.dirname(path), f"log-{filename}")
    df.to_parquet(new_path, index=False)
    print(f"Saved processed file: {new_path}")

print("All files processed and saved.")
# # 对于每一个文件，读取修改前后的第一条数据，打印
# for path in file_paths:
#     df = pd.read_parquet(path)
#     first_row = df.head(1).squeeze().tolist()
#     print(f"修改前:{first_row}")

#     # 构造 log 文件路径
#     dirname = os.path.dirname(path)
#     basename = os.path.basename(path)
#     log_path = os.path.join(dirname, f"log-{basename}")

#     df = pd.read_parquet(log_path)
#     first_row_log = df.head(1).squeeze().tolist()
#     print(f"修改后:{first_row_log}")

