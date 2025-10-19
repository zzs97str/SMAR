import pandas as pd
import numpy as np
import os

all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}
person_feat=["gender","age"]
number_feat = ['dense_feat9', 'dense_feat26', 'dense_feat37', 'dense_feat34', 'dense_feat25', 'dense_feat11', 'dense_feat20', 'dense_feat13', 'dense_feat10', 'dense_feat14', 'dense_feat24', 'dense_feat1', 'dense_feat33', 'dense_feat28', 'dense_feat36', 'follows_num', 'dense_feat2', 'dense_feat38', 'dense_feat18', 'dense_feat32', 'dense_feat12', 'dense_feat35', 'dense_feat31', 'dense_feat8']
need_log_feat = ['dense_feat26', 'dense_feat37', 'dense_feat34', 'dense_feat20', 'dense_feat10', 'dense_feat14', 'dense_feat24', 'dense_feat1', 'dense_feat33', 'dense_feat28', 'dense_feat36', 'follows_num', 'dense_feat2', 'dense_feat18', 'dense_feat32', 'dense_feat12', 'dense_feat35', 'dense_feat31', 'dense_feat8']




df = pd.read_parquet("dataset/PocessedQilin/user_feat/train-00000-of-00001.parquet")
# 特征处理函数
def process_user_feat(feat, value):
    # 性别处理
    if feat == "gender":
        gender_map = {"male": [1, 0], "female": [0, 1], "unknown": [0, 0], "": [0, 0]}
        return gender_map.get(value, [0, 0])
    
    # 年龄处理
    if feat == "age":
        age_buckets = ['1-12', '13-15', '16-18', '19-22', '23-25', '26-30', '31-35', '36-40', '40+', 'unknown', '']
        binary_str_1=  [1 if value == age else 0 for age in age_buckets]
        binary_str= binary_str_1
        return binary_str
    
    if feat in number_feat:
        if feat in need_log_feat:
            value = np.ceil(np.log2(value)) if int(value) > 0 else value
            # 转换为整数
        
        value = int(value)
        
        # 获取最大二进制位数
        max_val = all_feat.get(feat, 16)  # 默认16位
        total_bits=int(len(bin(int(max_val))) - 2)
        # 数值转二进制字符串并补零
        binary_str = bin(value)[2:].zfill(total_bits)
        
        # 转换为整数列表
        return [int(bit) for bit in binary_str]

for feat in all_feat.keys():
    df[feat] = df[feat].apply(lambda val: process_user_feat(feat, val))


path= "dataset/PocessedQilin/user_feat/train-00000-of-00001.parquet"

# 保存新文件（前缀 log-）
filename = os.path.basename(path)
new_path = os.path.join(os.path.dirname(path), f"log-{filename}")
df.to_parquet(new_path, index=False)
print(f"Saved processed file: {new_path}")


df = pd.read_parquet(path)
first_row = df.head(1).squeeze().tolist()
print(f"修改前:{first_row}")

# 构造 log 文件路径
dirname = os.path.dirname(path)
basename = os.path.basename(path)
log_path = os.path.join(dirname, f"log-{basename}")

df = pd.read_parquet(log_path)
first_row_log = df.head(1).squeeze().tolist()
print(f"修改后:{first_row_log}")
