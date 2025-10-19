import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from utils import *
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from registry import register_class
from datasets import DatasetDict
import time


@register_class
class CrossEncoderTrainingDataProcessor:
    def __init__(self, **kwargs):
        print("initialing CrossEncoderTrainingDataProcessor")
        data_path = kwargs.get('dataset_name_or_path')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        max_length = kwargs.get('max_length')
        # 负样本池
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        print(f"negative_pool: {self.negative_pool}")
        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.negative_samples = kwargs.get('negative_samples', 3) 
        print(f"negative_samples: {self.negative_samples}")
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'


    def load_data(self):
        print(f"此时的数据集是：{self.data_path}")
        print("loading CrossEncoderTrainingData dataset")
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
                    "dataset/notes/log-train-00001-of-00005.parquet",
                    "dataset/notes/log-train-00002-of-00005.parquet",
                    "dataset/notes/log-train-00003-of-00005.parquet",
                    "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.dataset = load_dataset("parquet",data_files="dataset/SingleModal/train_single_modal.parquet", split="train")

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def collate_fn(self, batch):
        # 正负样本构造
        queries = []
        notes = []
        labels = []

        for item in batch:
            query = item["query"]
            search_idx = item["search_idx"]
            # impression_result_details是一个列表，每个元素是一个字典，包含了用户点击的笔记的索引和点击标签
            # impression_result_details是 train数据集的search_result_details_with_idx
            impression_result_details = item[self.negative_pool]
            # 获取正样本
            positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            assert len(positives) > 0, 'No positive samples found for query: ' + query
            # 随机选择一个正样本
            positive_idx = random.choice(positives)
            note_content = self.get_note_content(positive_idx)
            queries.append(query)
            notes.append(note_content)
            labels.append(1)
        
            negatives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 0]
            if len(negatives) < self.negative_samples:
                additional_samples = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                negatives.extend(additional_samples)
            else:
                negatives = random.sample(negatives, k=self.negative_samples)
            
            for note_idx in negatives:
                note_content = self.get_note_content(note_idx)
                queries.append(query)
                notes.append(note_content)
                labels.append(0)

        # query_note_pairs是一个列表，每个元素都是一个字符串，格式为 [查询] [SEP] [文档内容]
        query_note_pairs = [f"{q} [SEP] {n}" for q, n in zip(queries, notes)]
        
        inputs = self.tokenizer(
            query_note_pairs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = torch.tensor(labels, dtype=torch.float)
        # 每个查询生成 ‌1个正样本对‌ 和 ‌N个负样本对‌（N = negative_samples）。
        # 样本对格式为 [查询] [SEP] [文档内容]，通过分隔符 [SEP] 区分查询和文档

        return {"inputs": inputs, "labels": labels}

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

@register_class
class CrossEncoderTrainingDataProcessor_PairWise:
    def __init__(self, **kwargs):
        print("initialing CrossEncoderTrainingDataProcessor_Pairwise")
        data_path = kwargs.get('dataset_name_or_path')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        max_length = kwargs.get('max_length')
        # 负样本池
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        print(f"negative_pool: {self.negative_pool}")
        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.negative_samples = kwargs.get('negative_samples', 3) 
        print(f"negative_samples: {self.negative_samples}")
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'

        self.feature_max_values = {
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

        self.all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}


    def load_data(self):
        print(f"此时的数据集是：{self.data_path}")
        print("loading CrossEncoderTrainingData dataset")
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
                    "dataset/notes/log-train-00001-of-00005.parquet",
                    "dataset/notes/log-train-00002-of-00005.parquet",
                    "dataset/notes/log-train-00003-of-00005.parquet",
                    "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.dataset = load_dataset("parquet",data_files="dataset/SingleModal/train_single_modal.parquet", split="train")
        self.user_feat = load_dataset("parquet",data_files="dataset/user_feat/log-train-00000-of-00001.parquet", split="train")

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret

    def collate_fn_old(self, batch):
        # 收集三元组
        q_pos_list, q_neg_list = [], []
        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]

            # 正样本池：click==1，按 page_time 降序
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )[:10]  # 添加切片操作，截断至最多10个

            # 对每个正样本，构造若干负样本
            for pos in pos_pool:
                pos_time = pos['page_time']
                pos_idx = pos['note_idx']

                # 负样本候选：要么未点击，要么点击但 page_time 小于当前正样本
                neg_cands = [
                    d for d in impression_result_details
                    if (int(d['click']) == 0)
                    or (int(d['click']) == 1 and (not pd.isna(d['page_time'])) and d.get('page_time', 0) < pos_time)
                ]
                # 若不足，则从全语料随机补
                if len(neg_cands) < self.negative_samples:
                    # 需要补多少个
                    k = self.negative_samples - len(neg_cands)
                    # 从整个语料库的索引里随机取 k 个作为负样本
                    extra_idxs = random.sample(range(len(self.corpus)), k=k)
                    # 原有的 neg_cands 里已经是 dict 了，先取它们的 note_idx
                    neg_idxs = [d['note_idx'] for d in neg_cands] + extra_idxs
                else:
                    neg_idxs = random.sample([d['note_idx'] for d in neg_cands], self.negative_samples)

                # 将 (q, d+) 与每个 (q, d-) 分别记录
                pos_text = self.get_note_content(pos_idx)
                for neg_idx in neg_idxs:
                    neg_text = self.get_note_content(neg_idx)
                    q_pos_list.append(f"{query} [SEP] {pos_text}")
                    q_neg_list.append(f"{query} [SEP] {neg_text}")

        # 分别编码正负对
        inp_pos = self.tokenizer(
            q_pos_list,
            padding="max_length", truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inp_neg = self.tokenizer(
            q_neg_list,
            padding="max_length", truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        #在典型的 pair-wise 训练（例如用 Hinge Loss 或 Logistic Loss）中，我们并不需要对每个正负对再额外构造一个 labels 张量。相反，正负对自身的顺序就蕴含了监督信号
        return {"inp_pos": inp_pos, "inp_neg": inp_neg}

    def collate_fn(self, batch):
        # 收集三元组
        q_pos_list, q_neg_list = [], []
        # 收集原始特征值
        pos_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        neg_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        batch_features = {}
        # 收集用户特征值
        pos_user_feat_vals = {feat: [] for feat in self.all_feat}
        neg_user_feat_vals = {feat: [] for feat in self.all_feat}
        user_feat_vals = {feat: [] for feat in self.all_feat}

        for item in batch:
            query = item["query"]
            search_idx = item["search_idx"]
            impression_result_details = item[self.negative_pool]
            user_idx = item["user_idx"]

            # 正样本池：click==1，按 page_time 降序
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )[:10]  # 添加切片操作，截断至最多10个

            # 对每个正样本，构造若干负样本
            for pos in pos_pool:
                pos_time = pos['page_time']
                pos_idx = pos['note_idx']


                # 负样本候选：要么未点击，要么点击但 page_time 小于当前正样本
                neg_cands = [
                    d for d in impression_result_details
                    if (int(d['click']) == 0)
                ]
                # 若不足，则从全语料随机补
                if len(neg_cands) < self.negative_samples:
                    # 需要补多少个
                    k = self.negative_samples - len(neg_cands)
                    # 从整个语料库的索引里随机取 k 个作为负样本
                    extra_idxs = random.sample(range(len(self.corpus)), k=k)
                    # 原有的 neg_cands 里已经是 dict 了，先取它们的 note_idx
                    neg_idxs = [d['note_idx'] for d in neg_cands] + extra_idxs
                else:
                    neg_idxs = random.sample([d['note_idx'] for d in neg_cands], self.negative_samples)

                # 将 (q, d+) 与每个 (q, d-) 分别记录
                pos_text = self.get_note_content(pos_idx)
                for neg_idx in neg_idxs:
                    neg_text = self.get_note_content(neg_idx)
                    q_pos_list.append(f"{query} [SEP] {pos_text}")
                    q_neg_list.append(f"{query} [SEP] {neg_text}")

                    for feat, thresholds in self.feature_max_values.items():
                        raw_value_pos = self.get_note_feat(feat, pos_idx)
                        # print(f"binary_vec_pos:{binary_vec_pos}")
                        # binary_vec_pos实际上是一个张量
                        binary_vec_pos = torch.tensor(raw_value_pos,dtype=torch.float32)
                        pos_batch_feat_vals[feat].append(binary_vec_pos)

                        raw_value_neg = self.get_note_feat(feat, neg_idx)
                        binary_vec_neg = torch.tensor(raw_value_neg,dtype=torch.float32)
                        neg_batch_feat_vals[feat].append(binary_vec_neg)

                    for feat, thresholds in self.all_feat.items():
                        feat_value = self.user_feat[user_idx][feat] #数值型或者字符串
                        feat_value = torch.tensor(feat_value, dtype=torch.float32)
                        pos_user_feat_vals[feat].append(feat_value)
                        neg_user_feat_vals[feat].append(feat_value)


        for feat, thresholds in self.feature_max_values.items():
            batch_features[feat]=  pos_batch_feat_vals[feat] + neg_batch_feat_vals[feat]
        for feat, thresholds in self.all_feat.items():
            user_feat_vals[feat]=  pos_user_feat_vals[feat] + neg_user_feat_vals[feat]
        
        # print(f"len(batch_features['image_num']):{len(batch_features['image_num'])}")
        # 转成 tensor 并放到 device
        # batch_features = {
        #     feat: [torch.tensor(val, dtype=torch.long) for val in vals]
        #     for feat, vals in batch_features.items()
        # }
        
        # print(f"batch_features['image_num'].shape:{len(batch_features['image_num'])}")
        # 分别编码正负对
        inp_pos = self.tokenizer(
            q_pos_list,
            padding="max_length", truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inp_neg = self.tokenizer(
            q_neg_list,
            padding="max_length", truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # print(f"inp_neg['input_ids'].shape:{inp_neg['input_ids'].shape}")
        return {
            "inp_pos": inp_pos,
            "inp_neg": inp_neg,
            "features": batch_features,
            "user_feat": user_feat_vals
        }
    def get_dataloader(self):

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )


@register_class
class CrossEncoderTestDataProcessor:
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.max_length = kwargs.get('max_length')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.results_key = results_key # search_results
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        self.dataset = self.load_data()
        self.feature_max_values = {
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

        self.all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}

    def load_data(self):
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
                    "dataset/notes/log-train-00001-of-00005.parquet",
                    "dataset/notes/log-train-00002-of-00005.parquet",
                    "dataset/notes/log-train-00003-of-00005.parquet",
                    "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.user_feat = load_dataset("parquet",data_files="dataset/user_feat/log-train-00000-of-00001.parquet", split="train")
        data = load_dataset("parquet", data_files ="dataset/search_test/train-00000-of-00001.parquet", split="train")

        data = data.select(range(min(self.sample_num, len(data))))
        # 将数据均匀划分为 num_processes 个分片，每个进程（GPU）处理一个分片
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret

    def collate_fn(self, batch):
        queries = []
        notes = []
        note_idxs = []
        search_idxs = []
        # 收集原始特征值
        batch_features = {feat: [] for feat in self.feature_max_values}
        user_feat_vals = {feat: [] for feat in self.all_feat}
        # 标记正负样本用于计算 AUC
        Pos_Neg=[]

        for item in batch:
            query = item["query"]
            user_idx = item["user_idx"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            # candidates为 xhs的曝光结果，为一个列表，是 note_idx 的列表，如果没有分数，则默认为0.0
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            # 按照分数降序排列，sort默认升序，reverse=True表示反转，即降序
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            search_result_details_with_idx=item["search_result_details_with_idx"]
            
            for candidate in candidates:
                note_idx = int(candidate[0])  
                note_content = self.get_note_content(note_idx)
                
                queries.append(query)
                notes.append(note_content)
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)
                for note in search_result_details_with_idx:
                    if note["note_idx"]==  note_idx:
                        pos_or_neg= note["click"]
                Pos_Neg.append(int(pos_or_neg))

                for feat, thresholds in self.feature_max_values.items():
                    raw_value = self.get_note_feat(feat, note_idx)
                    binary_vec = torch.tensor(raw_value,dtype=torch.float32)
                    # print(f"binary_vec_pos:{binary_vec_pos}")
                    # binary_vec_pos实际上是一个张量
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] #数值型或者字符串
                    feat_value = torch.tensor(feat_value, dtype=torch.float32)
                    user_feat_vals[feat].append(feat_value)

        query_note_pairs = [f"{q} [SEP] {n}" for q, n in zip(queries, notes)]
        
        inputs = self.tokenizer(
            query_note_pairs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # print(Pos_Neg)
        # 测试集输入为：用户 query，对应的 note 内容
        return {"inputs": inputs, "note_idxs": note_idxs, "search_idxs": search_idxs,"features": batch_features, "Pos_Neg": Pos_Neg,"user_feat": user_feat_vals}

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
@register_class
class VLMCrossEncoderTrainingDataProcessor:
    def __init__(self, **kwargs):
        data_path = kwargs.get('dataset_name_or_path')
        processor_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.max_length = kwargs.get('max_length', 1024)
        self.negative_samples = kwargs.get('negative_samples', 3)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        print(f"use_recent_clicked_note_images:{self.use_recent_clicked_note_images}")
        self.processor_name = processor_name
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels= 100 * 28 * 28,
            trust_remote_code=True,
            use_fast=True
        )
        self.default_image = self._create_default_image()

    def _create_default_image(self):
        # create a default image with white color
        default_image = Image.new('RGB', (1024, 1024), color='white')
        return default_image
    
    def load_data(self):
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
            "dataset/notes/log-train-00001-of-00005.parquet",
            "dataset/notes/log-train-00002-of-00005.parquet",
            "dataset/notes/log-train-00003-of-00005.parquet",
            "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.dataset = load_dataset("parquet",data_files="dataset/SingleModal/train_single_modal.parquet", split="train")

    def get_note_content(self, note_idx):
        note = self.corpus[note_idx]
        image = self.default_image
        image_path = note['image_path']
        if len(image_path):
            try:
                image_path = os.path.join('afs', image_path[0])
                image = Image.open(image_path)
                image = image.resize((1024, 1024))
                image_size = image.size
                if image_size[0]<=0 or image_size[1]<=0:
                    image = self.default_image
            except Exception as e:
                print(f"Warning: Failed to load image for note {note_idx}: {e}")
            
        return {
            'text': self._get_text_content(note),
            'image': image
        }
    
    def _get_text_content(self, note):
        ret = ''
        if self.use_title:
            ret += note['note_title']
        if self.use_content:
            ret += note['note_content']
        return ret

    def collate_fn(self, batch):
        queries = []
        images = []
        labels = []

        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            
            positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            # 可能有多个正例
            assert len(positives) > 0, 'No positive samples found for query: ' + query
            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # 如果启用最近点击的笔记图像，则处理这些图像并拼接
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            positive_idx = random.choice(positives)
            note_content = self.get_note_content(positive_idx)
            # 选择一个正例，获取其内容，包括文本和图像

            # Template of conversation
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                ]
            }]
            
            queries.append(conversation)
            images.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
            labels.append(1)
            
            negatives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 0]
            if len(negatives) < self.negative_samples:
                additional_samples = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                negatives.extend(additional_samples)
            else:
                negatives = random.sample(negatives, k=self.negative_samples)
            
            for note_idx in negatives:
                note_content = self.get_note_content(note_idx)
                
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                    ]
                }]
                
                queries.append(conversation)
                images.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                labels.append(0)

        text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in queries]
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = torch.tensor(labels, dtype=torch.float)
        
        return {"inputs": inputs, "labels": labels}

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

@register_class
class VLMCrossEncoderTrainingDataProcessor_pairwise:
    def __init__(self, **kwargs):
        data_path = kwargs.get('dataset_name_or_path')
        processor_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.max_length = kwargs.get('max_length', 1024)
        self.negative_samples = kwargs.get('negative_samples', 3)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        print(f"use_recent_clicked_note_images:{self.use_recent_clicked_note_images}")
        self.processor_name = processor_name
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels= 100 * 28 * 28,
            trust_remote_code=True,
            use_fast=True
        )
        self.default_image = self._create_default_image()

        self.feature_max_values = {
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

        self.all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}


    def _create_default_image(self):
        # create a default image with white color
        default_image = Image.new('RGB', (1024, 1024), color='white')
        return default_image
    
    def load_data(self):
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
            "dataset/notes/log-train-00001-of-00005.parquet",
            "dataset/notes/log-train-00002-of-00005.parquet",
            "dataset/notes/log-train-00003-of-00005.parquet",
            "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.dataset = load_dataset("parquet",data_files="dataset/SingleModal/train_single_modal.parquet", split="train")
        self.user_feat = load_dataset("parquet",data_files="dataset/user_feat/log-train-00000-of-00001.parquet", split="train")

    def get_note_content(self, note_idx):
        note = self.corpus[note_idx]
        image = self.default_image
        image_path = note['image_path']
        if len(image_path):
            try:
                image_path = os.path.join('afs', image_path[0])
                image = Image.open(image_path)
                image = image.resize((1024, 1024))
                image_size = image.size
                if image_size[0]<=0 or image_size[1]<=0:
                    image = self.default_image
            except Exception as e:
                print(f"Warning: Failed to load image for note {note_idx}: {e}")
            
        return {
            'text': self._get_text_content(note),
            'image': image
        }
    
    def _get_text_content(self, note):
        ret = ''
        if self.use_title:
            ret += note['note_title']
        if self.use_content:
            ret += note['note_content']
        return ret

    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret

    def collate_fn(self, batch):
        # 收集三元组
        pos_query_list, neg_query_list = [], []
        pos_images_list, neg_images_list = [], []
        # 收集原始特征值
        pos_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        neg_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        batch_features = {}
        # 收集用户特征值
        pos_user_feat_vals = {feat: [] for feat in self.all_feat}
        neg_user_feat_vals = {feat: [] for feat in self.all_feat}
        user_feat_vals = {feat: [] for feat in self.all_feat}

        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            user_idx = item["user_idx"]

            # positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            # 正样本池：click==1，按 page_time 降序
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )[:10]  # 添加切片操作，截断至最多10个

            # 可能有多个正例
            # assert len(pos_pool) > 0, 'No positive samples found for query: ' + query
            if len(pos_pool)==0:
                print(f"No positive samples found for query: {query}")
                continue

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # 如果启用最近点击的笔记图像，则处理这些图像并拼接
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            # 对每个正样本，构造若干负样本
            for pos in pos_pool:
                # positive_idx = random.choice(positives)
                pos_idx = pos['note_idx']
                note_content = self.get_note_content(pos_idx)
                
                negatives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 0]
                if len(negatives) < self.negative_samples:
                    additional_samples = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                    negatives.extend(additional_samples)
                else:
                    negatives = random.sample(negatives, k=self.negative_samples)
                
                for neg_idx in negatives:
                    # Template of conversation
                    # 正样本
                    conversation = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                        ]
                    }]
                
                    pos_query_list.append(conversation)
                    pos_images_list.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])

                    # 负样本
                    note_content = self.get_note_content(neg_idx)
                    conversation = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                        ]
                    }]
                    
                    neg_query_list.append(conversation)
                    neg_images_list.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                    
                    #其他特征
                    for feat, thresholds in self.feature_max_values.items():
                        raw_value_pos = self.get_note_feat(feat, pos_idx)
                        # print(f"binary_vec_pos:{binary_vec_pos}")
                        # binary_vec_pos实际上是一个张量
                        binary_vec_pos = torch.tensor(raw_value_pos,dtype=torch.float16)
                        pos_batch_feat_vals[feat].append(binary_vec_pos)

                        raw_value_neg = self.get_note_feat(feat, neg_idx)
                        binary_vec_neg = torch.tensor(raw_value_neg,dtype=torch.float16)
                        neg_batch_feat_vals[feat].append(binary_vec_neg)

                    for feat, thresholds in self.all_feat.items():
                        feat_value = self.user_feat[user_idx][feat] #数值型或者字符串
                        feat_value = torch.tensor(feat_value, dtype=torch.float16)
                        pos_user_feat_vals[feat].append(feat_value)
                        neg_user_feat_vals[feat].append(feat_value)

        for feat, thresholds in self.feature_max_values.items():
            batch_features[feat]= pos_batch_feat_vals[feat] + neg_batch_feat_vals[feat]
        for feat, thresholds in self.all_feat.items():
            user_feat_vals[feat]= pos_user_feat_vals[feat] + neg_user_feat_vals[feat]

        pos_text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in pos_query_list]
        inp_pos = self.processor(
            text=pos_text_prompts,
            images=pos_images_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        neg_text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in neg_query_list]
        inp_neg = self.processor(
            text=neg_text_prompts,
            images=neg_images_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # print(f"inp_pos.shape:{inp_pos.input_ids.shape}")
        # print(f"inp_neg.shape:{inp_neg.input_ids.shape}")       
        return {
            "inp_pos": inp_pos,
            "inp_neg": inp_neg,
            "features": batch_features,
            "user_feat": user_feat_vals
        }

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True
        )


@register_class
class VLMCrossEncoderTestDataProcessor(VLMCrossEncoderTrainingDataProcessor):
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        self.max_length = kwargs.get('max_length', 1024)
        self.num_machines = kwargs.get('num_machines', 0)
        self.machine_rank = kwargs.get('machine_rank', 0)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        
        processor_name = kwargs.get('tokenizer_name_or_path')
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels= 100 * 28 * 28,
            trust_remote_code=True,
            use_fast=True
        )
        
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        self.dataset = self.load_data()
        self.default_image = self._create_default_image()

        self.feature_max_values = {
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

        self.all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}        

    def load_data(self):
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
            "dataset/notes/log-train-00001-of-00005.parquet",
            "dataset/notes/log-train-00002-of-00005.parquet",
            "dataset/notes/log-train-00003-of-00005.parquet",
            "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.user_feat = load_dataset("parquet",data_files="dataset/user_feat/log-train-00000-of-00001.parquet", split="train")
        data = load_dataset("parquet", data_files ="dataset/search_test/train-00000-of-00001.parquet", split="train")
        data = data.select(range(min(self.sample_num, len(data))))
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data
    
    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret
    
    def collate_fn(self, batch):
        queries = []
        images = []
        note_idxs = []
        search_idxs = []
        # 收集原始特征值
        batch_features = {feat: [] for feat in self.feature_max_values}
        user_feat_vals = {feat: [] for feat in self.all_feat}
        # 标记正负样本用于计算 AUC
        Pos_Neg=[]
        
        for item in batch:
            query = item["query"]
            user_idx = item["user_idx"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            search_result_details_with_idx = item["search_result_details_with_idx"]

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            for candidate in candidates:
                # 每一个 query 都会有若干个结果需要和其算相似性
                note_idx = int(candidate[0])
                note_content = self.get_note_content(note_idx)
                
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                    ]
                }]
                
                queries.append(conversation)
                images.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)
                # 记录当前自然结果是否被点击
                for note in search_result_details_with_idx:
                    if note["note_idx"]==  note_idx:
                        pos_or_neg= note["click"]
                Pos_Neg.append(int(pos_or_neg))

                for feat, thresholds in self.feature_max_values.items():
                    raw_value = self.get_note_feat(feat, note_idx)
                    binary_vec = torch.tensor(raw_value,dtype=torch.float16)
                    # print(f"binary_vec_pos:{binary_vec_pos}")
                    # binary_vec_pos实际上是一个张量
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] #数值型或者字符串
                    feat_value = torch.tensor(feat_value, dtype=torch.float16)
                    user_feat_vals[feat].append(feat_value)

        text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in queries]
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # note_idxs记录了所有 batch 内，每一个query 对应的曝光的笔记数量总的note_idx
        return {
            "inputs": inputs, 
            "note_idxs": note_idxs, 
            "search_idxs": search_idxs,
            "features": batch_features, 
            "Pos_Neg": Pos_Neg,
            "user_feat": user_feat_vals
            }
    
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


@register_class
class MultiModalTrainingDataProcessor_listwise:
    def __init__(self, **kwargs):
        data_path = kwargs.get('dataset_name_or_path')
        processor_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.max_length = kwargs.get('max_length', 1024)
        self.negative_samples = kwargs.get('negative_samples', 3)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        print(f"use_recent_clicked_note_images:{self.use_recent_clicked_note_images}")
        self.processor_name = processor_name
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels= 100 * 28 * 28,
            trust_remote_code=True,
            use_fast=True
        )
        self.default_image = self._create_default_image()

        self.feature_max_values = {
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

        self.all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}
        
        with open("dataset/ProcessedDataset/MultiModal/multimodal_train_modal_index.json") as g:
            self.model_index = json.load(g)

    def _create_default_image(self):
        # create a default image with white color
        default_image = Image.new('RGB', (728, 728), color='white')
        return default_image
    
    def load_data(self):
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
            "dataset/notes/log-train-00001-of-00005.parquet",
            "dataset/notes/log-train-00002-of-00005.parquet",
            "dataset/notes/log-train-00003-of-00005.parquet",
            "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.dataset = load_dataset("parquet",data_files="dataset/ProcessedDataset/MultiModal/Multimodal_train.parquet", split="train")
        self.user_feat = load_dataset("parquet",data_files="dataset/user_feat/log-train-00000-of-00001.parquet", split="train")


    def get_note_content(self, note_idx, modal):
        if modal==1:
            # 图像模态
            note = self.corpus[note_idx]
            image = self.default_image
            image_path = note['image_path']
            if len(image_path):
                try:
                    image_path = os.path.join('afs', image_path[0])
                    image = Image.open(image_path)
                    image = image.resize((728, 728))
                    image_size = image.size
                    if image_size[0]<=0 or image_size[1]<=0:
                        image = self.default_image
                except Exception as e:
                    print(f"Warning: Failed to load image for note {note_idx}: {e}")
                
            return {
                'text': self._get_text_content(note),
                'image': image
            }
        elif modal==0:
            note = self.corpus[note_idx]
            # 文本模态
            ret = ''
            ret += note['note_title']
            ret += note['note_content']
            image = self.default_image
            return {
                'text': ret,
                'image': image
            }
        else:
            return None
    
    def _get_text_content(self, note):
        ret = ''
        if self.use_title:
            ret += note['note_title']
        return ret

    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret

    def collate_fn(self, batch):
        # 收集三元组
        query_list = []
        images_list = []
        labels = []
        # 收集原始特征值
        batch_features = {feat: [] for feat in self.feature_max_values}
        # # 收集用户特征值
        user_feat_vals = {feat: [] for feat in self.all_feat}

        # collect_strat_time = time.time()
        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            user_idx = item["user_idx"]
            search_idx = item["search_idx"]
            #listwise 输入，输入是一个列表，输出一个分数
            # 输入：用户query+<用户特征>+候选列表[结果1,结果2,...,结果n]+[结果统计特征]+位置编码(应该不需要加位置编码)
            # 正样本池：click==1，按 page_time 降序
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )
            pos_pool = pos_pool[:20] # 添加切片操作，截断至最多20个
            sub_pos_pool=[d for d in impression_result_details if (int(d['click'])==1 and pd.isna(d['page_time']))][:5]

            # 计算正样本总数
            num_pos = len(pos_pool) + len(sub_pos_pool)
            # 可能有多个正例
            if num_pos == 0:
                print(f"No positive samples found for query: {query}")
                continue

            # 负样本池
            neg_pool = [d for d in impression_result_details if int(d['click'])==0 ][:10]  # 添加切片操作，截断至最多10个

            total_pool = pos_pool + sub_pos_pool + neg_pool
            # 随机打乱顺序
            random.shuffle(total_pool)

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # 如果启用最近点击的笔记图像，则处理这些图像并拼接
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            for i, note in enumerate(total_pool):
                note_idx = note['note_idx']
                # print(type(note_idx))
                modal = self.model_index[str(search_idx)][str(note_idx)]["modal"]
                # modal==0 对应 文本模态
                # modal==1 对应 图像模态
                note_content = self.get_note_content(note_idx,modal)
                label = self.model_index[str(search_idx)][str(note_idx)]["position"]
                
                # Template of conversation
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                    ]
                }]

                query_list.append(conversation)
                images_list.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                labels.append(label)
                # 文本模态 只有query和纯文字title+内容+空白图 图像模态：query+title+图像
                
                #其他特征
                for feat, thresholds in self.feature_max_values.items():
                    note_features = self.get_note_feat(feat, note_idx)
                    # print(f"binary_vec_pos:{binary_vec_pos}")
                    # binary_vec_pos实际上是一个张量
                    binary_vec = torch.tensor(note_features,dtype=torch.float16)
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] #数值型或者字符串
                    feat_value = torch.tensor(feat_value, dtype=torch.float16)
                    user_feat_vals[feat].append(feat_value)

        # position 从 0 开始，依次是点击时长倒排，点击没有时长，没有点击, 由于 labels可能是不连续整数，这里负责映射回连续整数
        if len(labels) != len(set(labels)) and len(labels) > 0:
            print(f"labels:{labels}")
            raise ValueError("列表中存在重复的整数")
        sorted_labels = sorted(labels)
        mapping = {x: i for i, x in enumerate(sorted_labels)}
        labels = [mapping[x] for x in labels]

        text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in query_list]

        inputs = self.processor(
            text=text_prompts,
            images=images_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # print(f"inputs.shape:{inputs.input_ids.shape}")
        # inputs.shape:torch.Size([4, 512])
        # print(f"labels.shape:{len(labels)}")
        # labels.shape:4   
        return {
            "inputs": inputs,
            "labels": labels,
            "batch_features": batch_features,
            "user_feat": user_feat_vals
        }

    # shuffle=True行为：在每个epoch开始时，DataLoader会将整个数据集随机打乱（全局洗牌），影响batch之间的顺序
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True
        )


@register_class
class MultiModalTestDataProcessor(MultiModalTrainingDataProcessor_listwise):
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        self.max_length = kwargs.get('max_length', 1024)
        self.num_machines = kwargs.get('num_machines', 0)
        self.machine_rank = kwargs.get('machine_rank', 0)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        
        processor_name = kwargs.get('tokenizer_name_or_path')
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels= 100 * 28 * 28,
            trust_remote_code=True,
            use_fast=True
        )
        
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        self.dataset = self.load_data()
        self.default_image = self._create_default_image()

        with open("dataset/ProcessedDataset/MultiModal/multimodal_test_modal_index.json") as g:
            self.labels = json.load(g)
        
        self.feature_max_values = {
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

        self.all_feat={"gender":"unknown", "age":"unknown", 'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}        

    def load_data(self):
        file_paths = ["dataset/notes/log-train-00000-of-00005.parquet",
            "dataset/notes/log-train-00001-of-00005.parquet",
            "dataset/notes/log-train-00002-of-00005.parquet",
            "dataset/notes/log-train-00003-of-00005.parquet",
            "dataset/notes/log-train-00004-of-00005.parquet"]
        self.corpus = load_dataset("parquet", data_files=file_paths,split="train")
        self.user_feat = load_dataset("parquet",data_files="dataset/user_feat/log-train-00000-of-00001.parquet", split="train")
        data = load_dataset("parquet", data_files ="dataset/ProcessedDataset/MultiModal/Multimodal_test.parquet", split="train")
        data = data.select(range(min(self.sample_num, len(data))))
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data
    
    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret
    
    def collate_fn(self, batch):
        query_list = []
        images_list = []
        labels = []
        note_idxs = []
        search_idxs = []
        # 收集原始特征值
        batch_features = {feat: [] for feat in self.feature_max_values}
        user_feat_vals = {feat: [] for feat in self.all_feat}
        # 标记正负样本用于计算 AUC
        Pos_Neg=[]
        
        for item in batch:
            query = item["query"]
            user_idx = item["user_idx"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            search_result_details_with_idx = item["search_result_details_with_idx"]

            # 正样本池：click==1，按 page_time 降序
            pos_pool = [d for d in search_result_details_with_idx if int(d['click'])==1]
            # 负样本池
            neg_pool = [d for d in search_result_details_with_idx if int(d['click'])==0 ]
            total_pool = pos_pool+neg_pool
            
            # 随机打乱顺序
            random.shuffle(total_pool)

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # 如果启用最近点击的笔记图像，则处理这些图像并拼接
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            for i, note in enumerate(total_pool):
                note_idx = note['note_idx']
                # print(type(note_idx))
                modal = self.labels[str(search_idx)][str(note_idx)]["modal"]
                # modal==0 对应 文本模态
                # modal==1 对应 图像模态
                note_content = self.get_note_content(note_idx,modal)
                label = self.labels[str(search_idx)][str(note_idx)]["position"]
                
                # Template of conversation
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                    ]
                }]

                query_list.append(conversation)
                images_list.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                labels.append(label)
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)
                for note in search_result_details_with_idx:
                    if note["note_idx"]==  note_idx:
                        pos_or_neg= note["click"]
                Pos_Neg.append(int(pos_or_neg))
                # Pos_Neg的顺序是打乱的顺序
                
                for feat, thresholds in self.feature_max_values.items():
                    raw_value = self.get_note_feat(feat, note_idx)
                    binary_vec = torch.tensor(raw_value,dtype=torch.float16)
                    # print(f"binary_vec_pos:{binary_vec_pos}")
                    # binary_vec_pos实际上是一个张量
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] #数值型或者字符串
                    feat_value = torch.tensor(feat_value, dtype=torch.float16)
                    user_feat_vals[feat].append(feat_value)

        text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in query_list]
        inputs = self.processor(
            text=text_prompts,
            images=images_list,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {"inputs": inputs, 
                "note_idxs": note_idxs, 
                "search_idxs": search_idxs, 
                "Pos_Neg": Pos_Neg,
                "labels":labels,
                "batch_features": batch_features,
                "user_feat": user_feat_vals
                }
    
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
