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
        # Negative sample pool
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
        print(f"The current dataset is: {self.data_path}")
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
        # Construct positive and negative samples
        queries = []
        notes = []
        labels = []

        for item in batch:
            query = item["query"]
            search_idx = item["search_idx"]
            # impression_result_details is a list, each element is a dictionary containing the index of the note clicked by the user and the click label
            # impression_result_details is search_result_details_with_idx of the train dataset
            impression_result_details = item[self.negative_pool]
            # Get positive samples
            positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            assert len(positives) > 0, 'No positive samples found for query: ' + query
            # Randomly select one positive sample
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

        # query_note_pairs is a list, each element is a string in the format [query] [SEP] [document content]
        query_note_pairs = [f"{q} [SEP] {n}" for q, n in zip(queries, notes)]
        
        inputs = self.tokenizer(
            query_note_pairs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = torch.tensor(labels, dtype=torch.float)
        # Generate 1 positive sample pair and N negative sample pairs for each query (N = negative_samples).
        # The sample pair format is [query] [SEP] [document content], distinguished by the separator [SEP]

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
        # Negative sample pool
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
        print(f"The current dataset is: {self.data_path}")
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
        # Collect triplets
        q_pos_list, q_neg_list = [], []
        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]

            # Positive sample pool: click==1, sorted in descending order by page_time
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )[:10]  # Add slicing to truncate to max 10 items

            # For each positive sample, construct several negative samples
            for pos in pos_pool:
                pos_time = pos['page_time']
                pos_idx = pos['note_idx']

                # Negative sample candidates: either not clicked, or clicked but page_time is less than current positive sample
                neg_cands = [
                    d for d in impression_result_details
                    if (int(d['click']) == 0)
                    or (int(d['click']) == 1 and (not pd.isna(d['page_time'])) and d.get('page_time', 0) < pos_time)
                ]
                # If insufficient, randomly supplement from the entire corpus
                if len(neg_cands) < self.negative_samples:
                    # Number of samples to supplement
                    k = self.negative_samples - len(neg_cands)
                    # Randomly select k indices from the entire corpus as negative samples
                    extra_idxs = random.sample(range(len(self.corpus)), k=k)
                    # Existing neg_cands are dicts, first get their note_idx
                    neg_idxs = [d['note_idx'] for d in neg_cands] + extra_idxs
                else:
                    neg_idxs = random.sample([d['note_idx'] for d in neg_cands], self.negative_samples)

                # Record (q, d+) and each (q, d-) separately
                pos_text = self.get_note_content(pos_idx)
                for neg_idx in neg_idxs:
                    neg_text = self.get_note_content(neg_idx)
                    q_pos_list.append(f"{query} [SEP] {pos_text}")
                    q_neg_list.append(f"{query} [SEP] {neg_text}")

        # Encode positive and negative pairs separately
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

        # In typical pair-wise training (e.g., using Hinge Loss or Logistic Loss), we don't need an additional labels tensor for each positive-negative pair. Instead, the order of positive-negative pairs itself contains the supervision signal
        return {"inp_pos": inp_pos, "inp_neg": inp_neg}

    def collate_fn(self, batch):
        # Collect triplets
        q_pos_list, q_neg_list = [], []
        # Collect raw feature values
        pos_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        neg_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        batch_features = {}
        # Collect user feature values
        pos_user_feat_vals = {feat: [] for feat in self.all_feat}
        neg_user_feat_vals = {feat: [] for feat in self.all_feat}
        user_feat_vals = {feat: [] for feat in self.all_feat}

        for item in batch:
            query = item["query"]
            search_idx = item["search_idx"]
            impression_result_details = item[self.negative_pool]
            user_idx = item["user_idx"]

            # Positive sample pool: click==1, sorted in descending order by page_time
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )[:10]  # Add slicing to truncate to max 10 items

            # For each positive sample, construct several negative samples
            for pos in pos_pool:
                pos_time = pos['page_time']
                pos_idx = pos['note_idx']


                # Negative sample candidates: either not clicked
                neg_cands = [
                    d for d in impression_result_details
                    if (int(d['click']) == 0)
                ]
                # If insufficient, randomly supplement from the entire corpus
                if len(neg_cands) < self.negative_samples:
                    # Number of samples to supplement
                    k = self.negative_samples - len(neg_cands)
                    # Randomly select k indices from the entire corpus as negative samples
                    extra_idxs = random.sample(range(len(self.corpus)), k=k)
                    # Existing neg_cands are dicts, first get their note_idx
                    neg_idxs = [d['note_idx'] for d in neg_cands] + extra_idxs
                else:
                    neg_idxs = random.sample([d['note_idx'] for d in neg_cands], self.negative_samples)

                # Record (q, d+) and each (q, d-) separately
                pos_text = self.get_note_content(pos_idx)
                for neg_idx in neg_idxs:
                    neg_text = self.get_note_content(neg_idx)
                    q_pos_list.append(f"{query} [SEP] {pos_text}")
                    q_neg_list.append(f"{query} [SEP] {neg_text}")

                    for feat, thresholds in self.feature_max_values.items():
                        raw_value_pos = self.get_note_feat(feat, pos_idx)
                        # binary_vec_pos is actually a tensor
                        binary_vec_pos = torch.tensor(raw_value_pos,dtype=torch.float32)
                        pos_batch_feat_vals[feat].append(binary_vec_pos)

                        raw_value_neg = self.get_note_feat(feat, neg_idx)
                        binary_vec_neg = torch.tensor(raw_value_neg,dtype=torch.float32)
                        neg_batch_feat_vals[feat].append(binary_vec_neg)

                    for feat, thresholds in self.all_feat.items():
                        feat_value = self.user_feat[user_idx][feat] # Numeric or string
                        feat_value = torch.tensor(feat_value, dtype=torch.float32)
                        pos_user_feat_vals[feat].append(feat_value)
                        neg_user_feat_vals[feat].append(feat_value)


        for feat, thresholds in self.feature_max_values.items():
            batch_features[feat]=  pos_batch_feat_vals[feat] + neg_batch_feat_vals[feat]
        for feat, thresholds in self.all_feat.items():
            user_feat_vals[feat]=  pos_user_feat_vals[feat] + neg_user_feat_vals[feat]
        
        # Convert to tensor and move to device
        # batch_features = {
        #     feat: [torch.tensor(val, dtype=torch.long) for val in vals]
        #     for feat, vals in batch_features.items()
        # }
        
        # Encode positive and negative pairs separately
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
        # Evenly split the data into num_processes shards, each process (GPU) handles one shard
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
        # Collect raw feature values
        batch_features = {feat: [] for feat in self.feature_max_values}
        user_feat_vals = {feat: [] for feat in self.all_feat}
        # Mark positive and negative samples for AUC calculation
        Pos_Neg=[]

        for item in batch:
            query = item["query"]
            user_idx = item["user_idx"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            # candidates are the exposure results of xhs, a list of note_idx, default to 0.0 if no score
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            # Sort in descending order by score, sort defaults to ascending, reverse=True means descending
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
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] # Numeric or string
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
        # Test set input: user query, corresponding note content
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
            # May have multiple positive examples
            assert len(positives) > 0, 'No positive samples found for query: ' + query
            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # If using recently clicked note images, process and concatenate these images
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            positive_idx = random.choice(positives)
            note_content = self.get_note_content(positive_idx)
            # Select one positive example and get its content, including text and image

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
        # Collect triplets
        pos_query_list, neg_query_list = [], []
        pos_images_list, neg_images_list = [], []
        # Collect raw feature values
        pos_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        neg_batch_feat_vals = {feat: [] for feat in self.feature_max_values}
        batch_features = {}
        # Collect user feature values
        pos_user_feat_vals = {feat: [] for feat in self.all_feat}
        neg_user_feat_vals = {feat: [] for feat in self.all_feat}
        user_feat_vals = {feat: [] for feat in self.all_feat}

        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            user_idx = item["user_idx"]

            # Positive sample pool: click==1, sorted in descending order by page_time
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )[:10]  # Add slicing to truncate to max 10 items

            # May have multiple positive examples
            if len(pos_pool)==0:
                print(f"No positive samples found for query: {query}")
                continue

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # If using recently clicked note images, process and concatenate these images
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            # For each positive sample, construct several negative samples
            for pos in pos_pool:
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
                    # Positive sample
                    conversation = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                        ]
                    }]
                
                    pos_query_list.append(conversation)
                    pos_images_list.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])

                    # Negative sample
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
                    
                    # Other features
                    for feat, thresholds in self.feature_max_values.items():
                        raw_value_pos = self.get_note_feat(feat, pos_idx)
                        # binary_vec_pos is actually a tensor
                        binary_vec_pos = torch.tensor(raw_value_pos,dtype=torch.float16)
                        pos_batch_feat_vals[feat].append(binary_vec_pos)

                        raw_value_neg = self.get_note_feat(feat, neg_idx)
                        binary_vec_neg = torch.tensor(raw_value_neg,dtype=torch.float16)
                        neg_batch_feat_vals[feat].append(binary_vec_neg)

                    for feat, thresholds in self.all_feat.items():
                        feat_value = self.user_feat[user_idx][feat] # Numeric or string
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
        # Collect raw feature values
        batch_features = {feat: [] for feat in self.feature_max_values}
        user_feat_vals = {feat: [] for feat in self.all_feat}
        # Mark positive and negative samples for AUC calculation
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
                # Each query will have several results to calculate similarity with
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
                # Record whether the current natural result was clicked
                for note in search_result_details_with_idx:
                    if note["note_idx"]==  note_idx:
                        pos_or_neg= note["click"]
                Pos_Neg.append(int(pos_or_neg))

                for feat, thresholds in self.feature_max_values.items():
                    raw_value = self.get_note_feat(feat, note_idx)
                    binary_vec = torch.tensor(raw_value,dtype=torch.float16)
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] # Numeric or string
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
        
        # note_idxs records the total note_idx of the exposed notes corresponding to each query in all batches
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
            # Image modality
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
            # Text modality
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
        # Collect triplets
        query_list = []
        images_list = []
        labels = []
        # Collect raw feature values
        batch_features = {feat: [] for feat in self.feature_max_values}
        # Collect user feature values
        user_feat_vals = {feat: [] for feat in self.all_feat}

        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            user_idx = item["user_idx"]
            search_idx = item["search_idx"]
            # listwise input, input is a list, output a score
            # Input: user query + <user features> + candidate list [result1, result2,..., resultn] + [result statistical features] + position encoding (should not need to add position encoding)
            # Positive sample pool: click==1, sorted in descending order by page_time
            pos_pool = sorted(
                [d for d in impression_result_details if (int(d['click'])==1 and not pd.isna(d['page_time']))],
                key=lambda x: x['page_time'],
                reverse=True
            )
            pos_pool = pos_pool[:20] # Add slicing to truncate to max 20 items
            sub_pos_pool=[d for d in impression_result_details if (int(d['click'])==1 and pd.isna(d['page_time']))][:5]

            # Calculate total number of positive samples
            num_pos = len(pos_pool) + len(sub_pos_pool)
            # May have multiple positive examples
            if num_pos == 0:
                print(f"No positive samples found for query: {query}")
                continue

            # Negative sample pool
            neg_pool = [d for d in impression_result_details if int(d['click'])==0 ][:10]  # Add slicing to truncate to max 10 items

            total_pool = pos_pool + sub_pos_pool + neg_pool
            # Randomly shuffle the order
            random.shuffle(total_pool)

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # If using recently clicked note images, process and concatenate these images
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            for i, note in enumerate(total_pool):
                note_idx = note['note_idx']
                modal = self.model_index[str(search_idx)][str(note_idx)]["modal"]
                # modal==0 corresponds to text modality
                # modal==1 corresponds to image modality
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
                # Text modality only has query and pure text title+content+blank image Image modality: query+title+image
                
                # Other features
                for feat, thresholds in self.feature_max_values.items():
                    note_features = self.get_note_feat(feat, note_idx)
                    binary_vec = torch.tensor(note_features,dtype=torch.float16)
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] # Numeric or string
                    feat_value = torch.tensor(feat_value, dtype=torch.float16)
                    user_feat_vals[feat].append(feat_value)

        # Position starts from 0, sorted by click duration descending, click without duration, no click. Since labels may be discontinuous integers, map back to continuous integers here
        if len(labels) != len(set(labels)) and len(labels) > 0:
            print(f"labels:{labels}")
            raise ValueError("Duplicate integers exist in the list")
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

        return {
            "inputs": inputs,
            "labels": labels,
            "batch_features": batch_features,
            "user_feat": user_feat_vals
        }

    # shuffle=True behavior: At the beginning of each epoch, DataLoader randomly shuffles the entire dataset (global shuffling), affecting the order between batches
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
        # Collect raw feature values
        batch_features = {feat: [] for feat in self.feature_max_values}
        user_feat_vals = {feat: [] for feat in self.all_feat}
        # Mark positive and negative samples for AUC calculation
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

            # Positive sample pool: click==1, sorted in descending order by page_time
            pos_pool = [d for d in search_result_details_with_idx if int(d['click'])==1]
            # Negative sample pool
            neg_pool = [d for d in search_result_details_with_idx if int(d['click'])==0 ]
            total_pool = pos_pool+neg_pool
            
            # Randomly shuffle the order
            random.shuffle(total_pool)

            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                # If using recently clicked note images, process and concatenate these images
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            for i, note in enumerate(total_pool):
                note_idx = note['note_idx']
                modal = self.labels[str(search_idx)][str(note_idx)]["modal"]
                # modal==0 corresponds to text modality
                # modal==1 corresponds to image modality
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
                # The order of Pos_Neg is the shuffled order
                
                for feat, thresholds in self.feature_max_values.items():
                    raw_value = self.get_note_feat(feat, note_idx)
                    binary_vec = torch.tensor(raw_value,dtype=torch.float16)
                    batch_features[feat].append(binary_vec)

                for feat, thresholds in self.all_feat.items():
                    feat_value = self.user_feat[user_idx][feat] # Numeric or string
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