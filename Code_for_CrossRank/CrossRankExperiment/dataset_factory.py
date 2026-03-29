import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
# from utils import *
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from registry import register_class
from datasets import DatasetDict
import time
import math


@register_class
class CrossRankMultiModalTrainingProcessor:
    def __init__(self, **kwargs):
        data_path = kwargs.get('dataset_name_or_path')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        self.max_length = kwargs.get('max_length', 1024)
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        # Get the value corresponding to the key 'negative_pool' from the kwargs dictionary. If the key does not exist, use the default value 'search_result_details_with_idx'
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.query_features=['query_feat_5','query_feat_3',"query_feat_1"]
        self.statistic_features=["candidate_feat_1","candidate_feat_2","candidate_feat_3","candidate_feat_4","candidate_feat_5","upstream_label","query_feat_2","query_feat_4"]

    
    def load_data(self):
        self.corpus = load_dataset("parquet",data_files="dataset/CrossRankData/MultiModalCorpus.parquet", split="train")
        self.corpus_2 = load_dataset("parquet",data_files="dataset/CrossRankData/SingleModalCorpus.parquet", split="train")
        self.dataset = load_dataset("parquet",data_files="dataset/CrossRankData/Multimodal_Train.parquet", split="train")


    def get_candidate_content(self,search_idx, candidate_idx, modal):
        if modal==1:
            # 1 Queue
            # search_idx between 0 and 24009 belongs to mixed modality annotation data
            # Otherwise, it belongs to single modality annotation data
            if search_idx <= 24009:
                candidate = self.corpus[candidate_idx]
            else:
                candidate = self.corpus_2[candidate_idx]
            ret = ''
            if candidate['query_intent'] is not None:
                ret += f"Query Intent:{candidate['query_intent']}."
            else:
                # print(f"No query_intent found for candidate: {candidate_idx}")
                pass

            if candidate['real_title'] is not None:
                ret += f"Document Title:{candidate['real_title']}."
            else:
                # print(f"No Title found for candidate: {candidate_idx}")
                pass

            if candidate['abs_basic'] is not None:
                ret += f"Document Abstract:{candidate['abs_basic']}."
            else:
                # print(f"No abstract found for candidate: {candidate_idx}")
                pass

            if candidate['title_highlight_basic'] is not None:
                ret += f"Document Heightlight text:{candidate['title_highlight_basic']}."
            else:
                # print(f"No heightlight_basic found for candidate: {candidate_idx}")
                pass

            return {
                'text': ret,
            }

        elif modal==0:
            # 0 Queue
            if search_idx <= 24009:
                candidate = self.corpus[candidate_idx]
            else:
                candidate = self.corpus_2[candidate_idx]
            ret = ''
            if candidate['query_intent'] is not None:
                ret += f"Query Intent:{candidate['query_intent']}."
            else:
                # print(f"No query_intent found for candidate: {candidate_idx}")
                pass

            if candidate['real_title'] is not None:
                ret += f"Document Title:{candidate['real_title']}."
            else:
                # print(f"No Title found for candidate: {candidate_idx}")
                pass

            if candidate['abs_basic'] is not None:
                ret += f"Document Abstract:{candidate['abs_basic']}."
            else:
                # print(f"No abstract found for candidate: {candidate_idx}")
                pass

            if candidate['title_highlight_basic'] is not None:
                ret += f"Document Heightlight text:{candidate['title_highlight_basic']}."
            else:
                # print(f"No heightlight_basic found for candidate: {candidate_idx}")
                pass

            return {
                'text': ret,
            }

        else:
            raise ValueError("Wrong modal")
    
    def get_cand_feat(self,feat, candidate_idx , search_idx):
        if search_idx <= 24009:
            ret = self.corpus[candidate_idx][f'{feat}']
        else:
            ret = self.corpus_2[candidate_idx][f'{feat}']
        return ret

    def collate_fn(self, batch):
        # Collect triples
        query_candidate_list = []
        labels = []
        search_idxs=[]
        candidate_idxs=[]
        modal_indexs= []
        # Collect original feature values
        batch_query_features = {feat: [] for feat in self.query_features}
        batch_statistic_features = {feat: [] for feat in self.statistic_features}

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. <|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        task = 'Given a web search query, judge the extent to which the Document meets the requirements specified by the Query and the Instruct. Assign a score from 0 to 4, where:\n- 0 means the document completely fails to meet the requirements,\n- 1 means the document barely meets the requirements,\n- 2 means the document partially meets the requirements,\n- 3 means the document largely meets the requirements,\n- 4 means the document fully and perfectly meets the requirements.\nYour response should contain only the score (a single integer from 0 to 4) and nothing else.'

        # collect_strat_time = time.time()
        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            search_idx = item["search_idx"]

            # listwise input, input is a list, output a score
            # Input: user query + <user features> + candidate list [result1, result2, ..., resultn] + [result statistical features]

            # Positive sample pool: click == 1, sorted in descending order by page_time
            candidate_pool = [d for d in impression_result_details]

            # Calculate total number of positive samples
            num_candidate = len(candidate_pool)
            # Multiple positive examples may exist
            if num_candidate == 0:
                raise ValueError("No candidate samples found")

            # Random shuffle order
            random.shuffle(candidate_pool)

            # Local label mapping (labels are consecutive and unique within each query)
            local_labels_raw = [cand["position"] for cand in candidate_pool]

            # Check for duplicates within the same search_idx (optional)
            if len(local_labels_raw) != len(set(local_labels_raw)):
                raise ValueError(f"search_idx={search_idx} 内 label 存在重复: {local_labels_raw}")


            for candidate in candidate_pool:
                search_idxs.append(search_idx)
                # candidate form: {'upstream_label': 0.98092, 'candidate_idx': 2, 'candidate_feat_5': 0.0, 'candidate_feat_4': 0.0, 'label': 3.0, 'position': 0}
                candidate_idx = candidate['candidate_idx']
                candidate_idxs.append(candidate_idx)
                if search_idx <= 24009:
                    candidate_item = self.corpus[candidate_idx]
                else:
                    candidate_item = self.corpus_2[candidate_idx]

                if candidate_item['modal']=='0':
                    modal=0
                elif candidate_item['modal']=='1':
                    modal=1
                else:
                    raise ValueError("Wrong modal")
                modal_indexs.append(modal)

                candidate_content = self.get_candidate_content(search_idx,candidate_idx, modal)
                query_candidate_list.append(f"{prefix}<Instruct>:{task}\n<Query>:{query}\n<Document>: {candidate_content}{suffix}")

                label = candidate["position"]
                labels.append(label)

                # Query side features
                for feat in self.query_features:
                    assert len(self.query_features)==len(batch_query_features),"query_features 和 batch_query_features 长度不一致"
                    cand_features = self.get_cand_feat(feat, candidate_idx,search_idx)
                    # Convert cand_features to integer
                    cand_features = int(cand_features)
                    # Convert to binary
                    if feat == "query_feat_5":
                        total_bits= 2
                    elif feat == "query_feat_3":
                        total_bits= 4
                    elif feat == "query_feat_1":
                        total_bits= 4
                    binary_str = bin(cand_features)[2:].zfill(total_bits)
                    binary_list = [int(b) for b in binary_str]
                    cand_features = torch.tensor(binary_list, dtype=torch.float16) 

                    batch_query_features[feat].append(cand_features)

                # Candidate side features
                for feat in self.statistic_features:
                    cand_features = self.get_cand_feat(feat, candidate_idx,search_idx)
                    assert len(self.statistic_features)==len(batch_statistic_features),"statistic_features 和 batch_statistic_features 长度不一致"
                    if feat in ["candidate_feat_5","candidate_feat_1","upstream_label"]:
                        cand_features = math.ceil(cand_features * 10)
                        binary_str = bin(cand_features)[2:].zfill(4)
                        binary_list = [int(b) for b in binary_str]
                        cand_features = torch.tensor(binary_list, dtype=torch.float16) 
                        batch_statistic_features[feat].append(cand_features)
                    else:
                        # Convert cand_features to integer
                        cand_features = int(cand_features)
                        # Convert to binary
                        if feat in ["candidate_feat_4","candidate_feat_3","candidate_feat_2","query_feat_2"]:
                            total_bits= 4
                        elif feat in ["query_feat_4"]:
                            total_bits= 2
                        binary_str = bin(cand_features)[2:].zfill(total_bits)
                        binary_list = [int(b) for b in binary_str]
                        cand_features = torch.tensor(binary_list, dtype=torch.float16) 
                        batch_statistic_features[feat].append(cand_features)

        inputs = self.tokenizer(
            query_candidate_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # print(f"inputs['input_ids'].shape:{inputs['input_ids'].shape}")
        # inputs['input_ids'].shape: torch.Size([6, 512])
        # candidate_idxs.shape:6
        # candidate_idxs:[35084, 35083, 35085, 8726, 8725, 8724]
        # search_idxs.shape:6
        # search_idxs:[9465, 9465, 9465, 9266, 9266, 9266]
        # modal_indexs.shape:6
        # print(f"labels:{labels}")
        #labels.shape:6
        return {
            "inputs": inputs,
            "labels": labels,
            "candidate_idxs":candidate_idxs,
            "search_idxs":search_idxs,
            "modal_indexs":modal_indexs,
            "batch_query_features": batch_query_features,
            "batch_statistic_features": batch_statistic_features,
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
class CrossRankMultiModalTestProcessor(CrossRankMultiModalTrainingProcessor):
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        self.max_length = kwargs.get('max_length', 1024)
        self.num_machines = kwargs.get('num_machines', 0)
        self.machine_rank = kwargs.get('machine_rank', 0)
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        self.local_rank = local_rank
        self.num_processes = num_processes
        self.dataset = self.load_data()
        self.query_features=['query_feat_5','query_feat_3',"query_feat_1"]
        self.statistic_features=["candidate_feat_1","candidate_feat_2","candidate_feat_3","candidate_feat_4","candidate_feat_5","upstream_label","query_feat_2","query_feat_4"]
     

    def load_data(self):
        self.corpus = load_dataset("parquet",data_files="dataset/CrossRankData/MultiModalCorpus.parquet", split="train")
        # Test set
        data = load_dataset("parquet",data_files="dataset/CrossRankData/Multimodal_Test.parquet", split="train")
        # Evaluation set
        # data = load_dataset("parquet",data_files="dataset/CrossRankData/Multimodal_Eval.parquet", split="train")
        data = data.select(range(min(self.sample_num, len(data))))
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data
    
    def get_note_feat(self,feat, note_idx):
        ret = 0
        ret = self.corpus[note_idx][f'{feat}']
        return ret
    
    def collate_fn(self, batch):
        query_list = []
        labels = []
        candidate_idxs = []
        search_idxs = []
        modal_indexs = []
        query_candidate_list = []
        # Collect original feature values
        batch_query_features = {feat: [] for feat in self.query_features}
        batch_statistic_features = {feat: [] for feat in self.statistic_features}

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. <|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        task = 'Given a web search query, judge the extent to which the Document meets the requirements specified by the Query and the Instruct. Assign a score from 0 to 4, where:\n- 0 means the document completely fails to meet the requirements,\n- 1 means the document barely meets the requirements,\n- 2 means the document partially meets the requirements,\n- 3 means the document largely meets the requirements,\n- 4 means the document fully and perfectly meets the requirements.\nYour response should contain only the score (a single integer from 0 to 4) and nothing else.'

        for item in batch:
            query = item["query"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            search_result_details_with_idx = item["search_result_details_with_idx"]

            # Random shuffle order
            random.shuffle(search_result_details_with_idx)

            for i, candidate in enumerate(search_result_details_with_idx):
                candidate_idx = candidate['candidate_idx']
                candidate_item = self.corpus[candidate_idx]

                if candidate_item['modal']=='0':
                    modal=0
                elif candidate_item['modal']=='1':
                    modal=1
                else:
                    raise ValueError("Wrong modal")
                modal_indexs.append(modal)

                candidate_content = self.get_candidate_content(search_idx,candidate_idx, modal)
                query_candidate_list.append(f"{prefix}<Instruct>:{task}\n<Query>:{query}\n<Document>: {candidate_content}{suffix}")

                label = candidate["position"]
                labels.append(label)

                candidate_idxs.append(candidate_idx)
                search_idxs.append(search_idx)

                # Query side features
                for feat in self.query_features:
                    assert len(self.query_features)==len(batch_query_features),"query_features 和 batch_query_features 长度不一致"
                    cand_features = self.get_cand_feat(feat, candidate_idx,search_idx)
                    # Convert cand_features to integer
                    cand_features = int(cand_features)
                    # Convert to binary
                    if feat == "query_feat_5":
                        total_bits= 2
                    elif feat == "query_feat_3":
                        total_bits= 4
                    elif feat == "query_feat_1":
                        total_bits= 4
                    binary_str = bin(cand_features)[2:].zfill(total_bits)
                    binary_list = [int(b) for b in binary_str]
                    cand_features = torch.tensor(binary_list, dtype=torch.float16) 

                    batch_query_features[feat].append(cand_features)

                # Candidate side features
                for feat in self.statistic_features:
                    cand_features = self.get_cand_feat(feat, candidate_idx,search_idx)
                    assert len(self.statistic_features)==len(batch_statistic_features),"statistic_features 和 batch_statistic_features 长度不一致"
                    if feat in ["candidate_feat_5","candidate_feat_1","upstream_label"]:
                        cand_features = math.ceil(cand_features * 10)
                        binary_str = bin(cand_features)[2:].zfill(4)
                        binary_list = [int(b) for b in binary_str]
                        cand_features = torch.tensor(binary_list, dtype=torch.float16) 
                        batch_statistic_features[feat].append(cand_features)
                    else:
                        # Convert cand_features to integer
                        cand_features = int(cand_features)
                        # Convert to binary
                        if feat in ["candidate_feat_4","candidate_feat_3","candidate_feat_2","query_feat_2"]:
                            total_bits= 4
                        elif feat in ["query_feat_4"]:
                            total_bits= 2
                        binary_str = bin(cand_features)[2:].zfill(total_bits)
                        binary_list = [int(b) for b in binary_str]
                        cand_features = torch.tensor(binary_list, dtype=torch.float16) 
                        batch_statistic_features[feat].append(cand_features)


        inputs = self.tokenizer(
            query_candidate_list,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "inputs": inputs, 
            "candidate_idxs": candidate_idxs, 
            "search_idxs": search_idxs, 
            "labels":labels,
            "batch_query_features": batch_query_features,
            "batch_statistic_features": batch_statistic_features
            }
