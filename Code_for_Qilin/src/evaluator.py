import torch
import numpy as np
import os
# from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from utils import *
# import editdistance
# from rank_bm25 import BM25Okapi
from typing import List, Tuple
# import jieba 
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch.nn.functional as F
import math

def load_csv(file_path):
    """
    Load CSV file and return a mapping from qid to pids
    """
    df = pd.read_csv(file_path)
    qid_to_pids = defaultdict(list)
    for _, row in df.iterrows():
        qid_to_pids[int(row["qid"])].append(int(row["pid"]))
    return qid_to_pids

def calculate_metrics(sorted_results, ground_truth, k_list):
    """
    Calculate MRR@k, MAP@k, Recall@k, Precision@k for multiple k values

    Args:
        sorted_results (dict): Ranking results, qid -> [pid1, pid2, ...] (in ranked order)
        ground_truth (dict): Ground truth data, qid -> {pid1, pid2, ...} (set of positive samples)
        k_list (list): List of k values to calculate metrics for, e.g. [1, 3, 5, 10]

    Returns:
        dict: Average metrics, including MRR@k, MAP@k, Recall@k, Precision@k for each k value
    """
    max_k = max(k_list)  # Get maximum k value
    metrics = {k: {"mrr": 0.0, "map_sum": 0.0, "recall": 0.0, "precision": 0.0} for k in k_list}
    num_queries = len(ground_truth)
    valid_queries = 0

    for qid, relevant_pids in ground_truth.items():
        if qid not in sorted_results:
            num_queries -= 1  # Skip if qid not in ranking results
            continue
        
        valid_queries += 1
        # Get the top max_k sorted results for the current query.
        retrieved_pids = sorted_results[qid][:max_k]
        # Check if the k-th retrieved document is a positive sample, i.e., whether qid is in relevant_pids
        hits = [pid in relevant_pids for pid in retrieved_pids]
        # print(hits[:10])
        # Calculate metrics for each k value
        for k in k_list:
            hits_at_k = hits[:k]
            
            # Calculate MRR@k
            # If there are relevant PIDs in the top k results, calculate the reciprocal rank of the first relevant PID and add to mrr
            if any(hits_at_k):
                first_hit_rank = hits_at_k.index(True) + 1  # rank starts from 1
                metrics[k]["mrr"] += 1 / first_hit_rank

            # Calculate MAP@k
            avg_precision = 0.0
            num_correct = 0
            for i, is_relevant in enumerate(hits_at_k):
                if is_relevant:
                    num_correct += 1
                    precision_at_i = num_correct / (i + 1)
                    avg_precision += precision_at_i
            if num_correct > 0:  # Avoid division by zero
                avg_precision /= min(len(relevant_pids), k)  # Use min(|rel|, k) as denominator
                metrics[k]["map_sum"] += avg_precision

            # Calculate Recall@k
            metrics[k]["recall"] += sum(hits_at_k) / len(relevant_pids)

            # Calculate Precision@k
            metrics[k]["precision"] += sum(hits_at_k) / k

    # Return all zeros if no valid queries
    if valid_queries == 0:
        return {f"{metric}@{k}": 0.0 
                for k in k_list 
                for metric in ["MRR", "MAP", "Recall", "Precision"]}

    # Calculate average metrics for all k values
    results = {}
    for k in k_list:
        results[f"MRR@{k}"] = metrics[k]["mrr"] / valid_queries
        results[f"MAP@{k}"] = metrics[k]["map_sum"] / valid_queries
        results[f"Recall@{k}"] = metrics[k]["recall"] / valid_queries
        results[f"Precision@{k}"] = metrics[k]["precision"] / valid_queries

    return results


def calculate_ndcg(sorted_results, ground_truth, k_list=None):
    """
    Calculate NDCG (Normalized Discounted Cumulative Gain) for multiple k values
    
    Args:
        sorted_results (dict): Ranking results, qid -> [pid1, pid2, ...] (descending order)
        ground_truth (dict): Ground truth data, qid -> {pid1, pid2, ...} (set of positive samples)
        k_list (list/None): List of k values to calculate. If None, calculate for all possible k
        
    Returns:
        dict: {
            "NDCG@{k}": average_ndcg,
            ...
            "NDCG": overall_average_ndcg  # Average NDCG including all k values
        }
    """
    ndcg_metrics = {}
    total_ndcg = 0.0
    valid_queries = 0

    # Determine the set of k values to calculate
    if k_list is None:
        # Automatically get all possible k values (number of relevant documents for all queries)
        k_list = set()
        for qid in ground_truth:
            k_list.add(len(ground_truth[qid]))
        k_list = sorted(k_list, reverse=True)  # Sort from largest to smallest
    
    # Preprocess: create a mapping from qid to the number of relevant documents
    qid_rel_counts = {qid: len(pids) for qid, pids in ground_truth.items()}

    for qid, relevant_pids in ground_truth.items():
        # Skip queries without ranking results
        if qid not in sorted_results:
            continue
        
        # Get the number of relevant documents for the current query
        n = qid_rel_counts[qid]
        if n == 0:
            continue  # Do not calculate NDCG when there are no relevant documents
        
        valid_queries += 1
        
        # Get the top n predicted results
        predicted_pids = sorted_results[qid][:n]
        
        # Calculate DCG
        dcg = 0.0
        for i, pid in enumerate(predicted_pids):
            if pid in relevant_pids:
                dcg += 1.0 / math.log2(i + 2)  # Position starts counting from 1
        
        # Calculate IDCG (maximum DCG in ideal situation)
        idcg = 0.0
        for i in range(n):
            idcg += 1.0 / math.log2(i + 2)
        
        # Calculate NDCG for the current query
        ndcg = dcg / idcg if idcg != 0 else 0.0
        total_ndcg += ndcg

        # Calculate truncated NDCG for specified k values
        for k in k_list:
            current_k = min(k, n)  # Ensure k does not exceed the number of relevant documents
            truncated_predicted = predicted_pids[:current_k]
            
            # Calculate DCG after truncation
            truncated_dcg = 0.0
            for i, pid in enumerate(truncated_predicted):
                if pid in relevant_pids:
                    truncated_dcg += 1.0 / math.log2(i + 2)
            
            # Calculate IDCG after truncation
            truncated_idcg = 0.0
            for i in range(min(current_k, n)):
                truncated_idcg += 1.0 / math.log2(i + 2)
            
            # Calculate truncated NDCG
            truncated_ndcg = truncated_dcg / truncated_idcg if truncated_idcg != 0 else 0.0
            
            # Update metrics
            key = f"NDCG@{k}" if k is not None else "NDCG"
            if key not in ndcg_metrics:
                ndcg_metrics[key] = 0.0
            ndcg_metrics[key] += truncated_ndcg
            # For each qid in ground_truth, calculate a corresponding NDCG@k (multiple k values are available)

    # Calculate average values
    results = {}
    if valid_queries > 0:
        # Overall average NDCG (including all k values)
        overall_avg = total_ndcg / valid_queries
        results["NDCG"] = overall_avg
        
        # Average NDCG for each k value
        for k in k_list:
            key = f"NDCG@{k}"
            avg = ndcg_metrics.get(key, 0.0) / valid_queries
            results[key] = avg
    else:
        # Return 0 when there are no valid queries
        results["NDCG"] = 0.0
        for k in k_list:
            results[f"NDCG@{k}"] = 0.0

    return results

def format_scientific(value, precision=6):
    """Format float to scientific notation with specified significant digits"""
    return f"{value:.{precision-1}e}"  # .6e corresponds to 7 characters (e.g. 1.234567e-02), subtract 1 when precision=6


class CrossEncoderEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "eval/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    # Calculate AUC using formula method
    def calAUC(self,prob, labels):
        # Combine predicted values and labels into tuples
        data = list(zip(prob, labels))
        # Sort by prob in ascending order and get the label sequence
        rank = [label for pre, label in sorted(data, key=lambda x: x[0])]
        # Get indices of all positive samples
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]

        posNum = 0; negNum = 0
        for i in range(len(labels)):
            if (labels[i] == 1):
                posNum += 1
            else:
                negNum += 1
        return (sum(rankList) - posNum * (posNum + 1) / 2) / (posNum * negNum)

    def evaluate(self):
        """
        Evaluate CrossEncoder model performance.
        """
        local_rank = self.accelerator.process_index
        self.model.eval()
        
        predictions = defaultdict(list)
        # Total number of samples processed by the current process
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
        self.accelerator.wait_for_everyone()
        # If GPU0 processes 100 samples, GPU1 has shift=100 and starts with qid=100.
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        next_qid = shift
        search_idx_to_qid = {}
        user_aucs=[]
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                inputs = {key: val.to(self.accelerator.device) for key, val in batch['inputs'].items()}
                # batch_features={k:v.to(self.accelerator.device) for k,v in batch["features"].items()}
                batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["features"].items() }
                user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items()}
                note_idxs = batch['note_idxs']
                search_idxs = batch['search_idxs']
                Pos_Negs= batch['Pos_Neg']

                outputs = self.model(batch_features=batch_features,user_feat=user_feat, **inputs)
                scores = outputs.squeeze(-1)
                
                # Calculate overall AUC and user-level AUC
                # Overall AUC
                # print(f"scores:{scores}")
                # print(f"Pos_Negs:{Pos_Negs}")# The order here is the order of search _test
                # print(f"search_idxs:{search_idxs}")
                user_auc = self.calAUC(scores, Pos_Negs)
                # print(f"user_auc:{user_auc}")
                user_aucs.append(user_auc)

                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    if search_idx not in search_idx_to_qid:
                        # Assign a unique qid to each search_idx
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    # Record results
                    predictions[qid].append((note_idx, scores[i].item()))

        # Calculate average user-level AUC
        print(f"len(user_aucs):{len(user_aucs)}")
        print(f"Avg_auc:{sum(user_aucs) / len(user_aucs)}")

        # Current process writes prediction results to temporary file
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        # Main process merges all temporary files
        if self.accelerator.is_main_process:
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f)  
                    for line in f:
                        qid, pid, score = line.strip().split(',') 
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
            # Write final merged file
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            # Group by qid and sort documents by score
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            # Sort documents for each qid in descending order of score
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]
            
            # Calculate evaluation metrics
            # Get the first qid (in insertion order)
            # first_qid = next(iter(sorted_results))
            # Get and print the top 5 pids
            # top5_pids = sorted_results[first_qid][:5]
            # print(f"sorted_results[first_qid][:5]: {top5_pids}")

            # Print the first 5 qrels (assuming qrels is a dictionary)
            # first_qrel_qid = next(iter(self.qrels))
            # print(f"qrels[first_qid][:5]: {self.qrels[first_qrel_qid][:5]}")

            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            formatted_metrics = {
                k: format_scientific(v, precision=6) 
                for k, v in metrics.items()
            }
            print(f"metrics:{formatted_metrics}")
            # Calculate NDCG ranking metric
            NDCG_metrics = calculate_ndcg(sorted_results, self.qrels, k_list=None)
            print(f"NDCG:{NDCG_metrics}")

        return metrics


class VLMCrossEncoderEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    # Calculate AUC using formula method
    def calAUC(self,prob, labels):
        # Combine predicted values and labels into tuples
        data = list(zip(prob, labels))
        # Sort by prob in ascending order and get the label sequence
        rank = [label for pre, label in sorted(data, key=lambda x: x[0])]
        # Get indices of all positive samples
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]

        posNum = 0; negNum = 0
        for i in range(len(labels)):
            if (labels[i] == 1):
                posNum += 1
            else:
                negNum += 1
        return (sum(rankList) - posNum * (posNum + 1) / 2) / (posNum * negNum)

    def evaluate(self):
        """
        Evaluate VLM CrossEncoder model performance.
        """
        local_rank = self.accelerator.process_index
        self.accelerator.wait_for_everyone()
        self.model.eval()
        
        # Store prediction results for each query
        predictions = defaultdict(list)
        
        # Record current process's sample count
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
            
        self.accelerator.wait_for_everyone()
        
        # Calculate offset
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        
        next_qid = shift
        search_idx_to_qid = {}
        # Calculate AUC
        user_aucs=[]
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc=f"Evaluating on GPU {local_rank}"):
                inputs = {
                    key: val.to(self.accelerator.device)
                    for key, val in batch['inputs'].items() 
                }
                # print(f"inputs:{inputs}")
                batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["features"].items() }
                user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items()}
                note_idxs = batch['note_idxs']
                search_idxs = batch['search_idxs']
                Pos_Negs= batch['Pos_Neg']
                
                # Set mini-batch size
                mini_batch_size = 20
                batch_size = len(note_idxs)
                scores_list = []
                images_per_text = inputs['pixel_values'].size(0) // batch_size

                # Process by mini-batch
                for i in range(0, batch_size, mini_batch_size):
                    mini_batch_inputs = {
                        k: v[i:i+mini_batch_size] for k,v in inputs.items() if k != 'pixel_values'
                    }
                    mini_batch_inputs['pixel_values'] = inputs['pixel_values'][i*images_per_text:(i+mini_batch_size)*images_per_text]
                    
                    # minibatch features user feat
                    mini_batch_features={}
                    for k,v in batch_features.items():
                        mini_batch_features[k]=v[i:i+mini_batch_size]
                        
                    mini_batch_user_feat={}
                    for k,v in user_feat.items():
                        mini_batch_user_feat[k]=v[i:i+mini_batch_size]

                    mini_outputs = self.model(batch_features=mini_batch_features,
                                              user_feat=mini_batch_user_feat,
                                              **mini_batch_inputs)
                    mini_scores = mini_outputs.squeeze(-1)
                    scores_list.append(mini_scores)

                
                # Merge results from all mini-batches
                scores = torch.cat(scores_list, dim=0).reshape(-1)
                user_auc = self.calAUC(scores, Pos_Negs)
                # print(f"user_auc:{user_auc}")
                user_aucs.append(user_auc)

                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    # Assign qid for new search_idx
                    if search_idx not in search_idx_to_qid:
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    predictions[qid].append((note_idx, scores[i].item()))
                
        # Calculate average user-level AUC
        print(f"len(user_aucs):{len(user_aucs)}")
        print(f"Avg_auc:{sum(user_aucs) / len(user_aucs)}")

        # Write results to file
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        if self.accelerator.is_main_process:
            # Merge results from all GPUs
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f) 
                    for line in f:
                        qid, pid, score = line.strip().split(',')
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
            
            # Clean up temporary files
            for i in range(self.accelerator.num_processes):
                os.remove(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"))
            
            # Write final merged file
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            # Group by qid and sort documents by score
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            # Sort documents for each qid in descending order of score
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]
            
            # Calculate metrics
            # metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            formatted_metrics = {
                k: format_scientific(v, precision=6) 
                for k, v in metrics.items()
            }
            print(f"metrics:{formatted_metrics}")
            # Calculate NDCG ranking metric
            NDCG_metrics = calculate_ndcg(sorted_results, self.qrels, k_list=None)
            print(f"NDCG:{NDCG_metrics}")

            # Print detailed evaluation results
            # print("\nEvaluation Results:")
            # print("=" * 50)
            # for k, v in metrics.items():
            #     print(f"{k}: {v:.4f}")
            # print("=" * 50)
        
        return metrics

class MultiModalEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    # Calculate AUC using formula method
    def calAUC(self,prob, labels):
        # Combine predicted values and labels into tuples
        data = list(zip(prob, labels))
        # Sort by prob in ascending order and get the label sequence
        rank = [label for pre, label in sorted(data, key=lambda x: x[0])]
        # Get indices of all positive samples
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]

        posNum = 0; negNum = 0
        for i in range(len(labels)):
            if (labels[i] == 1):
                posNum += 1
            else:
                negNum += 1
        return (sum(rankList) - posNum * (posNum + 1) / 2) / (posNum * negNum)

    def evaluate(self):
        """
        Evaluate VLM CrossEncoder model performance.
        """
        local_rank = self.accelerator.process_index
        self.accelerator.wait_for_everyone()
        self.model.eval()
        
        # Store prediction results for each query
        predictions = defaultdict(list)
        
        # Record current process's sample count
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
            
        self.accelerator.wait_for_everyone()
        
        # Calculate offset
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        
        next_qid = shift
        search_idx_to_qid = {}
        # Calculate AUC
        user_aucs=[]
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc=f"Evaluating on GPU {local_rank}"):
                inputs = {
                    key: val.to(self.accelerator.device)
                    for key, val in batch['inputs'].items() 
                }

                batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["batch_features"].items() }
                user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items()}
                note_idxs = batch['note_idxs']
                search_idxs = batch['search_idxs']
                Pos_Negs= batch['Pos_Neg']
                
                # Set mini-batch size
                mini_batch_size = 20
                batch_size = len(note_idxs)
                scores_list = []
                images_per_text = inputs['pixel_values'].size(0) // batch_size

                # Process by mini-batch
                for i in range(0, batch_size, mini_batch_size):
                    mini_batch_inputs = {
                        k: v[i:i+mini_batch_size] for k,v in inputs.items() if k != 'pixel_values'
                    }
                    mini_batch_inputs['pixel_values'] = inputs['pixel_values'][i*images_per_text:(i+mini_batch_size)*images_per_text]
                
                    # minibatch features user feat
                    mini_batch_features={}
                    for k,v in batch_features.items():
                        mini_batch_features[k]=v[i:i+mini_batch_size]
                        
                    mini_batch_user_feat={}
                    for k,v in user_feat.items():
                        mini_batch_user_feat[k]=v[i:i+mini_batch_size]

                    mini_outputs = self.model(batch_features=mini_batch_features,
                                                user_feat=mini_batch_user_feat,
                                                **mini_batch_inputs)

                    mini_scores = mini_outputs.squeeze(-1)
                    # print(f"mini_scores:{mini_scores}")
                    # print(f"mini_scores.shape:{mini_scores.shape}")
                    if len(mini_scores.shape)==1:
                        mini_scores = mini_scores.unsqueeze(0)
                    scores_list.append(mini_scores)

                
                # Merge results from all mini-batches
                scores = torch.cat(scores_list, dim=1).reshape(-1)
                # The scores here should also be in shuffled order
                user_auc = self.calAUC(scores, Pos_Negs)
                # print(f"user_auc:{user_auc}")
                user_aucs.append(user_auc)

                # print(f"note_idxs:{note_idxs}")
                # print(f"search_idxs:{search_idxs}")

                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    # Assign qid for new search_idx
                    if search_idx not in search_idx_to_qid:
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    predictions[qid].append((note_idx, scores[i].item()))
                
        # Calculate average user-level AUC
        print(f"len(user_aucs):{len(user_aucs)}")
        print(f"Avg_auc:{sum(user_aucs) / len(user_aucs)}")

        # print(f"predictions:{predictions}")

        # Write results to file
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        if self.accelerator.is_main_process:
            # Merge results from all GPUs
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f) 
                    for line in f:
                        qid, pid, score = line.strip().split(',')
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
            
            # Clean up temporary files
            for i in range(self.accelerator.num_processes):
                os.remove(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"))
            
            # Write final merged file
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            # Group by qid and sort documents by score
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            # Sort documents for each qid in descending order of score
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]

            # Write sorted CSV file
            sorted_csv_path = os.path.join(self.output_dir, "sorted_rerank_results.csv")
            with open(sorted_csv_path, "w") as f:
                f.write("qid,pid,score\n")
                # Sort by qid (optional, ensure global qid order)
                for qid in sorted(sorted_results.keys()):
                    for pid in sorted_results[qid]:
                        f.write(f"{qid},{pid} \n")

            # Calculate metrics
            # metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            formatted_metrics = {
                k: format_scientific(v, precision=6) 
                for k, v in metrics.items()
            }
            print(f"metrics:{formatted_metrics}")
            # Calculate NDCG ranking metric
            NDCG_metrics = calculate_ndcg(sorted_results, self.qrels, k_list=None)
            print(f"NDCG:{NDCG_metrics}")

            # Print detailed evaluation results
            # print("\nEvaluation Results:")
            # print("=" * 50)
            # for k, v in metrics.items():
            #     print(f"{k}: {v:.4f}")
            # print("=" * 50)
        
        return metrics