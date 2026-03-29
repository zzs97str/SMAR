# from utils import *
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from registry import register_class
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import os
import jieba
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import seaborn as sns
import torch.distributed as dist
import datetime
import torch.nn.functional as F


def is_main_process():
    # Usually LOCAL_RANK of main process is 0
    return os.environ.get("LOCAL_RANK", "0") == "0"


### Define Transformer Classifier Model
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        # # Define learnable alpha parameter for each head independently, initial value 0.0
        # self.alpha = nn.Parameter(torch.zeros(n_heads),requires_grad=True)
        self.step=0
        self.vis_dir="/path_to_CrossRank/output/CrossRankEXP/fig"

    def forward(self, x):
        B, N, C = x.size() # B=1, N=number of positive/negative samples, C=feature dimension
        # print(f"B,N,C={B,N,C}") B,N,C=(1, 4, 3584)
        # 3 represents qkv
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        # print(f"qkv.shape={qkv.shape}") qkv.shape=torch.Size([1, 4, 3, 2, 1792])
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, n_heads, N, N)
        # Add alpha for each head on the diagonal
        identity = torch.eye(N, device=attn_scores.device, dtype=attn_scores.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)

        # One bias value per head, set to 1/N here, adjustable
        weights = torch.full((self.n_heads,), fill_value=3.0 / N, device=attn_scores.device, dtype=attn_scores.dtype)
        attn_scores = attn_scores + weights.view(1, self.n_heads, 1, 1) * identity
        attn_probs = F.softmax(attn_scores, dim=-1)

        # === Visualize attention distribution and statistics for head 1 and 2 ===
        if B == 1 and is_main_process() and self.step % 200 ==0:
            head_indices = [0, 1]
            for idx in head_indices:
                attn_matrix = attn_probs[0, idx].detach().cpu().numpy()  # (N, N)

                # 1. Draw heatmap
                plt.figure(figsize=(6, 5))
                sns.heatmap(attn_matrix, cmap='viridis', square=True,
                            xticklabels=True, yticklabels=True, cbar=True)
                plt.title(f'Head {idx} Attention Map')
                plt.xlabel('Key Token Index')
                plt.ylabel('Query Token Index')
                save_path = os.path.join(self.vis_dir, f"step{self.step}_head_{idx}_attn_heatmap.png")
                plt.savefig(save_path)
                plt.close()

                # 2. Calculate mean and variance per row
                mean_per_row = attn_probs[0, idx].mean(dim=-1).detach().cpu()
                var_per_row = attn_probs[0, idx].var(dim=-1).detach().cpu()

                # print(f"[Head {idx}] Variance of attention per row:\n{var_per_row}")

        # === Attention output ===
        attn_output = attn_probs @ v  # (B, n_heads, N, head_dim)
        # Merge all heads
        attn_output = attn_output.transpose(1, 2).contiguous()

        # # FlashAttention
        # # qkv should have dimension (batch_size, seqlen, 3, nheads, headdim)
        # # FlashAttention forward only supports head dimension at most 256
        # attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=0.1, softmax_scale=None, causal=False,
        #                   window_size=(-1, -1), alibi_slopes=None, deterministic=False)
        # # print(f"attn_output.shape={attn_output.shape}")
        # # attn_output.shape : (batch_size, seqlen, nheads, headdim)

        attn_output = attn_output.reshape(B, N, C)

        self.step += 1

        return self.out_proj(attn_output)

class CrossAttention(nn.Module):
    def __init__(self, user_dim, embed_dim, n_heads, vis_dir="/path_to_CrossRank/output/multimodal/fig"):
        """
        user_dim : User feature dimension
        embed_dim : Content embedding dimension
        n_heads : Number of attention heads
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        # Map user features to embed_dim as query
        self.user_proj_v = nn.Linear(user_dim, embed_dim, bias=True)
        self.user_proj_k = nn.Linear(user_dim, embed_dim, bias=True)

        # Content Key/Value projection
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.vis_dir = vis_dir
        self.step = 0


    def forward(self, user_feat, content_embeds):
        """
        user_feat: [B, user_dim]
        content_embeds: [B, N, embed_dim]
        """
        B, N, C = content_embeds.size()

        # === Generate query from embedding features ===
        # [B, N, embed_dim] → [B,N, embed_dim]
        q = self.q_proj(content_embeds)  # (B,N, embed_dim)
        # reshape for multi-head
        q = q.view(B, N,self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, N, head_dim)

        # === Generate key, value from user features ===
        # [B, 1, embed_dim] → [B, 1, embed_dim]
        k = self.user_proj_k(user_feat)
        v = self.user_proj_v(user_feat)

        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, 1, head_dim)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, 1, head_dim)

        # === Calculate Attention scores ===
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, N, 1)
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, n_heads, N, 1)

        # === Optional: Visualize attention ===
        # if B == 1 and self.step % 200 == 0:
        #     attn_matrix = attn_probs[0].detach().cpu().numpy()  # (n_heads, 1, N)
        #     for idx in range(min(2, self.n_heads)):
        #         plt.figure(figsize=(8,2))
        #         sns.heatmap(attn_matrix[idx][0:1], cmap='viridis', cbar=True)
        #         plt.title(f'Step {self.step} Head {idx} Attention')
        #         save_path = os.path.join(self.vis_dir, f"step{self.step}_head{idx}_cross_attn.png")
        #         plt.savefig(save_path)
        #         plt.close()

        # === Attention output ===
        attn_output = attn_probs @ v  # (B, n_heads, N, head_dim)
        attn_output=attn_output.transpose(1, 2).contiguous()
        # (B, N, n_heads, head_dim)

        # Merge heads
        attn_output = attn_output.reshape(B, N, C)  # (B, N,embed_dim)

        # Output projection
        output = self.out_proj(attn_output)  # (B, embed_dim)

        self.step += 1
        return output  # (B, embed_dim)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        mean_embeddings = sum_embeddings/sum_mask
        return mean_embeddings


class SelfAttentionBlock(nn.Module):
    def __init__(self, statistic_dim,dim, n_heads, config=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(config=config)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SelfAttentionClassifier(nn.Module):
    def __init__(self,statistic_dim=100, input_dim=1442, n_layers=2, n_heads=3, config=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim, bias=True)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(statistic_dim,input_dim, n_heads, config=config)
            for _ in range(n_layers)
        ])

        self.common_proj = nn.Linear(input_dim, 1024, bias=True)
        self.act_fn = nn.SiLU()
        self.out_proj = nn.Linear(1024, 1, bias=True)

    def forward(self, x):  # x shape: (batch, seq_len, dim)
        x = self.input_proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.common_proj(x)
        x = self.act_fn(x)
        return self.out_proj(x)


class CrossRankMultiModalRankModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']

        self.hf_model_config = AutoConfig.from_pretrained(self.model_config['model_name_or_path'])
        for key in self.model_config:
            self.hf_model_config.__dict__[key] = self.model_config[key]

        # Qwen3-reranker
        self.model = AutoModelForCausalLM.from_pretrained(
            "model/Qwen3-Reranker-4B",
            config=self.hf_model_config,
            trust_remote_code=True
        )
        self.mean_pooling = MeanPooling()

        # BERT
        # self.model = AutoModel.from_pretrained(
        #     "model/bert-base-chinese/",
        #     config=self.hf_model_config,
        #     trust_remote_code=True
        # )
        # # New
        # self.model.pooler = None  # Remove pooling layer

        # Define classifier structure (at least 3 linear layers)
        statistic_dim= 244
        all_hidden_size = 2640
        Attention_config = {
            "hidden_size": all_hidden_size,
            "intermediate_size": 4 * all_hidden_size # Usually 4x hidden_size
        }
        self.classifier = SelfAttentionClassifier(
            statistic_dim = statistic_dim,
            input_dim=all_hidden_size, 
            n_layers=6, 
            n_heads=2,
            config=Attention_config
            )        
        
        # Create directory, no error if exists
        os.makedirs(self.model_config['lora_checkpoint_dir'], exist_ok=True)
        classifier_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'classifier_base.pt')
        if dist.is_initialized():
            dist.barrier()
            print("Waiting to load together")
        if os.path.exists(classifier_base_ckpt):
            self.classifier.load_state_dict(torch.load(classifier_base_ckpt))
            print("Loaded classifier from previous base model")
        else:
            if dist.is_initialized():
                dist.barrier()
            if is_main_process():
                torch.save(self.classifier.state_dict(), classifier_base_ckpt)
                print("Saved classifier and base_model.")
        if dist.is_initialized():
            dist.barrier()
            print("Main process saved successfully, continuing execution")

        self.query_feature_bits={
            'query_feat_5':2,
            'query_feat_3':4,
            "query_feat_1":4}
        self.query_binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True),
            ) for feat, total_bits in self.query_feature_bits.items()
        })

        # Candidate side features
        self.max_statistic_bits={
            "candidate_feat_1":4,
            "candidate_feat_2":4,
            "candidate_feat_3":4,
            "candidate_feat_4":4,
            "candidate_feat_5":4,
            "upstream_label":4,
            "query_feat_2":4,
            "query_feat_4":2}
        self.statistic_binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True),
            ) for feat, total_bits in self.max_statistic_bits.items()
        })

        self.cls_norm = RMSNorm(2560)  
        self.query_norm = RMSNorm(20)
        self.statistic_norm = RMSNorm(60)

        # Load checkpoint
        query_binary_encoders_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'query_binary_encoders.pt')
        if os.path.exists(query_binary_encoders_path):
            self.query_binary_encoders.load_state_dict(torch.load(query_binary_encoders_path))
            print(f"Loaded query_binary_encoders parameters from {query_binary_encoders_path}")
            print("Loaded query_binary_encoders")
        else:
            print("Initializing query_binary_encoders from init")

        statistic_binary_encoders_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'statistic_binary_encoders.pt')
        if os.path.exists(statistic_binary_encoders_path):
            self.statistic_binary_encoders.load_state_dict(torch.load(statistic_binary_encoders_path))
            print(f"Loaded statistic_binary_encoders parameters from {statistic_binary_encoders_path}")
            print("Loaded statistic_binary_encoders_path")
        else:
            print("Initializing statistic_binary_encoders from init")

        cls_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'cls_norm.pt')
        if os.path.exists(cls_norm_path):
            self.cls_norm.load_state_dict(torch.load(cls_norm_path))
            print(f"loaded cls_norm from {cls_norm_path}")
        else:
            print("no checkpoint for cls_norm")

        query_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'query_norm.pt')
        if os.path.exists(query_norm_path):
            self.query_norm.load_state_dict(torch.load(query_norm_path))
            print(f"loaded query_norm from {query_norm_path}")
        else:
            print("no checkpoint for query_norm")

        statistic_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'statistic_norm.pt')
        if os.path.exists(statistic_norm_path):
            self.statistic_norm.load_state_dict(torch.load(statistic_norm_path))
            print(f"loaded statistic_norm from {statistic_norm_path}")
        else:
            print("no checkpoint for statistic_norm")

        for param in self.model.parameters():
            param.requires_grad = True
        if isinstance(self.model, PeftModel):
            print("self.model is a lora model")
        else:
            print("self.model is NOT a lora model ")

        for param in self.classifier.parameters():
            param.requires_grad = True
        if self.model_config['use_lora']:
            self.model, self.classifier =self._setup_lora(self.model, self.classifier)

        for encoder in self.query_binary_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        for encoder in self.statistic_binary_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        for param in self.cls_norm.parameters():
            param.requires_grad = True
        for param in self.query_norm.parameters():
            param.requires_grad = True
        for param in self.statistic_norm.parameters():
            param.requires_grad = True


        if self.model_config['gradient_checkpointing']:
            print("Gradient checkpoint enabled")
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        else:
            print("Gradient checkpoint disabled")


        # Default FP32
        self.model = self.model.to(torch.float16)
        self.classifier = self.classifier.to(torch.float16)
        self.statistic_binary_encoders = self.statistic_binary_encoders.to(torch.float16)
        self.query_binary_encoders = self.query_binary_encoders.to(torch.float16)
        self.cls_norm = self.cls_norm.to(torch.float16)
        self.query_norm = self.query_norm.to(torch.float16)
        self.statistic_norm = self.statistic_norm.to(torch.float16)

    def _setup_lora(self, model, classifier):
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        Qwen_lora_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'Qwen_lora')
        if os.path.exists(os.path.join(Qwen_lora_path, 'adapter_config.json')):

            model = PeftModel.from_pretrained(model, Qwen_lora_path)
            print(f"Loaded Qwen LORA from {Qwen_lora_path}")
            for name, param in model.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True  
        else:
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj","down_proj"],
            )

            model = PeftModel(model, peft_config)
            for name, param in model.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True  
            print('Add Qwen lora adapter from init')

        # ----------------------------
        # Step 2: Setup LoRA for the classifier
        # ----------------------------
        print(f"Classifier: Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        classifier_lora_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], "classifier_lora")
        if os.path.exists(classifier_lora_ckpt):
            # Initialize LoRA weights (not using from_pretrained)
            peft_classifier_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_proj", "up_proj", "down_proj","out_proj","input_proj","common_proj","user_proj_v","user_proj_k","q_proj"],  # LoRA on these modules
            )
            classifier = PeftModel(classifier, peft_config=peft_classifier_config)

            # Load LoRA adapter weights
            classifier.load_adapter(classifier_lora_ckpt, adapter_name="default", is_trainable=True)
            print(f"Loaded classifier LoRA adapter from {classifier_lora_ckpt}")
                
            for name, param in classifier.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
 
        else:
            peft_classifier_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_proj", "up_proj", "down_proj","out_proj","input_proj","common_proj","user_proj_v","user_proj_k","q_proj"],  # LoRA on these modules
            )
            classifier = PeftModel(classifier, peft_classifier_config)
            for name, param in classifier.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
            print('Add LoRA adapter to classifier from init')

        if isinstance(classifier, PeftModel):
            print("classifier is a lora model")
        else:
            print("classifier is NOT a lora model ")

        return model, classifier

    def unique_keep_order(self,tensor):
        seen = set()
        unique_indices = []
        for i, v in enumerate(tensor.tolist()):
            if v not in seen:
                seen.add(v)
                unique_indices.append(i)
        unique_indices = torch.tensor(unique_indices, device=tensor.device)
        unique_values = tensor[unique_indices]
        return unique_values, unique_indices

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, 
                    device=last_hidden_states.device), sequence_lengths]

    def forward(self, search_idxs, query_feat, statistic_feat, **inputs):   
        # for key, value in inputs.items():
        #     # Print key and shape of corresponding value
        #     print(f"Key: {key}, Shape: {value.shape}")

        outputs = self.model(**inputs, output_hidden_states=True)
        # cls
        all_hidden_states = outputs.hidden_states
        last_hidden_state = all_hidden_states[-1]
        # print(f"last_hidden_state.shape:{last_hidden_state.shape}")
        attention_mask = inputs["attention_mask"]
        # print(f"attention_mask.shape:{attention_mask.shape}")
        cls_output = self.mean_pooling(last_hidden_state, attention_mask)
        cls_output = cls_output.to(self.classifier.base_model.model.layers[0].mlp.up_proj.lora_A.default.weight)

        # # Construct feature tensor: query side features [B, F]
        feat_tensors = []
        for feat, encoder in self.query_binary_encoders.items():
            feat_tensor = query_feat[feat]
            # print(f"feat_tensor:{feat_tensor}")
            encoded = torch.stack([encoder(i) for i in feat_tensor], dim=0)  # encoded shape [B, 20], B=total positive/negative samples, N=dimension after encoding
            # Each element of feat_tensor unified to dimension 20
            feat_tensors.append(encoded)

        # # Concatenate all features: [B, 5*20] = [B, 100]
        feat_tensor = torch.cat(feat_tensors, dim=1)  # [B, 100], B=total positive/negative samples, 100=dimension after encoding all features


        Statistic_feats = []
        for feat, encoder in self.statistic_binary_encoders.items():
            user_feat_tensor = statistic_feat[feat]
            # feat_tensor is a list of tensors, each tensor represents binary converted tensor of current value
            encoded = torch.stack([encoder(i) for i in user_feat_tensor], dim=0)  # encoded shape [B, 20]
            # Each element of feat_tensor unified to dimension 20
            Statistic_feats.append(encoded)

        Statistic_feats = torch.cat(Statistic_feats, dim=1) # [B, X], B=total positive/negative samples, X=dimension after encoding all features

        cls_output = self.cls_norm(cls_output)         # [B, H]


        feat_tensor = self.query_norm(feat_tensor)      # [B, F1]
        # print(f"feat_tensor.dtype:{feat_tensor.dtype}")
        Statistic_feats = self.statistic_norm(Statistic_feats)        # [B, F2]
        # print(f"Statistic_feats.dtype:{Statistic_feats.dtype}")

        concat = torch.cat([feat_tensor, cls_output], dim=1)  # [B, H+F]
        concat = torch.cat([concat, Statistic_feats], dim=1)  # [B, H+F]
        # print(f"concat.shape:{concat.shape}")

        # Split by query: Core code - split into list of tensors by search_idx
        search_tensor = torch.tensor(search_idxs)
        unique_ids, idx = self.unique_keep_order(search_tensor)

        batch_features = [concat[search_tensor == uid] for uid in unique_ids]

        batch_logits=[]
        # Batch processing
        for feat in batch_features:
            # feat.shape: [seq_len, 848]
            # x refers to embedding vector of query and doc
            x = feat.unsqueeze(dim=0) # From [N, H+F] -> [1, N, H+F]
            # print(f"x.shape:{x.shape}")
            logits = self.classifier(x).squeeze(dim=0) # [N,1]
            batch_logits.append(logits)

        return batch_logits

    def save_pretrained(self, save_path):
        # Merge LoRA adapter into base model
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(os.path.join(save_path, "Qwen_lora"))
            print("Saved Qwen LoRA adapter.")

        # Merge LoRA adapter into base model
        if isinstance(self.classifier, PeftModel):
            self.classifier.save_pretrained(os.path.join(save_path, "classifier_lora"))
            print("Saved classifier LoRA adapter.")

        # Save binary_encoders (new part)
        query_binary_encoders_path = os.path.join(save_path, 'query_binary_encoders.pt')
        torch.save(self.query_binary_encoders.state_dict(), query_binary_encoders_path)

        statistic_binary_encoders_path = os.path.join(save_path, 'statistic_binary_encoders.pt')
        torch.save(self.statistic_binary_encoders.state_dict(), statistic_binary_encoders_path)
        # RMSNorm parameters
        torch.save(self.cls_norm.state_dict(), os.path.join(save_path, 'cls_norm.pt'))
        torch.save(self.query_norm.state_dict(), os.path.join(save_path, 'query_norm.pt'))
        torch.save(self.statistic_norm.state_dict(), os.path.join(save_path, 'statistic_norm.pt'))