from utils import *
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from registry import register_class
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
# from transformers import pipeline
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import os
import jieba
import numpy as np
from typing import List, Tuple
# from rank_bm25 import BM25Okapi
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import time
import seaborn as sns
import torch.distributed as dist
import datetime
dist.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(seconds=3000)
)

def is_main_process():
    # 通常主进程的 LOCAL_RANK 为 0
    return os.environ.get("LOCAL_RANK", "0") == "0"

class BaseModel:
    def __init__(self, config):
        self.model_config = config['model']
        self.hf_model_config = AutoConfig.from_pretrained("model/bert-base-chinese/")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "model/bert-base-chinese/",
            trust_remote_code=True
        )
        
        for key in self.model_config:
            self.hf_model_config.__dict__[key] = self.model_config[key]
            
        self.is_bert = 'bert' in self.model_config['model_name_or_path']
        self.model = self._create_model()
        
        if self.model_config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            
        if not self.is_bert:
            try:
                self._freeze_non_crossattention_parameters()
            except:
                print("freeze_non_crossattention_parameters failed")



### 定义Transformer分类器模型
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

# class SelfAttention(nn.Module):
#     def __init__(self, dim, n_heads):
#         super().__init__()
#         self.n_heads = n_heads
#         self.head_dim = dim // n_heads
#         assert dim % n_heads == 0
#         self.qkv_proj = nn.Linear(dim, dim * 3, bias=True)
#         self.out_proj = nn.Linear(dim, dim, bias=True)
#         # # 定义每个 head 独立的可学习 alpha 参数，初始值设为0.0
#         # self.alpha = nn.Parameter(torch.zeros(n_heads),requires_grad=True)
#         self.step=0
#         self.vis_dir="/root/paddlejob/workspace/env_run/output/multimodal/fig"

#     def forward(self, x):
#         B, N, C = x.size() # B为1，N 为正负样本数 ，C 为特征维度
#         # print(f"B,N,C={B,N,C}") B,N,C=(1, 4, 3584)
#         # 3 表示qkv
#         qkv = self.qkv_proj(x).reshape(B, N, 3, self.n_heads, self.head_dim)
#         # print(f"qkv.shape={qkv.shape}") qkv.shape=torch.Size([1, 4, 3, 2, 1792])

#         # qkv[:, :, 0] qkv[:, :, 0，：，：],两者都对,PyTorch支持自动省略尾部冒号
#         q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, N, n_heads, head_dim)
#         # q.shape == (B, N, n_heads, head_dim)
#         q = q.transpose(1, 2)  # (B, n_heads, N, head_dim)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, n_heads, N, N)

#         # 在对角线上加不同 head 的 alpha
#         identity = torch.eye(N, device=attn_scores.device).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
#         # 每个head一个偏置值，这里设置为1/N，可调整
#         weights = torch.full((self.n_heads,), fill_value=3.0 / N, device=attn_scores.device, dtype=attn_scores.dtype)
#         attn_scores = attn_scores + weights.view(1, self.n_heads, 1, 1) * identity

#         attn_probs = F.softmax(attn_scores, dim=-1)

#         # === 可视化第1和第2个head的注意力分布和统计信息 ===
#         if B == 1 and is_main_process() and self.step % 200 ==0:
#             head_indices = [0, 1]
#             for idx in head_indices:
#                 attn_matrix = attn_probs[0, idx].detach().cpu().numpy()  # (N, N)

#                 # 1. 画热力图
#                 plt.figure(figsize=(6, 5))
#                 sns.heatmap(attn_matrix, cmap='viridis', square=True,
#                             xticklabels=True, yticklabels=True, cbar=True)
#                 plt.title(f'Head {idx} Attention Map')
#                 plt.xlabel('Key Token Index')
#                 plt.ylabel('Query Token Index')
#                 save_path = os.path.join(self.vis_dir, f"step{self.step}_head_{idx}_attn_heatmap.png")
#                 plt.savefig(save_path)
#                 plt.close()

#                 # 2. 计算每行均值和方差
#                 mean_per_row = attn_probs[0, idx].mean(dim=-1).detach().cpu()
#                 var_per_row = attn_probs[0, idx].var(dim=-1).detach().cpu()

#                 print(f"[Head {idx}] 每行注意力方差:\n{var_per_row}")

#                 # 3. 可视化均值 & 方差（柱状图）
#                 # plt.figure(figsize=(8, 4))
#                 # plt.plot(mean_per_row.detach().numpy(), marker='o', label='Mean')
#                 # plt.plot(var_per_row.detach().numpy(), marker='x', label='Variance')
#                 # plt.title(f'Head {idx} Attention Row Stats')
#                 # plt.xlabel('Query Token Index')
#                 # plt.ylabel('Value')
#                 # plt.legend()
#                 # save_path = os.path.join(self.vis_dir, f"step{self.step}_head_{idx}_attn_stats.png")
#                 # plt.savefig(save_path)
#                 # plt.close()

#         # === attention输出 ===

#         attn_output = attn_probs @ v  # (B, n_heads, N, head_dim)
#         # 合并所有头部
#         attn_output = attn_output.transpose(1, 2).contiguous()

#         # # FlashAttention
#         # # qkv应该的维度是 (batch_size, seqlen, 3, nheads, headdim)
#         # # FlashAttention forward only supports head dimension at most 256
#         # attn_output = flash_attn_qkvpacked_func(qkv, dropout_p=0.1, softmax_scale=None, causal=False,
#         #                   window_size=(-1, -1), alibi_slopes=None, deterministic=False)
#         # # print(f"attn_output.shape={attn_output.shape}")
#         # # attn_output.shape : (batch_size, seqlen, nheads, headdim)

#         attn_output = attn_output.reshape(B, N, C)

#         self.step += 1

#         return self.out_proj(attn_output)

class CrossAttention(nn.Module):
    def __init__(self, user_dim, embed_dim, n_heads, vis_dir="/root/paddlejob/workspace/env_run/output/multimodal/fig"):
        """
        user_dim : 用户特征维度
        embed_dim : 内容嵌入维度
        n_heads : 注意力头数
        """
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0

        # 用户特征映射到 embed_dim 作为 query
        self.user_proj_v = nn.Linear(user_dim, embed_dim, bias=True)
        self.user_proj_k = nn.Linear(user_dim, embed_dim, bias=True)

        # 内容 Key/Value 映射
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # 输出 projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.vis_dir = vis_dir
        self.step = 0
        # print("交叉注意力！")

    def forward(self, user_feat, content_embeds):
        """
        user_feat: [B, user_dim]
        content_embeds: [B, N, embed_dim]
        """
        B, N, C = content_embeds.size()

        # === 嵌入特征生成 query ===
        # [B, N, embed_dim] → [B,N, embed_dim]
        q = self.q_proj(content_embeds)  # (B,N, embed_dim)
        # reshape for multi-head
        q = q.view(B, N,self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, N, head_dim)

        # === 用户生成 key, value ===
        # [B, 1, embed_dim] → [B, 1, embed_dim]
        k = self.user_proj_k(user_feat)
        v = self.user_proj_v(user_feat)

        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, 1, head_dim)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1,2)  # (B, n_heads, 1, head_dim)

        # === 计算 Attention scores ===
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_heads, N, 1)
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, n_heads, N, 1)

        # === 可选：可视化注意力 ===
        if B == 1 and self.step % 200 == 0:
            attn_matrix = attn_probs[0].detach().cpu().numpy()  # (n_heads, 1, N)
            for idx in range(min(2, self.n_heads)):
                plt.figure(figsize=(8,2))
                sns.heatmap(attn_matrix[idx][0:1], cmap='viridis', cbar=True)
                plt.title(f'Step {self.step} Head {idx} Attention')
                save_path = os.path.join(self.vis_dir, f"step{self.step}_head{idx}_cross_attn.png")
                plt.savefig(save_path)
                plt.close()

        # === Attention 输出 ===
        attn_output = attn_probs @ v  # (B, n_heads, N, head_dim)
        attn_output=attn_output.transpose(1, 2).contiguous()
        # (B, N, n_heads, head_dim)

        # 合并 heads
        attn_output = attn_output.reshape(B, N, C)  # (B, N,embed_dim)

        # 输出 projection
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

class BinaryMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_multiplier=2, activation='silu', use_residual=True):
        """
        二值特征编码 MLP 模块，支持残差连接和门控机制。

        Args:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            hidden_multiplier (int): 中间层维度 = input_dim * hidden_multiplier
            activation (str): 'silu' 或 'relu'
            use_residual (bool): 是否启用残差连接
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = input_dim * hidden_multiplier
        self.use_residual = use_residual

        self.gate_proj = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.up_proj = nn.Linear(input_dim, self.hidden_size, bias=True)
        self.down_proj = nn.Linear(self.hidden_size, output_dim, bias=True)

        self.act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        
        if isinstance(self.act_fn, nn.ReLU):
            print("激活函数为nn.ReLU()")


        # 如果启用残差且维度不一致，添加投影层
        if use_residual and input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim, bias=True)
        else:
            self.res_proj = None

    def forward(self, x):
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        activated = self.act_fn(gate_output) * up_output
        out = self.down_proj(activated)

        if self.use_residual:
            if self.res_proj:
                x = self.res_proj(x)
            if x.shape == out.shape:
                out = out + x
        return out

class ResidualBinaryMLPBlock(nn.Module):
    def __init__(self, mlp_layers: list):
        """
        封装多个 BinaryMLP 层，自动添加跨层残差连接。

        Args:
            mlp_layers (list): BinaryMLP 实例的列表
        """
        super().__init__()
        self.layers = nn.ModuleList(mlp_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # 如果维度一致，默认 residual connection 已在 BinaryMLP 中处理
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, config=None):
        super().__init__()
        # self.norm1 = RMSNorm(dim)
        # self.attn = SelfAttention(dim, n_heads)
        # self.norm2 = RMSNorm(dim)
        self.mlp = MLP(config=config)

    def forward(self, x):
        # x = x + self.attn(self.norm1(x))
        x = x + self.mlp(x)
        return x

class TextDecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, config=None):
        super().__init__()
        # self.norm1 = RMSNorm(dim)
        # self.attn = SelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        # self.norm = nn.BatchNorm1d(dim)
        self.mlp = MLP(config=config)

    def forward(self, x):
        # x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        # print(f"x.shape:{x.shape}")
        # x = x + self.mlp(self.norm(x))

        return x

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=2210, n_layers=2, n_heads=3, config=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim, bias=True)
        self.layers = nn.ModuleList([
            TextDecoderBlock(input_dim, n_heads, config=config)
            for _ in range(n_layers)
        ])

        # self.out_proj = nn.Linear(input_dim, 1, bias=True)
        self.common_proj = nn.Linear(input_dim, 1024, bias=True)
        self.out_proj = nn.Linear(1024, 1, bias=True)

    def forward(self, x):  # x shape: (batch, seq_len, dim)
        self.half()
        x = self.input_proj(x)
        # batch_size, seq_len, dim = x.size()
        # # 调整形状为(batch*seq_len, dim)以适配BatchNorm1d
        # x = x.reshape(-1, dim)
        # x = self.batch_norm1(x)
        # x = x.reshape(batch_size, seq_len, dim)      
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # if i ==3:
            #     x=self.norm_out3(x)
            #     x=layer(x)
            # else:
            #     x = layer(x)
        x = self.common_proj(x)
        
        return self.out_proj(x)

class VLM_TransformerClassifier(nn.Module):
    def __init__(self, input_dim=2210, n_layers=2, n_heads=3, config=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim, bias=True)
        self.layers = nn.ModuleList([
            DecoderBlock(input_dim, n_heads, config=config)
            for _ in range(n_layers)
        ])

        self.common_proj = nn.Linear(input_dim, 1024, bias=True)
        self.act_fn = nn.SiLU()
        self.out_proj = nn.Linear(1024, 1, bias=True)

    def forward(self, x):  # x shape: (batch, seq_len, dim)
        x = self.input_proj(x)
        # batch_size, seq_len, dim = x.size()
        # # 调整形状为(batch*seq_len, dim)以适配BatchNorm1d
        # x = x.reshape(-1, dim)
        # x = self.batch_norm1(x)
        # x = x.reshape(batch_size, seq_len, dim)      
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # print("无 norm")
        x = self.common_proj(x)
        x = self.act_fn(x)
        # x_cls = x[:, 0]  # 默认 cls token 在第0个位置
        return self.out_proj(x)

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

class CrossEncoderModel(torch.nn.Module, BaseModel):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        hf_config = AutoConfig.from_pretrained(config['model']['model_name_or_path'])

        self.bert_hidde_size = 1130
        # 定义分类器结构（至少3层线性层）
        Attention_config = {
            "hidden_size": self.bert_hidde_size,
            "intermediate_size": 4 * self.bert_hidde_size # 通常是 hidden_size 的 4 倍
        }
        self.classifier = TransformerClassifier(
            input_dim=self.bert_hidde_size, 
            n_layers=6, 
            n_heads=1,
            config=Attention_config
            )   

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
        # 对字典中的值进行 log2 操作，并取上整
        self.feature_max_values = {
            key: np.ceil(np.log2(value)) if value > 0 else value
            for key, value in self.feature_max_values.items()
        }
        self.feature_bits={feat: len(bin(int(max_value))) - 2 for feat, max_value in self.feature_max_values.items()}
        self.binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True), #添加两层全连接层，增强模型表达能力
                # nn.ReLU(),
                # nn.Linear(total_bits*2, total_bits*4, bias=True),
                # nn.ReLU(),
                # nn.Linear(total_bits*4, total_bits*2, bias=True)
            ) for feat, total_bits in self.feature_bits.items()
        })
        print("已初始化 binary_encoders")

        self.all_feat={'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}
        self.user_feat_bits={feat: len(bin(int(max_value))) - 2 for feat, max_value in self.all_feat.items()}
        self.user_feat_bits["age"]=11
        self.user_feat_bits["gender"]=2
        self.user_binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True), #添加两层全连接层，增强模型表达能力
                # nn.ReLU(),
                # nn.Linear(total_bits*2, total_bits*4, bias=True),
                # nn.ReLU(),
                # nn.Linear(total_bits*4, total_bits*2, bias=True)
            ) for feat, total_bits in self.user_feat_bits.items()
        })
        print("user_binary_encoders已初始化完成")

        self.cls_norm = RMSNorm(768)  # hidden_size 是 cls_output 的维度
        self.feat_norm = RMSNorm(118)  # feat_dim 是 feat_tensor 的维度，例如 100
        self.user_norm = RMSNorm(244)


        init_weights=(1.0, 1.0, 1.0)
        self.alpha = nn.Parameter(torch.tensor(init_weights[0], dtype=torch.float16),requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(init_weights[1], dtype=torch.float16),requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor(init_weights[2], dtype=torch.float16),requires_grad=True)
        # self.mean_pooling = MeanPooling()
        BaseModel.__init__(self, config)

        # for encoder in self.binary_encoders.values():
        #     for param in encoder.parameters():
        #         param.requires_grad = True


    def _create_model(self):
        if self.is_bert:
            model = AutoModel.from_pretrained(
                "model/bert-base-chinese/",
                config=self.hf_model_config,
                trust_remote_code=True
            )
            # 新增
            model.pooler = None  # 移除池化层
            # 对 BERT 模型应用 LoRA（新增）,必须要赋值给 model
            # model = self._setup_lora(model)
            

        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config['model_name_or_path'],
                config=self.hf_model_config,
                attn_implementation='eager',
                trust_remote_code=True
            )
            model.base_model.get_input_embeddings().weight.requires_grad = False
            
            model = self._setup_lora(model)
        
        self.model_config['model_name_or_path']= "model/Submodel_ckpt/2025-07-10-22-53-29/bert_checkpoints"

        bert_path = self.model_config['model_name_or_path']
        print(f"bert_path:{bert_path}")
        if os.path.exists(bert_path):
            model = AutoModel.from_pretrained(bert_path)
            print(f"Loaded bert model parameters from {bert_path}")

        classifier_path = os.path.join(self.model_config['model_name_or_path'], 'classifier.pt')
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path))
            print(f"已加载 classifier parameters from {classifier_path}")

        binary_encoders_path = os.path.join(self.model_config['model_name_or_path'], 'binary_encoders.pt')
        if os.path.exists(binary_encoders_path):
            self.binary_encoders.load_state_dict(torch.load(binary_encoders_path))
            print(f"Loaded binary_encoders parameters from {binary_encoders_path}")
            print("已加载 binary_encoders")

        user_binary_encoders_path = os.path.join(self.model_config['model_name_or_path'], 'user_binary_encoders.pt')
        if os.path.exists(user_binary_encoders_path):
            self.user_binary_encoders.load_state_dict(torch.load(user_binary_encoders_path))
            print(f"Loaded binary_encoders parameters from {user_binary_encoders_path}")
            print("已加载 user_binary_encoders_path")

        # 设置 BERT 的 requires_grad 为 True
        # for param in model.parameters():
        #     param.requires_grad = False
        # print("冻结 BERT")

        for param in model.parameters():
            param.requires_grad = True
        print("不冻结 BERT")
        model.pooler = None

        # 设置 classifier 的 requires_grad 为 True
        for param in self.classifier.parameters():
            param.requires_grad = True

        # # 设置所有参数的 requires_grad 为 True
        for param in self.binary_encoders.parameters():
            param.requires_grad = True


        alpha_path = os.path.join(self.model_config['model_name_or_path'], 'alpha.pt')
        if os.path.exists(alpha_path):
            self.alpha = torch.load(os.path.join(alpha_path))
            print(f"loaded alpha from {alpha_path}")
        else:
            print("no checkpoint for alpha")
        beta_path = os.path.join(self.model_config['model_name_or_path'], 'beta.pt')
        if os.path.exists(beta_path):
            self.beta =  torch.load(os.path.join(beta_path))
            print(f"loaded beta from {beta_path}")
        else:
            print("no checkpoint for beta")
        gamma_path = os.path.join(self.model_config['model_name_or_path'], 'gamma.pt')
        if os.path.exists(gamma_path):
            # self.gamma.load_state_dict(torch.load(gamma_path))
            self.gamma =  torch.load(os.path.join(gamma_path))
            print(f"loaded gamma from {gamma_path}")
        else:
            print("no checkpoint for gamma")

        cls_norm_path = os.path.join(self.model_config['model_name_or_path'], 'cls_norm.pt')
        if os.path.exists(cls_norm_path):
            self.cls_norm.load_state_dict(torch.load(cls_norm_path))
            print(f"loaded cls_norm from {cls_norm_path}")
        else:
            print("no checkpoint for cls_norm")

        feat_norm_path = os.path.join(self.model_config['model_name_or_path'], 'feat_norm.pt')
        if os.path.exists(feat_norm_path):
            self.feat_norm.load_state_dict(torch.load(feat_norm_path))
            print(f"loaded feat_norm from {feat_norm_path}")
        else:
            print("no checkpoint for feat_norm")

        user_norm_path = os.path.join(self.model_config['model_name_or_path'], 'user_norm.pt')
        if os.path.exists(user_norm_path):
            self.user_norm.load_state_dict(torch.load(user_norm_path))
            print(f"loaded user_norm from {user_norm_path}")
        else:
            print("no checkpoint for user_norm")

        return model

    def _setup_lora(self, model):
        # 不会运行这里的代码
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        if os.path.exists(os.path.join(self.model_config['lora_checkpoint_dir'], 'adapter_config.json')):
            # model.load_adapter(self.model_config['lora_checkpoint_dir'], 'cross_encoder')
            # print("Load cross_encoder lora adapter from", self.model_config['lora_checkpoint_dir'])
            model = PeftModel.from_pretrained(model, self.model_config['lora_checkpoint_dir'])
            print(f"Loaded LORA from {self.model_config['lora_checkpoint_dir']}")
            for name, param in model.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True  

            # #全量微调
            # model = model.merge_and_unload()
            # print("全量微调！")
            # for name, param in model.named_parameters():  
            #     param.requires_grad = True 

        else:
            print('No lora adapter found, add lora adapter from init')
            peft_config = LoraConfig(
                lora_alpha=128,
                lora_dropout=0.1,
                r=64,
                bias='none',
                # task_type="CAUSAL_LM",
                task_type="FEATURE_EXTRACTION",
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                target_modules = ["query", "key", "value"]
            )
            # model.add_adapter(peft_config, "cross_encoder")
            # 新增
            model = get_peft_model(model, peft_config)
            print('Add cross_encoder lora adapter from init')

        if isinstance(model, PeftModel):
            print("model is a lora model")
        else:
            print("model is NOT a lora model ")
        return model

    def save_pretrained(self, save_path):
        # print(f"正在 Save model to {save_path}")
        # self.model.save_pretrained(save_path)
        # # self.tokenizer.save_pretrained(save_path)
        # classifier_path = os.path.join(save_path, 'classifier.pt')
        # torch.save(self.classifier.state_dict(), classifier_path)
        # # # 保存 binary_encoders（新增部分）
        # binary_encoders_path = os.path.join(save_path, 'binary_encoders.pt')
        # torch.save(self.binary_encoders.state_dict(), binary_encoders_path)
        # user_binary_encoders_path = os.path.join(save_path, 'user_binary_encoders.pt')
        # torch.save(self.user_binary_encoders.state_dict(), user_binary_encoders_path)
        # # RMSNorm 参数
        # torch.save(self.cls_norm.state_dict(), os.path.join(save_path, 'cls_norm.pt'))
        # torch.save(self.feat_norm.state_dict(), os.path.join(save_path, 'feat_norm.pt'))
        # torch.save(self.user_norm.state_dict(), os.path.join(save_path, 'user_norm.pt'))
        # # Batchnrom
        # # torch.save(self.batch_norm1.state_dict(), os.path.join(save_path, 'batch_norm1.pt'))
        # # torch.save(self.batch_norm2.state_dict(), os.path.join(save_path, 'batch_norm2.pt'))
        # # torch.save(self.batch_norm3.state_dict(), os.path.join(save_path, 'batch_norm3.pt'))

        # # torch.save(self.x_norm.state_dict(), os.path.join(save_path, 'x_norm.pt'))

        # torch.save(self.alpha, os.path.join(save_path, 'alpha.pt'))
        # torch.save(self.beta, os.path.join(save_path, 'beta.pt'))
        # torch.save(self.gamma, os.path.join(save_path, 'gamma.pt'))
        pass

    def forward(self, batch_features,user_feat, **inputs):
        # 遍历所有子模块，转换为 float16
        self.model = self.model.to(torch.float16)
        for feat, encoder in self.binary_encoders.items():
            encoder.to(torch.float16) 
        for feat, encoder in self.user_binary_encoders.items():
            encoder.to(torch.float16)
        # self.alpha = self.alpha.to(torch.float16)
        # self.beta = self.beta.to(torch.float16)
        # self.gamma = self.gamma.to(torch.float16)
        self.cls_norm = self.cls_norm.to(torch.float16)
        self.feat_norm = self.feat_norm.to(torch.float16)
        self.user_norm = self.user_norm.to(torch.float16)

        outputs = self.model(**inputs)
        # cls
        last_hidden_states = outputs.last_hidden_state
        cls_output = last_hidden_states[:, 0, :]
        
        # 使用平均池化
        # last_hidden_states = outputs.last_hidden_state
        # attention_mask = inputs.get("attention_mask")  # 获取attention mask
        # mean_embeddings = self.mean_pooling(last_hidden_states, attention_mask)

        # shape:[num1+num2, H]
        # logits = self.classifier(cls_output)

        # 构造 feature tensor： [B, F]
        feat_tensors = []
        for feat, encoder in self.binary_encoders.items():
            feat_tensor = batch_features[feat]
            # feat_tensor是一个张量列表，每一个张量代表当前数值的二进制转化后的张量
            # print(f"feat_tensor:{feat_tensor}")
            # print(f"feat_tensor.shape:{feat_tensor.shape}")
            encoded = torch.stack([encoder(i) for i in feat_tensor], dim=0)  # encoded 形状为 [B, 20]，B为正负样本总数，N为特征编码后的维度
            # feat_tensor的每一个元素维度统一为 20
            feat_tensors.append(encoded)

        # print(f"len(feat_tensors):{len(feat_tensors)}") # 5
        # print(f"feat_tensors[0][0].shape:{feat_tensors[0][0].shape}") # 15
        # feat_tensor = torch.stack(feat_tensors, dim=1).float()  # [B, F]

        # 拼接所有特征：[B, 5*20] = [B, 100]
        feat_tensor = torch.cat(feat_tensors, dim=1)  # [B, 100]，B为正负样本总数，100为所有特征编码后的维度

        # print(f"feat_tensor.shape:{feat_tensor.shape}")
        # 当 dim=1 时，堆叠操作会在第 1 维（列方向）合并这些张量，结果形状为 [2N, 5]

        User_feats = []
        for feat, encoder in self.user_binary_encoders.items():
            user_feat_tensor = user_feat[feat]
            # feat_tensor是一个张量列表，每一个张量代表当前数值的二进制转化后的张量
            # print(f"feat_tensor:{feat_tensor}")
            # print(f"feat_tensor.shape:{feat_tensor.shape}")
            # encoded = [encoder(i) for i in feat_tensor]
            encoded = torch.stack([encoder(i) for i in user_feat_tensor], dim=0)  # encoded 形状为 [B, 20]
            # feat_tensor的每一个元素维度统一为 20
            User_feats.append(encoded)

        User_feats = torch.cat(User_feats, dim=1) # [B, X]，B为正负样本总数，X为所有特征编码后的维度
        # print(f"User_feats.shape:{User_feats.shape}")

        cls_output = self.cls_norm(cls_output)         # [B, H]
        # cls_output = F.normalize(cls_output, p=2, dim=1)  # L2 归一化
        # cls_output = self.batch_norm1(cls_output)
        # mean = cls_output.mean(dim=1, keepdim=True)
        # std = cls_output.std(dim=1, keepdim=True)
        # cls_output = (cls_output - mean) / (std + 1e-6)

        feat_tensor = self.feat_norm(feat_tensor)      # [B, F1]
        User_feats = self.user_norm(User_feats)        # [B, F2]
        # feat_tensor = F.normalize(feat_tensor, p=2, dim=1)
        # User_feats = F.normalize(User_feats, p=2, dim=1)
        # feat_tensor = self.batch_norm2(feat_tensor)
        # User_feats = self.batch_norm3(User_feats)

        cls_output = self.alpha * cls_output
        feat_tensor = self.beta * feat_tensor
        User_feats = self.gamma * User_feats
        # print("cls_output dtype:", cls_output.dtype)  # 输出数据类型
        # print("feat_tensor dtype:", feat_tensor.dtype)  # 输出数据类型
        # print("User_feats dtype:", User_feats.dtype)  # 输出数据类型
        # 新拼接并分类
        concat = torch.cat([cls_output, feat_tensor], dim=1)  # [B, H+F]
        concat = torch.cat([concat, User_feats], dim=1)  # [B, H+F]
        # ƒconcat = self.x_norm(concat)

        # x = concat.unsqueeze(1) # 从 [B, H+F] -> [B, 1, H+F]

        # x = cls_output.unsqueeze(1)
        # x = mean_embeddings.unsqueeze(1)

        ############ 特征分析模块，可插拔
        # if is_main_process():
        #     # 去掉第二维（1） -> [B, H+F]
        #     x_2d = x.squeeze(1)
        #     HF=  x_2d.shape[1] #[B, H+F]
        #     # 计算每个特征维度上的统计信息
        #     mean_per_dim = x_2d.mean(dim=0).detach().cpu().numpy()
        #     std_per_dim = x_2d.std(dim=0).detach().cpu().numpy()
        #     max_per_dim = x_2d.max(dim=0)[0].detach().cpu().numpy()
        #     min_per_dim = x_2d.min(dim=0)[0].detach().cpu().numpy()
        #     var_per_dim = x_2d.var(dim=0).detach().cpu().numpy()

        #     # 打印数值范围（可选）
        #     # print("特征维度上的均值范围：", mean_per_dim.min(), "~", mean_per_dim.max())
        #     # print("特征维度上的方差范围：", var_per_dim.min(), "~", var_per_dim.max())

        #     # 可视化：特征维度上的均值和标准差
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(mean_per_dim, label='Mean')
        #     plt.plot(std_per_dim, label='Std')
        #     plt.fill_between(range(HF), mean_per_dim - std_per_dim, mean_per_dim + std_per_dim, color='lightblue', alpha=0.4)
        #     plt.title("mean & std")
        #     plt.xlabel("index")
        #     plt.ylabel("value")
        #     plt.legend()
        #     plt.grid(True)
        #     # 添加三条红色虚线（垂直线）
        #     # for x_pos in [768, 768 + 118, 768 + 118 + 244]:
        #     #     plt.axvline(x=x_pos, color='red', linestyle='--', linewidth=0.5)
        #     plt.tight_layout()
        #     plt.savefig("output/x_visavle.png")
        ############ 特征分析模块，可插拔
        # print("concat dtype:", concat.dtype)  # 输出数据类型
        logits = self.classifier(concat).squeeze(-1)          # [B]
        # logits = self.classifier(x).squeeze(-1)          # [B]
        # print(f"concat.shape:{concat.shape}")
        # concat.shape:torch.Size([16, 773])
        
        return logits

class GatedVisualTextFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # concat 后维度翻倍，Linear 默认包含 bias
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual_feat, text_feat):
        """
        :param visual_feat: [batch_size, embed_dim]
        :param text_feat: [batch_size, embed_dim]
        :return: fused_feat [batch_size, embed_dim]
        """
        concat_feat = torch.cat([visual_feat, text_feat], dim=-1)  # [B, 2*D]
        gate_logits = self.gate_linear(concat_feat)  # [B, D], 包含 bias
        gate = self.sigmoid(gate_logits)  # [B, D], gate value ∈ (0,1)

        # gated weighted fusion
        fused_feat = gate * visual_feat + (1 - gate) * text_feat
        return fused_feat

class VLMCrossEncoderModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        torch.backends.cudnn.enabled = False
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_config['model_name_or_path'],
            trust_remote_code=True,
            load_in_4bit=self.model_config['load_in_4bit'],
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        )
        
        self.vlm_hidde_size = 1898
        # 定义分类器结构（至少3层线性层）
        Attention_config = {
            "hidden_size": self.vlm_hidde_size,
            "intermediate_size": 4 * self.vlm_hidde_size # 通常是 hidden_size 的 4 倍
        }
        self.classifier = VLM_TransformerClassifier(
            input_dim=self.vlm_hidde_size, 
            n_layers=6, 
            n_heads=13,
            config=Attention_config
            )


        self.model_config['lora_checkpoint_dir']="model/Submodel_ckpt/2025-07-11-16-57-46/vlm_checkpoints"
        # 创建文件夹，如果已存在则不报错
        os.makedirs(self.model_config['lora_checkpoint_dir'], exist_ok=True)
        classifier_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'classifier_base.pt')
        if dist.is_initialized():
            dist.barrier()
            print("等待同时尝试加载")
        if os.path.exists(classifier_base_ckpt):
            self.classifier.load_state_dict(torch.load(classifier_base_ckpt))
            print("从先前的base model 加载classifier")
        else:
            if dist.is_initialized():
                dist.barrier()
            if is_main_process():
                torch.save(self.classifier.state_dict(), classifier_base_ckpt)
                print("Saved classifier and base_model.")
        if dist.is_initialized():
            dist.barrier()
            print("主进程保存成功,继续运行")

        #笔记统计特征
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
        # 对字典中的值进行 log2 操作，并取上整
        self.feature_max_values = {
            key: np.ceil(np.log2(value)) if value > 0 else value
            for key, value in self.feature_max_values.items()
        }
        self.feature_bits={feat: len(bin(int(max_value))) - 2 for feat, max_value in self.feature_max_values.items()}
        self.binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True), #添加两层全连接层，增强模型表达能力
                # nn.ReLU(),
                # nn.Linear(total_bits*2, total_bits*4, bias=True),
                # nn.ReLU(),
                # nn.Linear(total_bits*4, total_bits*2, bias=True)
            ) for feat, total_bits in self.feature_bits.items()
        })
        print("已初始化 binary_encoders")
        # 用户特征
        self.all_feat={'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}
        self.user_feat_bits={feat: len(bin(int(max_value)))- 2 for feat, max_value in self.all_feat.items()}
        self.user_feat_bits["age"]=11
        self.user_feat_bits["gender"]=2
        self.user_binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True), #添加两层全连接层，增强模型表达能力
                # nn.ReLU(),
                # nn.Linear(total_bits*2, total_bits*4, bias=True),
                # nn.ReLU(),
                # nn.Linear(total_bits*4, total_bits*2, bias=True)
            ) for feat, total_bits in self.user_feat_bits.items()
        })
        print("user_binary_encoders已初始化完成") 

        self.cls_norm = RMSNorm(1536)  # hidden_size 是 mean_pool 的维度
        self.feat_norm = RMSNorm(118)  # feat_dim 是 feat_tensor 的维度，例如 100
        self.user_norm = RMSNorm(244)
        # init_weights=(1.0, 1.0, 1.0)
        # self.alpha = nn.Parameter(torch.tensor(init_weights[0], dtype=torch.float16))
        # self.beta = nn.Parameter(torch.tensor(init_weights[1], dtype=torch.float16))
        # self.gamma = nn.Parameter(torch.tensor(init_weights[2], dtype=torch.float16))
        embed_dim = 1536
        self.fusion_module = GatedVisualTextFusion(embed_dim=embed_dim)
        print("早期融合和晚期融合结合!")

        fusion_module_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'fusion_module_base.pt')
        if dist.is_initialized():
            dist.barrier()
            print("等待同时尝试加载")
        if os.path.exists(fusion_module_base_ckpt):
            self.fusion_module.load_state_dict(torch.load(fusion_module_base_ckpt))
            print("从先前的base model 加载 fusion_module")
        else:
            if dist.is_initialized():
                dist.barrier()
            if is_main_process():
                torch.save(self.fusion_module.state_dict(),fusion_module_base_ckpt )
                print("Saved fusion_module and base_model.")
        if dist.is_initialized():
            dist.barrier()
            print("主进程保存成功,继续运行")

        # #加载检查点 
        binary_encoders_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'binary_encoders.pt')
        if os.path.exists(binary_encoders_path):
            self.binary_encoders.load_state_dict(torch.load(binary_encoders_path))
            print(f"已加载 binary_encoders from {binary_encoders_path}")
        else:
            print("初始化 binary_encoders from init")

        user_binary_encoders_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'user_binary_encoders.pt')
        if os.path.exists(user_binary_encoders_path):
            self.user_binary_encoders.load_state_dict(torch.load(user_binary_encoders_path))
            print(f"Loaded user_binary_encoders_path parameters from {user_binary_encoders_path}")
            print("已加载 user_binary_encoders_path")
        else:
            print("初始化 user_binary_encoders_path from init")

        # alpha_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'alpha.pt')
        # if os.path.exists(alpha_path):
        #     self.alpha = torch.load(alpha_path)
        #     print(f"loaded alpha from {alpha_path}")
        # else:
        #     print("no checkpoint for alpha")
        # beta_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'beta.pt')
        # if os.path.exists(beta_path):
        #     self.beta =  torch.load(beta_path)
        #     print(f"loaded beta from {beta_path}")
        # else:
        #     print("no checkpoint for beta")
        # gamma_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'gamma.pt')
        # if os.path.exists(gamma_path):
        #     self.gamma =  torch.load(gamma_path)
        #     print(f"loaded gamma from {gamma_path}")
        # else:
        #     print("no checkpoint for gamma")

        cls_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'cls_norm.pt')
        if os.path.exists(cls_norm_path):
            self.cls_norm.load_state_dict(torch.load(cls_norm_path))
            print(f"loaded cls_norm from {cls_norm_path}")
        else:
            print("no checkpoint for cls_norm")

        feat_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'feat_norm.pt')
        if os.path.exists(feat_norm_path):
            self.feat_norm.load_state_dict(torch.load(feat_norm_path))
            print(f"loaded feat_norm from {feat_norm_path}")
        else:
            print("no checkpoint for feat_norm")

        user_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'user_norm.pt')
        if os.path.exists(user_norm_path):
            self.user_norm.load_state_dict(torch.load(user_norm_path))
            print(f"loaded user_norm from {user_norm_path}")
        else:
            print("no checkpoint for user_norm")

        # fusion_module_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'fusion_module_checkpoint.pt')
        # if os.path.exists(fusion_module_path):
        #     self.fusion_module.load_state_dict(torch.load(fusion_module_path))
        # else:
        #     print("no checkpoint for fusion_module")

        for param in self.classifier.parameters():
            param.requires_grad = True
        if self.model_config['use_lora']:
            self.model,self.classifier,self.fusion_module =self._setup_lora(self.model,self.classifier,self.fusion_module)

        for encoder in self.binary_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        for encoder in self.user_binary_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        for param in self.cls_norm.parameters():
            param.requires_grad = True
        for param in self.feat_norm.parameters():
            param.requires_grad = True
        for param in self.user_norm.parameters():
            param.requires_grad = True
        for param in self.model.visual.parameters():
            param.requires_grad = False

        if self.model_config['gradient_checkpointing']:
            print("开启梯度检查点")
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        else:
            print("梯度检查点关闭")


    def _setup_lora(self, model, classifier,fusion_module):
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        VLM_lora_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'VLM_lora')
        if os.path.exists(os.path.join(VLM_lora_path, 'adapter_config.json')):
            # model.load_adapter(self.model_config['lora_checkpoint_dir'], 'cross_encoder', is_trainable=True)
            # print("Load cross_encoder lora adapter from", self.model_config['lora_checkpoint_dir'])
            model = PeftModel.from_pretrained(model, VLM_lora_path)
            print(f"Loaded VLM LORA from {VLM_lora_path}")
            for name, param in model.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True  
        else:
            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
            )
            #model.add_adapter(peft_config, "cross_encoder")
            # 新增
            model = PeftModel(model, peft_config)
            for name, param in model.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True  
            print('Add VLM lora adapter from init')

        # for param in model.parameters():
        #     param.requires_grad = False
        # print("冻结VLM!")
        for name, param in model.named_parameters():
            if "vision_encoder" in name:
                print(f"{name} requires_grad: {param.requires_grad}")

        # ----------------------------
        # Step 2: Setup LoRA for the classifier
        # ----------------------------
        print(f"Classifier: Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        classifier_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'classifier_base.pt')
        if os.path.exists(classifier_base_ckpt):
            # if isinstance(classifier, PeftModel):
            #     print("classifier is a lora model")
            # else:
            #     print("classifier is NOT a lora model ")
            classifier.load_state_dict(torch.load(classifier_base_ckpt))
            # 初始化 LoRA 权重（注意不使用 from_pretrained）
            peft_classifier_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["input_proj","gate_proj", "up_proj", "down_proj","qkv_proj","out_proj","common_proj"],  # LoRA on these modules
            )
            classifier = PeftModel(classifier, peft_config=peft_classifier_config)

            # 加载 LoRA adapter 权重
            classifier_lora_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], "classifier_lora")
            if os.path.exists(classifier_lora_ckpt):
                classifier.load_adapter(classifier_lora_ckpt, adapter_name="default", is_trainable=True)
                print(f"Loaded classifier LoRA adapter from {classifier_lora_ckpt}")
            else:
                print("No LoRA adapter found, training from scratch LoRA weights.")

            for name, param in classifier.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    
        else:
            peft_classifier_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["input_proj","gate_proj", "up_proj", "down_proj","qkv_proj","out_proj","common_proj"],  # LoRA on these modules
            )
            classifier = PeftModel(classifier, peft_classifier_config)
            for name, param in classifier.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
            print('Add LoRA adapter to classifier from init')

        print(f"fusion_module: Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        fusion_module_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'fusion_module_base.pt')
        if os.path.exists(fusion_module_base_ckpt):
            fusion_module.load_state_dict(torch.load(fusion_module_base_ckpt))
            # 初始化 LoRA 权重（注意不使用 from_pretrained）
            peft_fusion_module_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_linear"],  # LoRA on these modules
            )
            fusion_module = PeftModel(fusion_module, peft_config=peft_fusion_module_config)

            # 加载 LoRA adapter 权重
            fusion_module_lora_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], "fusion_module_lora")
            if os.path.exists(fusion_module_lora_ckpt):
                fusion_module.load_adapter(fusion_module_lora_ckpt, adapter_name="default", is_trainable=True)
                print(f"Loaded fusion_module LoRA adapter from {fusion_module_lora_ckpt}")
            else:
                print("No LoRA adapter found, training from scratch LoRA weights.")

            for name, param in fusion_module.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    
        else:
            peft_fusion_module_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=8,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_linear"],  # LoRA on these modules
            )
            fusion_module = PeftModel(fusion_module, peft_fusion_module_config)
            for name, param in fusion_module.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
            print('Add LoRA adapter to fusion_module from init')


        if isinstance(model, PeftModel):
            print("model is a lora model")
        else:
            print("model is NOT a lora model ")
        if isinstance(classifier, PeftModel):
            print("classifier is a lora model")
        else:
            print("classifier is NOT a lora model ")
        if isinstance(fusion_module, PeftModel):
            print("fusion_module is a lora model")
        else:
            print("fusion_module is NOT a lora model ")

        return model,classifier,fusion_module

    def forward(self, batch_features,user_feat, **inputs): 
        # 默认FP32
        self.model = self.model.to(torch.float16)
        self.classifier = self.classifier.to(torch.float16)
        self.user_binary_encoders = self.user_binary_encoders.to(torch.float16)
        self.binary_encoders = self.binary_encoders.to(torch.float16)
        self.fusion_module = self.fusion_module.to(torch.float16)
        self.cls_norm = self.cls_norm.to(torch.float16)
        self.feat_norm = self.feat_norm.to(torch.float16)
        self.user_norm = self.user_norm.to(torch.float16)

        inputs["pixel_values"].requires_grad_()
        # for key, value in inputs.items():
        #     # 打印键和对应值的形状
        #     print(f"Key: {key}, Shape: {value.shape}")
            # Key: input_ids, Shape: torch.Size([5, 512])
            # Key: attention_mask, Shape: torch.Size([5, 512])
            # Key: pixel_values, Shape: torch.Size([2000, 1176])
            # Key: image_grid_thw, Shape: torch.Size([5, 3])

        # outputs = self.model(**inputs, output_hidden_states=True)
        # hidden_states = outputs.hidden_states[-1]
        # features = mean_token_pool(
        #     last_hidden_states=hidden_states,
        #     attention_mask=inputs['attention_mask']
        # )

        #======================================================
        visual_features = self.model.visual(hidden_states=inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])  # ViT输出
        # print(f"split 之前 visual_features.shape : {visual_features.shape}")
        # 经过 split 之前，visual_features.shape : torch.Size([400, 1536])
        # image_grid_thw.prod(-1) 用于计算张量在最后一个维度，计算每个图像的总块数（T*H*W）
        split_sizes = (inputs["image_grid_thw"].prod(-1) // self.model.visual.spatial_merge_size**2).tolist()
        # print(f"split_sizes:{split_sizes}") [100,100,..,100]
        visual_features = torch.split(visual_features, split_sizes)
        # 经过 split 之后是一个元组，代表一个 batch 内每一个图像的 token 的 embedding
        # split 之后: {[f.shape for f in visual_features]}")  [torch.Size([100, 1536])
        # Qwen2-vl 中 已知模型的图片最大分辨率为280*280，patchsize为14 merge size为2，那么一个图片要占token 数=(H/PM)*(W/PM)=100 个

        #图像 token，将每100个图像token压缩成一个token
        visual_features_stacked = torch.stack(visual_features)  # 堆叠 → [batch_size, 100, 1536]
        # visual_features_stacked.shape:torch.Size([4, 100, 1536])
        pooled_features = visual_features_stacked.mean(dim=1)  # 在token维度(dim=1)求平均 → [batch_size, 1536]

        # 文本token
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # print(f"hidden_states.shape:{hidden_states.shape}")
        # hidden_states.shape:torch.Size([13, 512, 1536]) 13个 note
        text_feature = hidden_states[:,-1]

        #早期融合和晚期融合
        pooled_features.requires_grad_()
        text_feature.requires_grad_()

        features = self.fusion_module(pooled_features, text_feature)
        # features = features.to(self.classifier.base_model.model.layers[0].mlp.up_proj.lora_A.default.weight)
        target_dtype = self.classifier.layers[0].mlp.up_proj.weight.dtype
        # target_dtype = self.classifier.base_model.model.layers[0].mlp.up_proj.lora_A.default.weight.dtype
        if features.dtype != target_dtype:
            features = features.to(dtype=target_dtype)
        #======================================================
        # print(f"features.shape:{features.shape}") # [B,1536]

        # # 构造 feature tensor： [B, F]
        feat_tensors = []
        # 获取当前 rank（若未初始化 dist，则默认为 0）

        for feat, encoder in self.binary_encoders.items():
            feat_tensor = batch_features[feat]
            # feat_tensor是一个张量列表，每一个张量代表当前数值的二进制转化后的张量
            # print(f"feat_tensor:{feat_tensor}")
            # print(f"feat_tensor.shape:{feat_tensor.shape}")
            encoded = torch.stack([encoder(i) for i in feat_tensor], dim=0)  # encoded 形状为 [B, 20]，B为正负样本总数，N为特征编码后的维度

            # encoded_list = []
            # for i in feat_tensor:
            #     # print(f"i.dtype:{i.dtype}") #i.dtype:torch.float16
            #     encoded_i = encoder(i)
            #     # for n, p in encoder.named_parameters():
            #     #     print(n, p.device, p.dtype)
            #     # 0.weight cuda:0 torch.float16
            #     # 0.bias cuda:0 torch.float16
            #     # print(f"encoded_i.dtype:{encoded_i.dtype}") 
            #     #encoded_i.dtype:torch.bfloat16
            #     encoded_list.append(encoded_i)
            # encoded = torch.stack(encoded_list, dim=0)  # encoded 形状为 [B, 20]
            # # feat_tensor的每一个元素维度统一为 20

            feat_tensors.append(encoded)
        # 拼接所有特征：[B, 5*20] = [B, 100]
        feat_tensor = torch.cat(feat_tensors, dim=1)  # [B, 100]，B为正负样本总数，100为所有特征编码后的维度
        # print(type(feat_tensor), isinstance(feat_tensor, torch.Tensor))
        # print("feat_tensor.requires_grad:", feat_tensor.requires_grad)

        User_feats = []
        for feat, encoder in self.user_binary_encoders.items():
            user_feat_tensor = user_feat[feat]
            # feat_tensor是一个张量列表，每一个张量代表当前数值的二进制转化后的张量
            encoded = torch.stack([encoder(i) for i in user_feat_tensor], dim=0)  # encoded 形状为 [B, 20]
            # encoded_list = []
            # for i in user_feat_tensor:
            #     encoded_i = encoder(i)

            #     # print(f"user_binary_encoders: {feat}, input requires_grad: {i.requires_grad}, output requires_grad: {encoded_i.requires_grad}")
            #     encoded_list.append(encoded_i)
            # encoded = torch.stack(encoded_list, dim=0)  # encoded 形状为 [B, 20]
            # feat_tensor的每一个元素维度统一为 20
            User_feats.append(encoded)
        User_feats = torch.cat(User_feats, dim=1) # [B, X]，B为正负样本总数，X为所有特征编码后的维度

        features = self.cls_norm(features)         # [B, H]
        feat_tensor = self.feat_norm(feat_tensor)      # [B, F1]
        User_feats = self.user_norm(User_feats)        # [B, F2]

        # features = self.alpha * features
        # feat_tensor = self.beta * feat_tensor
        # User_feats = self.gamma * User_feats

        # target_dtype = self.classifier.layers[0].mlp.up_proj.weight.dtype
        target_dtype = torch.float16
        if features.dtype != target_dtype:
            features = features.to(dtype=target_dtype)

        concat1 = torch.cat([features, feat_tensor], dim=1)  # [B, H+F]
        # # print(f"concat1.shape:{concat.shape}")
        concat = torch.cat([concat1, User_feats], dim=1)  # [B, H+F]
        # # print(f"concat2.shape:{concat.shape}")
        x = concat.unsqueeze(1) # 从 [B, H+F] -> [B, 1, H+F]
        logits = self.classifier(x).squeeze(-1)          # [B]

        return logits

    def save_pretrained(self, save_path):
        # self.model.save_pretrained(os.path.join(save_path, "VLM_lora"))

        # # Merge LoRA adapter into base model
        # if isinstance(self.classifier, PeftModel):
        #     self.classifier.save_pretrained(os.path.join(save_path, "classifier_lora"))
        #     print("Saved classifier LoRA adapter.")

        # # 保存 binary_encoders（新增部分）
        # binary_encoders_path = os.path.join(save_path, 'binary_encoders.pt')
        # torch.save(self.binary_encoders.state_dict(), binary_encoders_path)
        # user_binary_encoders_path = os.path.join(save_path, 'user_binary_encoders.pt')
        # torch.save(self.user_binary_encoders.state_dict(), user_binary_encoders_path)
        # # RMSNorm 参数
        # torch.save(self.cls_norm.state_dict(), os.path.join(save_path, 'cls_norm.pt'))
        # torch.save(self.feat_norm.state_dict(), os.path.join(save_path, 'feat_norm.pt'))
        # torch.save(self.user_norm.state_dict(), os.path.join(save_path, 'user_norm.pt'))
        # # 权重
        # # torch.save(self.alpha.data.cpu(), os.path.join(save_path, 'alpha.pt'))
        # # torch.save(self.beta.data.cpu(), os.path.join(save_path, 'beta.pt'))
        # # torch.save(self.gamma.data.cpu(), os.path.join(save_path, 'gamma.pt'))

        # # 融合模块
        # if isinstance(self.fusion_module, PeftModel):
        #     self.fusion_module.save_pretrained(os.path.join(save_path, "fusion_module_lora"))
        #     print("Saved fusion_module LoRA adapter.")
        pass

class SelfAttentionBlock(nn.Module):
    def __init__(self, user_dim,dim, n_heads, config=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CrossAttention(user_dim,dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(config=config)

    def forward(self, user_feat,x):
        x = x + self.attn(user_feat, self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SelfAttentionClassifier(nn.Module):
    def __init__(self,user_dim=100, input_dim=1442, n_layers=2, n_heads=3, config=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, input_dim, bias=True)
        self.layers = nn.ModuleList([
            SelfAttentionBlock(user_dim,input_dim, n_heads, config=config)
            for _ in range(n_layers)
        ])
        self.common_proj = nn.Linear(input_dim, 1024, bias=True)
        self.act_fn = nn.SiLU()
        self.out_proj = nn.Linear(1024, 1, bias=True)

    def forward(self, User_feats, x):  # x shape: (batch, seq_len, dim)
        x = self.input_proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(User_feats, x)
        x = self.common_proj(x)
        x = self.act_fn(x)
        return self.out_proj(x)

class GatedVisualTextFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # concat 后维度翻倍，Linear 默认包含 bias
        self.gate_linear = nn.Linear(embed_dim * 2, embed_dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual_feat, text_feat):
        """
        :param visual_feat: [batch_size, embed_dim]
        :param text_feat: [batch_size, embed_dim]
        :return: fused_feat [batch_size, embed_dim]
        """
        concat_feat = torch.cat([visual_feat, text_feat], dim=-1)  # [B, 2*D]
        gate_logits = self.gate_linear(concat_feat)  # [B, D], 包含 bias
        gate = self.sigmoid(gate_logits)  # [B, D], gate value ∈ (0,1)

        # gated weighted fusion
        fused_feat = gate * visual_feat + (1 - gate) * text_feat
        return fused_feat

class MultiModalRankModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_config['model_name_or_path'],
            trust_remote_code=True,
            load_in_4bit=self.model_config['load_in_4bit']
        )

        # 定义分类器结构（至少3层线性层）
        user_dim= 244
        all_hidden_size = 1654
        Attention_config = {
            "hidden_size": all_hidden_size+user_dim,
            "intermediate_size": 4 * all_hidden_size # 通常是 hidden_size 的 4 倍
        }
        self.classifier = SelfAttentionClassifier(
            user_dim=user_dim,
            input_dim=all_hidden_size, 
            n_layers=6, 
            n_heads=2,
            config=Attention_config
            )        
        
        # 创建文件夹，如果已存在则不报错
        os.makedirs(self.model_config['lora_checkpoint_dir'], exist_ok=True)
        classifier_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'classifier_base.pt')
        if dist.is_initialized():
            dist.barrier()
            print("等待同时尝试加载")
        if os.path.exists(classifier_base_ckpt):
            self.classifier.load_state_dict(torch.load(classifier_base_ckpt))
            print("从先前的base model 加载classifier")
        else:
            if dist.is_initialized():
                dist.barrier()
            if is_main_process():
                torch.save(self.classifier.state_dict(), classifier_base_ckpt)
                print("Saved classifier and base_model.")
        if dist.is_initialized():
            dist.barrier()
            print("主进程保存成功,继续运行")

        # 笔记统计特征
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
        # 对字典中的值进行 log2 操作，并取上整
        self.feature_max_values = {
            key: np.ceil(np.log2(value)) if value > 0 else value
            for key, value in self.feature_max_values.items()
        }
        self.feature_bits={feat: len(bin(int(max_value))) - 2 for feat, max_value in self.feature_max_values.items()}
        self.binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True), #添加两层全连接层，增强模型表达能力
            ) for feat, total_bits in self.feature_bits.items()
        })
        print("已初始化 binary_encoders")
        # 用户特征
        self.all_feat={'dense_feat9': 28.0, 'dense_feat26': 15.0, 'dense_feat37': 20.0, 'dense_feat34': 20.0, 'dense_feat25': 28.0, 'dense_feat11': 28.0, 'dense_feat20': 15.0, 'dense_feat13': 28.0, 'dense_feat10': 15.0, 'dense_feat14': 13.0, 'dense_feat24': 14.0, 'dense_feat1': 14.0, 'dense_feat33': 20.0, 'dense_feat28': 16.0, 'dense_feat36': 16.0, 'follows_num': 16.0, 'dense_feat2': 13.0, 'dense_feat38': 7.0, 'dense_feat18': 17.0, 'dense_feat32': 20.0, 'dense_feat12': 15.0, 'dense_feat35': 20.0, 'dense_feat31': 20.0, 'dense_feat8': 15.0}
        self.user_feat_bits={feat: len(bin(int(max_value))) - 2 for feat, max_value in self.all_feat.items()}
        self.user_feat_bits["age"]=11
        self.user_feat_bits["gender"]=2
        self.user_binary_encoders = nn.ModuleDict({
            feat: nn.Sequential(
                nn.Linear(total_bits, total_bits*2, bias=True), #添加两层全连接层，增强模型表达能力
            ) for feat, total_bits in self.user_feat_bits.items()
        })
        print("user_binary_encoders已初始化完成") 

        self.cls_norm = RMSNorm(1536)  # hidden_size 是 mean_pool 的维度
        self.feat_norm = RMSNorm(118)  # feat_dim 是 feat_tensor 的维度，例如 100
        self.user_norm = RMSNorm(244)
        embed_dim = 1536
        self.fusion_module = GatedVisualTextFusion(embed_dim=embed_dim)
        print("早期融合和晚期融合结合!")

        fusion_module_base_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], 'fusion_module_base.pt')
        if dist.is_initialized():
            dist.barrier()
            print("等待同时尝试加载")
        if os.path.exists(fusion_module_base_ckpt):
            self.fusion_module.load_state_dict(torch.load(fusion_module_base_ckpt))
            print("从先前的base model 加载 fusion_module")
        else:
            if dist.is_initialized():
                dist.barrier()
            if is_main_process():
                torch.save(self.fusion_module.state_dict(),fusion_module_base_ckpt )
                print("Saved fusion_module and base_model.")
        if dist.is_initialized():
            dist.barrier()
            print("主进程保存成功,继续运行")

        #加载检查点 
        binary_encoders_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'binary_encoders.pt')
        if os.path.exists(binary_encoders_path):
            self.binary_encoders.load_state_dict(torch.load(binary_encoders_path))
            print(f"Loaded binary_encoders parameters from {binary_encoders_path}")
            print("已加载 binary_encoders")
        else:
            print("初始化 binary_encoders from init")

        user_binary_encoders_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'user_binary_encoders.pt')
        if os.path.exists(user_binary_encoders_path):
            self.user_binary_encoders.load_state_dict(torch.load(user_binary_encoders_path))
            print(f"Loaded user_binary_encoders parameters from {user_binary_encoders_path}")
            print("已加载 user_binary_encoders_path")
        else:
            print("初始化 user_binary_encoders from init")     

        cls_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'cls_norm.pt')
        if os.path.exists(cls_norm_path):
            self.cls_norm.load_state_dict(torch.load(cls_norm_path))
            print(f"loaded cls_norm from {cls_norm_path}")
        else:
            print("no checkpoint for cls_norm")

        feat_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'feat_norm.pt')
        if os.path.exists(feat_norm_path):
            self.feat_norm.load_state_dict(torch.load(feat_norm_path))
            print(f"loaded feat_norm from {feat_norm_path}")
        else:
            print("no checkpoint for feat_norm")

        user_norm_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'user_norm.pt')
        if os.path.exists(user_norm_path):
            self.user_norm.load_state_dict(torch.load(user_norm_path))
            print(f"loaded user_norm from {user_norm_path}")
        else:
            print("no checkpoint for user_norm")
        
        fusion_module_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'fusion_module_checkpoint.pt')
        if os.path.exists(fusion_module_path):
            self.fusion_module.load_state_dict(torch.load(fusion_module_path))
        else:
            print("no checkpoint for fusion_module")

        for param in self.classifier.parameters():
            param.requires_grad = True
        if self.model_config['use_lora']:
            self.model,self.classifier,self.fusion_module =self._setup_lora(self.model,self.classifier,self.fusion_module)

        for encoder in self.binary_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        for encoder in self.user_binary_encoders.values():
            for param in encoder.parameters():
                param.requires_grad = True
        for param in self.cls_norm.parameters():
            param.requires_grad = True
        for param in self.feat_norm.parameters():
            param.requires_grad = True
        for param in self.user_norm.parameters():
            param.requires_grad = True
        for param in self.model.visual.parameters():
            param.requires_grad = False

        if self.model_config['gradient_checkpointing']:
            print("开启梯度检查点")
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        else:
            print("梯度检查点关闭")

        # 默认FP32
        self.model = self.model.to(torch.float16)
        self.classifier = self.classifier.to(torch.float16)
        self.user_binary_encoders = self.user_binary_encoders.to(torch.float16)
        self.binary_encoders = self.binary_encoders.to(torch.float16)
        self.fusion_module = self.fusion_module.to(torch.float16)
        self.cls_norm = self.cls_norm.to(torch.float16)
        self.feat_norm = self.feat_norm.to(torch.float16)
        self.user_norm = self.user_norm.to(torch.float16)

    def _setup_lora(self, model, classifier,fusion_module):
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        VLM_lora_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'VLM_lora')
        if os.path.exists(os.path.join(VLM_lora_path, 'adapter_config.json')):
            model = PeftModel.from_pretrained(model, VLM_lora_path)
            print(f"Loaded VLM LORA from {VLM_lora_path}")
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
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
            )
            model = PeftModel(model, peft_config)
            for name, param in model.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True  
            print('Add VLM lora adapter from init')

        for name, param in model.named_parameters():
            if "vision_encoder" in name:
                print(f"{name} requires_grad: {param.requires_grad}")

        # ----------------------------
        # Step 2: Setup LoRA for the classifier
        # ----------------------------
        print(f"Classifier: Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        classifier_lora_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], "classifier_lora")
        if os.path.exists(classifier_lora_ckpt):
            # 初始化 LoRA 权重（注意不使用 from_pretrained）
            peft_classifier_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_proj", "up_proj", "down_proj","out_proj","input_proj","common_proj","user_proj_v","user_proj_k","q_proj"],  # LoRA on these modules
            )
            classifier = PeftModel(classifier, peft_config=peft_classifier_config)

            # 加载 LoRA adapter 权重
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

        print(f"fusion_module: Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        fusion_module_lora_ckpt = os.path.join(self.model_config['lora_checkpoint_dir'], "fusion_module_lora")
        if os.path.exists(fusion_module_lora_ckpt):
            # 初始化 LoRA 权重（注意不使用 from_pretrained）
            peft_fusion_module_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_linear"],  # LoRA on these modules
            )
            fusion_module = PeftModel(fusion_module, peft_config=peft_fusion_module_config)

            # 加载 LoRA adapter 权重
            fusion_module.load_adapter(fusion_module_lora_ckpt, adapter_name="default", is_trainable=True)
            print(f"Loaded fusion_module LoRA adapter from {fusion_module_lora_ckpt}")

            for name, param in fusion_module.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    
        else:
            peft_fusion_module_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="FEATURE_EXTRACTION",
                target_modules=["gate_linear"],  # LoRA on these modules
            )
            fusion_module = PeftModel(fusion_module, peft_fusion_module_config)
            for name, param in fusion_module.named_parameters():  
                if 'lora_' in name:  
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
            print('Add LoRA adapter to fusion_module from init')


        if isinstance(model, PeftModel):
            print("model is a lora model")
        else:
            print("model is NOT a lora model ")
        if isinstance(classifier, PeftModel):
            print("classifier is a lora model")
        else:
            print("classifier is NOT a lora model ")
        if isinstance(fusion_module, PeftModel):
            print("fusion_module is a lora model")
        else:
            print("fusion_module is NOT a lora model ")

        return model,classifier,fusion_module

    def forward(self, batch_features,user_feat, **inputs):   
        # for key, value in inputs.items():
        #     # 打印键和对应值的形状
        #     print(f"Key: {key}, Shape: {value.shape}")
        # Key: input_ids, Shape: torch.Size([14, 512])
        # Key: attention_mask, Shape: torch.Size([14, 512])
        # Key: pixel_values, Shape: torch.Size([5600, 1176])
        # Key: image_grid_thw, Shape: torch.Size([14, 3])
        inputs["pixel_values"].requires_grad_()


        outputs = self.model(**inputs, output_hidden_states=True)


        visual_features = self.model.visual(hidden_states=inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])  # ViT输出
        # print(f"split 之前 visual_features.shape : {visual_features.shape}")
        # 经过 split 之前，visual_features.shape : torch.Size([400, 1536])
        # image_grid_thw.prod(-1) 用于计算张量在最后一个维度，计算每个图像的总块数（T*H*W）
        split_sizes = (inputs["image_grid_thw"].prod(-1) // self.model.visual.spatial_merge_size**2).tolist()
        # print(f"split_sizes:{split_sizes}") [100,100,..,100]
        visual_features = torch.split(visual_features, split_sizes)
        # 经过 split 之后是一个元组，代表一个 batch 内每一个图像的 token 的 embedding
        # split 之后: {[f.shape for f in visual_features]}")  [torch.Size([100, 1536])
        # Qwen2-vl 中 已知模型的图片最大分辨率为280*280，patchsize为14 merge size为2，那么一个图片要占token 数=(H/PM)*(W/PM)=100 个

        # 图像 token，将每100个图像token压缩成一个token
        visual_features_stacked = torch.stack(visual_features)  # 堆叠 → [batch_size, 100, 1536]
        # visual_features_stacked.shape:torch.Size([4, 100, 1536])
        pooled_features = visual_features_stacked.mean(dim=1)  # 在token维度(dim=1)求平均 → [batch_size, 1536]

        # 文本token
        hidden_states = outputs.hidden_states[-1]
        # print(f"hidden_states.shape:{hidden_states.shape}")
        # hidden_states.shape:torch.Size([13, 512, 1536]) 13个 note
        text_feature = hidden_states[:,-1]

        #早期融合和晚期融合
        features = self.fusion_module(pooled_features, text_feature)
        features = features.to(self.classifier.base_model.model.layers[0].mlp.up_proj.lora_A.default.weight)


        # # 构造 feature tensor： [B, F]
        feat_tensors = []
        for feat, encoder in self.binary_encoders.items():
            feat_tensor = batch_features[feat]
            # print(f"feat_tensor:{feat_tensor}")
            # print(f"feat_tensor.shape:{feat_tensor.shape}")
            encoded = torch.stack([encoder(i) for i in feat_tensor], dim=0)  # encoded 形状为 [B, 20]，B为正负样本总数，N为特征编码后的维度
            # feat_tensor的每一个元素维度统一为 20
            feat_tensors.append(encoded)

        # # 拼接所有特征：[B, 5*20] = [B, 100]
        feat_tensor = torch.cat(feat_tensors, dim=1)  # [B, 100]，B为正负样本总数，100为所有特征编码后的维度

        # # print(f"feat_tensor.shape:{feat_tensor.shape}")
        # # 当 dim=1 时，堆叠操作会在第 1 维（列方向）合并这些张量，结果形状为 [2N, 5]

        User_feats = []
        for feat, encoder in self.user_binary_encoders.items():
            user_feat_tensor = user_feat[feat]
            # feat_tensor是一个张量列表，每一个张量代表当前数值的二进制转化后的张量
            # print(f"feat_tensor:{feat_tensor}")
            # print(f"feat_tensor.shape:{feat_tensor.shape}")
            encoded = torch.stack([encoder(i) for i in user_feat_tensor], dim=0)  # encoded 形状为 [B, 20]
            # feat_tensor的每一个元素维度统一为 20
            User_feats.append(encoded)

        User_feats = torch.cat(User_feats, dim=1) # [B, X]，B为正负样本总数，X为所有特征编码后的维度
        # print(f"User_feats.shape:{User_feats.shape}")

        features = self.cls_norm(features)         # [B, H]
        feat_tensor = self.feat_norm(feat_tensor)      # [B, F1]
        User_feats = self.user_norm(User_feats)        # [B, F2]
        # print(f"User_feats.shape:{User_feats.shape}")
        # print(f"User_feats:{User_feats}")
        # 判断第0维元素是否全部相同
        if torch.allclose(User_feats, User_feats[0].unsqueeze(0).expand_as(User_feats)):
            User_feats = User_feats[0]  # 压缩成 [F2]
        else:
            raise ValueError("User_feats 不完全相同")

        User_feats=User_feats.unsqueeze(dim=0)

        concat = torch.cat([features, feat_tensor], dim=1)  # [B, H+F]
        # print(f"concat.shape:{concat.shape}")
        # print(f"User_feats.shape:{User_feats.shape}")
        # concat = torch.cat([concat, User_feats], dim=1)  # [B, H+F]

        # print(f"concat.shape:{concat.shape}")
        # x指的是query与doc的嵌入向量
        x = concat.unsqueeze(dim=0) # 从 [N, H+F] -> [1, N, H+F]
        # print(f"x.shape:{x.shape}")
        # x.shape:torch.Size([1, 4, 3584])

        logits = self.classifier(User_feats, x).squeeze(dim=0)          # [N,1]


        return logits

    def save_pretrained(self, save_path):
        # VLM保持冻结，所以不保存
        self.model.save_pretrained(os.path.join(save_path, "VLM_lora"))

        # Merge LoRA adapter into base model
        if isinstance(self.classifier, PeftModel):
            self.classifier.save_pretrained(os.path.join(save_path, "classifier_lora"))
            print("Saved classifier LoRA adapter.")

        # 保存 binary_encoders（新增部分）
        binary_encoders_path = os.path.join(save_path, 'binary_encoders.pt')
        torch.save(self.binary_encoders.state_dict(), binary_encoders_path)
        user_binary_encoders_path = os.path.join(save_path, 'user_binary_encoders.pt')
        torch.save(self.user_binary_encoders.state_dict(), user_binary_encoders_path)
        # RMSNorm 参数
        torch.save(self.cls_norm.state_dict(), os.path.join(save_path, 'cls_norm.pt'))
        torch.save(self.feat_norm.state_dict(), os.path.join(save_path, 'feat_norm.pt'))
        torch.save(self.user_norm.state_dict(), os.path.join(save_path, 'user_norm.pt'))
        # # 融合模块
        if isinstance(self.fusion_module, PeftModel):
            self.fusion_module.save_pretrained(os.path.join(save_path, "fusion_module_lora"))
            print("Saved fusion_module LoRA adapter.")

