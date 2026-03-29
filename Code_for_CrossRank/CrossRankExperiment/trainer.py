import os
os.environ["NCCL_SOCKET_TIMEOUT"] = "1800000"
import sys
from accelerate import Accelerator
from accelerate.utils import set_seed
from evaluator import *
from dataset_factory import *
from utils import get_config, print_args,print_trainable_params_stats
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import torch_optimizer as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import datetime
import shutil
from registry import registry, register_class
import time
from glob import glob
from model_factory import CrossRankMultiModalRankModel
from torch.utils.cpp_extension import CUDA_HOME
# sys.path.append("../extensions/accelerate")
import logging
import json


# Configure the logging module here; any subsequent configurations will be ineffective
logging.basicConfig(filename="/path_to_CrossRank/output/multimodal/logger.log",
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                    level=logging.INFO,
                    filemode='w')

# create a logger object
logger = logging.getLogger(__name__)

optimizer_class = {"AdamW": FusedAdam, "Lamb": optim.Lamb, "DeepSpeedCPUAdam": DeepSpeedCPUAdam}
scheduler_class = {"CosineAnnealingLR": CosineAnnealingLR, "LinearLR": LinearLR}
os.environ['CUDA_HOME']="/usr/local/cuda"

from pprint import pprint
# Check if CUDA is available
if torch.cuda.is_available():
    # If CUDA is available, print its details
    print("Installation path of CUDA:", CUDA_HOME)
    print("CUDA version:", torch.version.cuda)
    print("CUDA device count:", torch.cuda.device_count())
else:
    # If CUDA is not available, provide a message
    print("CUDA is not available.")

def dataset_class(class_name):
    cls = registry.get_class(class_name)
    if cls:
        return cls
    else:
        raise ValueError(f"Class {class_name} not found")

class BaseTrainer:
    """Base Trainer Class"""

    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_tracking()
        self.setup_model()
        self.setup_data()
        self.setup_optimization()

    def setup_environment(self):
        """Set up training environment"""
        self.accelerator = Accelerator(
            log_with=self.config['logger']['log_with'],
            #device_placement=True,
            project_dir=self.config['project_dir']
        )
        if self.accelerator.is_main_process:
            print_args(self.config)
        self.accelerator.init_trackers(project_name=f'{self.config["project_name"]}')

        self.local_rank = self.accelerator.process_index  # Current device number
        self.fix_seed=True
        if self.fix_seed:
            set_seed(42) 
            print("Fixed random seed")
        else:
            print("Random seed NOT fixed")
        self.num_processes = self.accelerator.num_processes
        self.step = 0

    def setup_model(self):
        """Initialize model - to be implemented by subclass"""
        raise NotImplementedError

    def setup_data(self):
        """Set up data loading - to be implemented by subclass"""
        raise NotImplementedError

    def setup_optimization(self):
        """Set up optimizer and scheduler"""
        self.load_optimizer()
        self.load_scheduler()
        self.prepare_for_training()

    def setup_tracking(self):
        """Set up metric tracking"""
        self.target_metric = self.config['evaluation']['target_metric']
        self.best_metric = -1

    def load_optimizer(self):
        """Load optimizer"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']

        Multimodal_params = [
            {'params': [p for p in self.model.model.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.classifier.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.query_binary_encoders.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.statistic_binary_encoders.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            ]

        # Pass parameter groups when creating the optimizer
        self.optimizer = optimizer_class[optimizer_name](
            Multimodal_params, 
            **optimizer_config['kwargs']
        )


    def load_scheduler(self):
        """Load learning rate scheduler"""
        scheduler_config = self.config['scheduler']
        scheduler_name = scheduler_config['name']
        self.scheduler = scheduler_class[scheduler_name](
            self.optimizer, 
            **scheduler_config['kwargs']
        )

    def prepare_for_training(self):
        """Prepare for training - to be implemented by subclass"""
        raise NotImplementedError

    def train(self):
        """Training process"""
        # self.evaluate()
        for epoch in range(1, self.config['training']['num_epochs']):
            self.train_epoch(epoch)
            # self.evaluate()

    def train_epoch(self, epoch):
        """train one epoch"""
        raise NotImplementedError

    def evaluate(self):
        """evaluation - to be implemented by subclass"""
        raise NotImplementedError

    def save_checkpoint(self, suffix='', is_best=True):
        """save checkpoint - to be implemented by subclass"""
        raise NotImplementedError

    def _dist_gather_tensor(self, t):
        """gather tensors from all processes"""
        if t is None:
            return None
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.num_processes)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors


def get_top_p_notes(data, search_idx, p):
    """
    Return the note_idx with the top p scores from the given search_idx
    
    Parameters:
    data: Raw JSON data (parsed as a dictionary)
    search_idx: The search index to query
    p: Ratio between 0 and 1, indicating the top p notes to return
    
    Returns:
    List of note_idx with the top p scores, sorted in descending order of score
    """
    # Check if search_idx exists in the data
    if search_idx not in data:
        return []
    
    # Get all notes and their scores under this search_idx
    note_scores = data[search_idx]
    
    # Sort notes by score in descending order
    sorted_notes = sorted(note_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate the number to return
    total = len(sorted_notes)
    if total == 0:
        return []
    
    # Calculate the number to return (round up)
    count = max(1, int(round(total * p)))  # Ensure at least 1 is returned
    count = min(count, total)  # Do not exceed the total number
    
    # Return the indices of the first count notes
    return [note[0] for note in sorted_notes[:count]]


class CrossRankMultiModalTrainer(BaseTrainer):
    """VLM cross-encoder model trainer"""

    def setup_model(self):
        # Copy the best checkpoint to the current directory
        self._handle_previous_checkpoints()
        self.model = CrossRankMultiModalRankModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)
        
        # # Annotation strategy: randomly select 50% of data from each modality for annotation
        # self.top_p = 0.5
        # print(f"self.top_p:{self.top_p}")

    def setup_data(self):
        """Set up data loading and evaluator"""
        self.load_training_data()
        self.build_evaluator()


    def prepare_for_training(self):
        """Prepare training environment"""
        self.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        """Handle previous checkpoints"""
        if self.accelerator.is_main_process:
            self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        """Load the best checkpoint"""
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            self._copy_checkpoint_files(best_file_path)
    
    def _find_best_checkpoint(self, file_paths):
        """Find the best checkpoint"""
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path

    def _copy_checkpoint_files(self, source_path):
        """Copy checkpoint files"""
        source_dir = os.path.dirname(source_path)
        self._copy_from_dir(source_dir)
        print(f'Best {self.target_metric} is {self.best_metric}')

    def _copy_from_dir(self, source_dir):
        """Copy files from specified directory"""
        for cand in ['base_lora_checkpoint_dir']:
            cand_path = self.config['model'][cand]
            if os.path.exists(f'{source_dir}/{cand_path}'):
                if self.accelerator.is_main_process:
                    shutil.copytree(
                        f'{source_dir}/{cand_path}',
                        f"{self.config['project_dir']}/{cand_path}"
                    )
                self.config['model']['lora_checkpoint_dir'] = f"{self.config['project_dir']}/{cand_path}"

    def load_training_data(self):
        """Load training data"""
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def build_evaluator(self):
        """Build evaluator"""
        self.test_loader = self._create_test_loader()
        self.evaluation_config = self.config['evaluation']
        self.evaluator = CrossRankMultiModalEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        """Create test data loader"""
        accelerator = self.accelerator
        return CrossRankMultiModalTestProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )
        if self.step == 0:
            print(f"Start testing directly")
            self.accelerator.wait_for_everyone()
            self.evaluate()
            print(f"Start training")

        for step, batch in enumerate(self.train_data_loader):
            # loss = self._train_step(batch)

            if step == 1 and self.accelerator.is_local_main_process:
                freeze1=True
                for name, param in self.model.module.model.named_parameters():
                    if param.requires_grad:
                        freeze1=False
                if not freeze1:
                    print("Qwen NOT frozen")
                else:
                    print("Qwen frozen")

                freeze2=True
                for name, param in self.model.module.classifier.named_parameters():
                    if param.requires_grad:
                        freeze2=False
                if not freeze2:
                    print("classifier NOT frozen")
                else:
                    print("classifier frozen")

                freeze3=True
                for name, param in self.model.module.query_binary_encoders.named_parameters():
                    if param.requires_grad:
                        freeze3=False
                if not freeze3:
                    print("query_binary_encoders NOT frozen")
                else:
                    print("query_binary_encoders frozen")

                freeze4=True
                for name, param in self.model.module.statistic_binary_encoders.named_parameters():
                    if param.requires_grad:
                        freeze4=False
                if not freeze4:
                    print("statistic_binary_encoders NOT frozen")
                else:
                    print("statistic_binary_encoders frozen")

            loss = self._train_step_Multimodal(batch)

            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)

        pbar.close()


    def compute_listmle_loss(self, logits): 
        """
        Parameter description:
        logits : Raw scores output by the model [batch_size, n_items]
        
        Returns:
        ListMLE loss value
        """

        # Ensure logits are scaled to [-8, 8]
        logits = logits - logits.max(dim=1, keepdim=True).values  # Stability shift
        # if self.accelerator.is_local_main_process:
        #     print(f"Logits after shift:{logits}")
        scale =1.0
        logits = scale * torch.tanh(logits / scale)
        # The tanh function changes little beyond the (-2,2) interval

        # if self.accelerator.is_local_main_process:
        #     print(f"Logits after scaling:{logits}")
        assert not torch.isnan(logits).any(), "Logits contains NaN"
        assert not torch.isinf(logits).any(), "Logits contains Inf"

        # Ensure input is float type
        logits = logits.float()
        
        note_nums=logits.shape[1]

        # 1. Calculate exponential values
        exp_logits = torch.exp(logits)  # [batch_size, n_items]
        
        # 2. Reverse cumulative sum calculation (right to left)
        reversed_exp = torch.flip(exp_logits, dims=[1])  # Reverse the second dimension
        cumsums = torch.cumsum(reversed_exp, dim=1)      # Cumulative sum [batch_size, n_items]
        cumsums = torch.flip(cumsums, dims=[1])          # Reverse back to original order
        
        # 3. Calculate log cumulative sum
        log_cumsums = torch.log(cumsums + 1e-5)        # Prevent log(0)
        
        # 4. Calculate loss term for each position
        loss_per_position = log_cumsums - logits        # [batch_size, n_items]
        loss_per_position = loss_per_position.to(torch.float16)

        # if self.accelerator.is_local_main_process:
        #     print(f"log_cumsums:{log_cumsums}")
            
        # 5. Aggregate loss
        return loss_per_position.sum(dim=1).mean() / note_nums      # Batch average 

    def _train_step_Multimodal(self, batch):
        """Train one step"""
        self.model.train()

        # Listwise training
        inputs = {k: v.to(self.accelerator.device) for k, v in batch["inputs"].items()}
        
        search_idxs = batch["search_idxs"]
        # if self.accelerator.is_local_main_process:
        #     print(f"search_idxs:{search_idxs}")
 
        candidate_idxs = batch["candidate_idxs"]
        # print(f"candidate_idxs:{candidate_idxs}")

        labels = torch.tensor(batch["labels"])
        # print(f"labels:{labels}")

        # Group by query
        grouped_labels = defaultdict(list)
        for search_id, label in zip(search_idxs, labels):
            grouped_labels[search_id].append(label)
        label_tensor = [torch.stack(v) for v in grouped_labels.values()]
        query_lengths = [len(v) for v in grouped_labels.values()]
        # if self.accelerator.is_local_main_process:
        #     print(f"query_lengths:{query_lengths}")

        # The order of modal_indexs is after shuffling
        modal_indexs = batch["modal_indexs"]
        # print(f"modal_indexs:{modal_indexs}")
        # modal_indexs:[0, 1, 0, 0, 1, 0, 0, 0]

        # argsort: after sorting in ascending order, the indices of the sorted values in the original list
        sorted_indices_per_query = [torch.argsort(label_seq) for label_seq in label_tensor]

        # Split text and image modalities by modality, 0 for text, 1 for image
        text_inputs={}
        figure_inputs={}
        assert len(modal_indexs) == inputs["input_ids"].shape[0], "modal_indexs batch size inconsistent with input_ids"


        # Shuffled listwise
        batch_query_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["batch_query_features"].items() }
        # Any key corresponds to a list of Tensors

        batch_statistic_features = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["batch_statistic_features"].items() }

        assert not torch.isnan(inputs["input_ids"]).any(), "Input contains NaN"
        assert not torch.isinf(inputs["input_ids"]).any(), "Input contains Inf"

        batch_logits = self.model(search_idxs=search_idxs,
                            query_feat=batch_query_features,            
                            statistic_feat=batch_statistic_features,
                            **inputs)
        # if self.accelerator.is_local_main_process:
        #     print(f"batch_logits:{batch_logits}")

        assert len(batch_logits)==len(sorted_indices_per_query),"len(batch_logits) not equal to len(sorted_indices_per_query)"
        
        total_loss = 0.0
        num_queries = 0
        for i, logits in enumerate(batch_logits):
            logits = logits.squeeze(dim=-1)  # view(-1): flatten tensor to 1D
            # if not self.accelerator.is_local_main_process:
            #     print(f"logits:{logits}")
            # Sort shuffled logits according to label order
            single_sorted_indices_per_query=sorted_indices_per_query[i]
            # if not self.accelerator.is_local_main_process:
            #     print(f"single_sorted_indices_per_query:{single_sorted_indices_per_query}")
            logits_sortedby_label = logits[single_sorted_indices_per_query]
            # if not self.accelerator.is_local_main_process:
            #     print(f"logits_sortedby_label:{logits_sortedby_label}")
            logits_sortedby_label = logits_sortedby_label.unsqueeze(dim=0)
            # if not self.accelerator.is_local_main_process:
            #     print(f"logits_sortedby_label.shape:{logits_sortedby_label.shape}")
            # Calculate ListMLE loss for this query
            loss_i = self.compute_listmle_loss(logits_sortedby_label)
            total_loss += loss_i
            num_queries += 1

        # Final loss: average of all query losses
        loss_multimodal = total_loss / num_queries if num_queries > 0 else total_loss

        # # Feed data to single modality model for forward propagation, logits order follows the shuffled modality order
        # # print(f"text_inputs['input_ids'].shape:{text_inputs['input_ids'].shape}")
        # # text_inputs['input_ids'].shape:torch.Size([9, 321])

        # # text_logits order is shuffled order
        # # Single text modality order, shuffled distillation for text modality
        # try:
        #     unique_search_idx_list = list(dict.fromkeys(search_idxs))
        #     print(f"unique_search_idx_list:{unique_search_idx_list}")
        #     text_note2logits = self.text_label[str(search_idx)]
        #     # Single modality scores text modality in shuffled data
        #     text_note2logits = {k: v for k, v in text_note2logits.items() if int(k) in candidate_idxs}
        #     if len(text_note2logits)>0:
        #         sorted_text_note_idx=[note_idx for note_idx, _ in sorted(
        #             text_note2logits.items(),
        #             key=lambda item: item[1],  # Sort by score
        #             reverse=True               # Descending order
        #         )]
        #         assert len(note_idxs)==len(logits),"note_idxs and logits are not one-to-one, error"
        #         note_to_logit = dict(zip(note_idxs, logits))
        #         print(f"len(sorted_text_note_idx):{len(sorted_text_note_idx)}")
        #         # note_idxs type is int
        #         # hunpai_text_logits generated according to text modality order
        #         hunpai_text_logits = [note_to_logit[int(note_idx)] for note_idx in sorted_text_note_idx]
        #         # hunpai_text_logits = torch.tensor(hunpai_text_logits).unsqueeze(dim=0)
        #         hunpai_text_logits = torch.stack(hunpai_text_logits).unsqueeze(dim=0)
        #         loss_text_kd = self.compute_listmle_loss(hunpai_text_logits)
        #     else:
        #         print("No text modality under current query")
        #         loss_text_kd = torch.tensor(0.0, device=self.accelerator.device, requires_grad=False)
        # except Exception as e:
        #     print(f"search_idx:{search_idx} is all image modality natively")                
        #     loss_text_kd = torch.tensor(0.0, device=self.accelerator.device, requires_grad=False)

        # if self.accelerator.is_local_main_process:
        #     print(f"loss_text_kd:{loss_text_kd}")

        # # print(f"figure_inputs['input_ids'].shape:{figure_inputs['input_ids'].shape}")
        # # figure_inputs['input_ids'].shape:torch.Size([5, 512])

        # # Single image modality order, shuffled distillation for image modality
        # # If no image modality under current query, nothing
        # try:
        #     figure_note2logits = self.figure_label[str(search_idx)]
        #     figure_note2logits = {k: v for k, v in figure_note2logits.items() if int(k) in note_idxs}
        #     if len(figure_note2logits)>0:
        #         sorted_figure_note_idx=[note_idx for note_idx, _ in sorted(
        #             figure_note2logits.items(),
        #             key=lambda item: item[1],  # Sort by score
        #             reverse=True               # Descending order
        #         )]
        #         print(f"len(sorted_figure_note_idx):{len(sorted_figure_note_idx)}")
        #         note_to_logit = dict(zip(note_idxs, logits))
        #         # hunpai_figure_logits generated according to image modality order
        #         hunpai_figure_logits = [note_to_logit[int(note_idx)] for note_idx in sorted_figure_note_idx]
        #         # hunpai_figure_logits = torch.tensor(hunpai_figure_logits).unsqueeze(dim=0)
        #         hunpai_figure_logits = torch.stack(hunpai_figure_logits).unsqueeze(dim=0)
        #         loss_figure_kd = self.compute_listmle_loss(hunpai_figure_logits)
        #     else:
        #         print("No image modality after truncation under current query")
        #         loss_figure_kd = torch.tensor(0.0, device=self.accelerator.device, requires_grad=False)
        # except Exception as e:
        #     print(f"len(figure_inputs):{len(figure_inputs)}")
        #     print(f"search_idx:{search_idx} is all text modality natively")
        #     loss_figure_kd = torch.tensor(0.0, device=self.accelerator.device, requires_grad=False)
                    
        # if self.accelerator.is_local_main_process:
        #     print(f"loss_figure_kd:{loss_figure_kd}")


        # # loss from multimidal
        # # Shuffled dataset->logits->only annotate top 10% of each single modality
        # # Single image modality order
        # if len(figure_inputs) > 0:
        #     # Problem 1: For unclicked, modality is random, use the same random result
        #     # multimodal_exp_1_modal_index is for full training set, multimodal_train_modal_index only for shuffled training set, no conflict
        #     # figure_label and text_label are generated based on cleaned_search_train, no duplicate note_idx
        #     # Shuffled dataset avoids these data via skip_search
            
        #     n = len(sorted_figure_note_idx)
        #     fig_k = round(n * self.top_p)
        #     fig_k = max(1, fig_k)           # Ensure at least 1 element
        #     # Take top p% note_idx from single modality annotation
        #     # sorted_figure_note_idx = sorted_figure_note_idx[:fig_k]
        #     # Take random p% note_idx as annotation
        #     random_figure_note_idx = random.sample(sorted_figure_note_idx, min(fig_k, n))
        # else:
        #     print(f"len(figure_inputs):{len(figure_inputs)}")

        # # Single text modality order
        # if len(text_inputs) > 0:
        #     m = len(sorted_text_note_idx)
        #     text_k = round(m * self.top_p)
        #     text_k = max(1, text_k)           # Ensure at least 1 element
        #     # sorted_text_note_idx = sorted_text_note_idx[:text_k]
        #     # Take random p% note_idx as annotation
        #     random_text_note_idx = random.sample(sorted_text_note_idx, min(text_k, m))
        # else:
        #     print(f"len(text_inputs):{len(text_inputs)}")

        # # Shuffled annotation loss
        # if len(text_inputs)>0 and len(figure_inputs)>0:
        #     hunpai_note_idxs= random_figure_note_idx + random_text_note_idx
        #     print(f"Shuffled annotation count:{len(hunpai_note_idxs)}")
        #     # print(f"Top 0.1 of image:{sorted_figure_note_idx}")
        #     # print(f"Top 0.1 of text:{sorted_text_note_idx}")

        #     hunpai_logits = []
        #     # logits_sortedby_label is shuffled label order, need to restore note_idxs to label order
        #     note_idxs = torch.tensor(note_idxs)
        #     note_idxs_sortedby_label= note_idxs[sorted_indices]
        #     # print(f"note_idxs_sortedby_label:{note_idxs_sortedby_label}")
        #     # print(f"logits_sortedby_label:{logits_sortedby_label}")
        #     for note_idx,logit in zip(note_idxs_sortedby_label, logits_sortedby_label):
        #         if str(note_idx.item()) in hunpai_note_idxs:
        #             # print(f"note_idx:{note_idx}")
        #             hunpai_logits.append(logit)

        #     # hunpai_logits = torch.tensor(hunpai_logits).unsqueeze(dim=0)
        #     hunpai_logits = torch.stack(hunpai_logits).unsqueeze(dim=0)
        #     loss_multimodal =self.compute_listmle_loss(hunpai_logits)
        # else:
        #     loss_multimodal = torch.tensor(0.0, device=self.accelerator.device, requires_grad=False)
        
        # if self.accelerator.is_local_main_process:
        #     print(f"loss_multimodal:{loss_multimodal}")

        # total loss
        loss = loss_multimodal
        if self.accelerator.is_local_main_process:
            print(f"total loss:{loss}")

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"[No grad] {name}")
        
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        # self.optimizer.zero_grad()

        return loss

    def save_checkpoint(self, suffix='', is_best=True):
        """Save checkpoint"""
        save_paths = self._get_save_paths(suffix)

        model = self.accelerator.unwrap_model(self.model)

        model.save_pretrained(save_paths['lora'])

        if is_best:
            self._save_best_metric(save_paths['project'])

    def _get_save_paths(self, suffix):
        base_paths = {
            'lora': self.config['model']['lora_checkpoint_dir'],
            'project': self.config['project_dir']
        }

        save_paths = {}
        for key, base_path in base_paths.items():
            if key != 'project':
                save_paths[key] = os.path.join(base_path, suffix) if suffix else base_path
                os.makedirs(save_paths[key], exist_ok=True)
        save_paths['project'] = base_paths['project']

        return save_paths

    def _save_best_metric(self, project_path):
        """save the results of best metric"""
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)

    def _update_progress(self, pbar, epoch, step, loss):
        """Update progress bar and step count"""
        self.step += 1
        print(f"step:{self.step},epoch:{epoch},loss:{loss:.5f}")
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _log_training_info(self, epoch, step, loss):
        """Log training information"""
        if self.accelerator.is_local_main_process:
            info = {
                'epoch': epoch,
                'step': step,
                'loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            self.accelerator.log(info, step=self.step)

    def _handle_periodic_actions(self, loss, epoch, step):
        """Handle periodic operations"""
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}
        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            print(f"step:{self.step},epoch:{epoch},start evaluate")
            logger.info(f"step:{self.step},epoch:{epoch},start evaluate")
            self.accelerator.wait_for_everyone()
            self.evaluate()
            # pass

        if self.step % self.config['training']['save_steps'] == 0 or (epoch % self.config['training']['save_epochs'] == 0 and step==0):
            if self.accelerator.is_main_process and not (self.step==0):
                print(f"step:{self.step},epoch:{epoch},start saving")
                self.save_checkpoint(suffix=f"epoch{epoch}_step{step}", is_best=False)

        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def _log_metrics(self, metrics):
        """Log evaluation metrics"""
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def evaluate(self):
        """Evaluate model"""
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)


trainer_class = {
    'CrossRank_multimodal_trainer': CrossRankMultiModalTrainer
}

if __name__ == "__main__":
    config_path = sys.argv[1]
    print(f"Starting Training on {config_path}")
    config = get_config(config_path)
    time_stamp = sys.argv[2]
    if len(sys.argv) > 3:
        machine_rank = int(sys.argv[3])
        num_machines = int(sys.argv[4])
    else:
        machine_rank = 0
        num_machines = 1
    config['evaluation']['machine_rank'] = machine_rank
    config['evaluation']['num_machines'] = num_machines
    config['base_project_dir'] = config['project_dir']
    config['project_dir'] = os.path.join(config['project_dir'], f"{time_stamp}")
    project_dir = config['project_dir']
    config['evaluation']['output_dir'] = os.path.join(project_dir, config['evaluation']['output_dir'])
    if config['model']['load_from_new']:
        config['model']['lora_checkpoint_dir'] = os.path.join(config['model']['lora_checkpoint_dir'], 'new')
    config['model']['base_lora_checkpoint_dir'] = config['model']['lora_checkpoint_dir']
    config['model']['lora_checkpoint_dir'] = os.path.join(project_dir, config['model']['lora_checkpoint_dir'])
    config['optimizer']['kwargs']['lr'] = float(config['optimizer']['kwargs']['lr'])
    config['optimizer']['kwargs']['eps'] = float(config['optimizer']['kwargs']['eps'])
    trainer = trainer_class[config['trainer']](config)
    print(f"Starting Training")
    trainer.train()