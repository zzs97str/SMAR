import sys
from accelerate import Accelerator
from accelerate.utils import set_seed
from evaluator import *
from dataset_factory import *
from utils import *
from tqdm import tqdm
import os
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
from model_factory import *
from torch.utils.cpp_extension import CUDA_HOME
sys.path.append("../extensions/accelerate")
import logging


# Configure the logging module here; any subsequent configurations will be ineffective
logging.basicConfig(filename="/root/paddlejob/workspace/env_run/output/multimodal/logger.log",
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
            print("固定随机种子")
        else:
            print(" ！不！ 固定随机种子")
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

    # def load_optimizer(self):
    #     """Load optimizer"""
    #     optimizer_config = self.config['optimizer']
    #     optimizer_name = optimizer_config['name']

    #     # 正确构造参数列表（保留参数名信息）
    #     param_groups = []
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             param_groups.append({'params': param, 'name': name})  # 保留参数名
    #             print(f"{name} requires_grad")

    #     # 初始化优化器（正确传递参数组）
    #     self.optimizer = optimizer_class[optimizer_name](param_groups, **optimizer_config['kwargs'])


    def load_optimizer(self):
        """Load optimizer"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']

        Multimodal_params = [
            {'params': [p for p in self.model.model.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.classifier.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.binary_encoders.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.user_binary_encoders.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            # {'params': [p for p in self.model.fusion_module.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            # {'params': [self.model.alpha, self.model.beta, self.model.gamma], 'lr': 1e-3} # 新增参数组
            ]

        # 创建优化器时传入参数分组
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


class DenseRetrievalTrainer(BaseTrainer):
    """Dense Retrieval Model Trainer"""

    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = DenseRetrievalModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model.model)

    def setup_data(self):
        self.load_training_data()
        self.build_evaluator()
        self.negatives_x_device = self.config['training']['negatives_x_device']

    def prepare_for_training(self):
        self.model.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        """Handle previous checkpoints"""
        if self.accelerator.is_main_process:
            if self.config['model']['load_from_new']:
                self._load_latest_checkpoint()
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
        """find the best checkpoint"""
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path

    def _load_latest_checkpoint(self):
        """Load the latest checkpoint"""
        latest_dir = find_latest_dir_with_subdir(self.config['base_project_dir'])
        if latest_dir:
            self._copy_from_dir(latest_dir)

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
                shutil.copytree(
                    f'{source_dir}/{cand_path}', 
                    f"{self.config['project_dir']}/{cand_path}"
                )
                self.config['model']['model_name_or_path'] = f"{self.config['project_dir']}/{cand_path}"

    def build_evaluator(self):
        """build evaluator"""
        self.evaluation_config = self.config['evaluation']
        if self.evaluation_config['evaluate_type'] == 'rerank':
            self.test_loader = self._create_test_loader()
            self.evaluator = DenseRetrievalRerankingEvaluator(
                self.accelerator,
                self.model,
                self.test_loader,
                **self.evaluation_config
            )
        else:
            self.note_loader = self._create_note_loader()
            self.query_loader = self._create_query_loader()
            self.evaluator = DenseRetrievalEvaluator(
                self.accelerator,
                self.model,
                self.note_loader,
                self.query_loader,
                **self.evaluation_config
            )
    
    def _create_test_loader(self):
        return DenseRetrievalRerankingTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def _create_note_loader(self):
        return NoteDataProcessor(
            self.local_rank,
            self.num_processes,
            **self.config['datasets']
        ).get_dataloader()

    def _create_query_loader(self):
        return QueryDataProcessor(
            self.local_rank,
            self.num_processes,
            **self.config['datasets']
        ).get_dataloader()

    def load_training_data(self):
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )

        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)

        pbar.close()

    def _train_step(self, batch):
        self.model.model.train()
        self.optimizer.zero_grad()

        query_emb, passage_emb = self._get_embeddings(batch)
        loss = self.contrastive_loss(query_emb, passage_emb)

        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def _get_embeddings(self, batch):
        """get the embeddings"""
        if self.config['model']['tie_model_weights']:
            return self._get_tied_embeddings(batch)
        else:
            raise NotImplementedError

    def _get_tied_embeddings(self, batch):
        batch_size = batch['queries_tokenized']['input_ids'].shape[0]
        merged_tokenized = batch['merged_tokenized']
        merged_emb = self.model.forward(**merged_tokenized)
        query_emb = merged_emb[:batch_size, :]
        passage_emb = merged_emb[batch_size:, :]
        return query_emb, passage_emb

    def _update_progress(self, pbar, epoch, step, loss):
        self.step += 1
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _handle_periodic_actions(self, loss, epoch, step):
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}

        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            self.evaluate()

        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)

        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def contrastive_loss(self, query_emb, passage_emb):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        if self.negatives_x_device:
            query_emb = self._dist_gather_tensor(query_emb)
            passage_emb = self._dist_gather_tensor(passage_emb)

        scores = torch.matmul(query_emb, passage_emb.transpose(0, 1))
        scores = scores.view(query_emb.size(0), -1)

        labels = torch.arange(
            scores.size(0), 
            device=scores.device, 
            dtype=torch.long
        )
        labels = labels * (passage_emb.size(0) // query_emb.size(0))

        return cross_entropy_loss(scores, labels)

    def evaluate(self):
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def save_checkpoint(self, suffix='', is_best=True):
        save_paths = self._get_save_paths(suffix)
        model = self.accelerator.unwrap_model(self.model.model)
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
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)


class DCNTrainer(BaseTrainer):
    """DCN Model Trainer for Search"""

    def __init__(self, config):
        super().__init__(config)
        self.grad_stats = {
            'max_grad': 0.0,
            'min_grad': float('inf'),
            'grad_norm_history': [],
            'gradient_vanishing_count': 0,
            'gradient_exploding_count': 0
        }
        # Set thresholds for gradient vanishing and exploding
        self.grad_vanish_threshold = 1e-4
        self.grad_explode_threshold = 10.0

    def load_optimizer(self):
        """Load optimizer"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']
        params = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad]
        non_embedding_params = {'params': [v for k, v in params if 'embedding' not in k]}
        embedding_params = {'params': [v for k, v in params if 'embedding' in k], 'lr':1e-1}
        self.optimizer = optimizer_class[optimizer_name]([non_embedding_params, embedding_params], **optimizer_config['kwargs'])

    def _check_gradients(self):
        """Check gradient status, including vanishing and exploding"""
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        has_valid_grad = False
        no_grad_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                no_grad_params.append(name)
                continue

            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

            # Get max and min values of non-zero gradients
            grad_abs = param.grad.data.abs()
            grad_nonzero = grad_abs[grad_abs > 0]
            if grad_nonzero.numel() > 0:
                has_valid_grad = True
                max_grad = max(max_grad, grad_nonzero.max().item())
                min_grad = min(min_grad, grad_nonzero.min().item())

            # Check gradient vanishing
            if param_norm < self.grad_vanish_threshold:
                self.grad_stats['gradient_vanishing_count'] += 1
                if self.accelerator.is_local_main_process:
                    print(f"\nWarning: Parameter {name} may have vanishing gradient (norm: {param_norm:.6f})")

            # Check gradient exploding
            if param_norm > self.grad_explode_threshold:
                self.grad_stats['gradient_exploding_count'] += 1
                if self.accelerator.is_local_main_process:
                    print(f"\nWarning: Parameter {name} may have exploding gradient (norm: {param_norm:.6f})")

            # Record detailed gradient information
            if self.accelerator.is_local_main_process and self.step % 100 == 0:
                print(f"Gradient statistics for parameter {name}:\n"
                      f"  - Norm: {param_norm:.6f}\n"
                      f"  - Max value: {grad_abs.max().item():.6f}\n"
                      f"  - Min non-zero value: {grad_nonzero.min().item() if grad_nonzero.numel() > 0 else 0:.6f}\n"
                      f"  - Zero gradient ratio: {(grad_abs == 0).float().mean().item():.2%}")

        # Print warning if parameters have no gradients
        if no_grad_params and self.accelerator.is_local_main_process:
            print(f"\nWarning: The following parameters have no gradients:\n{', '.join(no_grad_params)}")

        total_norm = total_norm ** 0.5
        self.grad_stats['max_grad'] = max(self.grad_stats['max_grad'], max_grad)
        if has_valid_grad:
            self.grad_stats['min_grad'] = min(self.grad_stats['min_grad'], min_grad)

        return total_norm

    def _log_gradient_stats(self):
        """Log gradient statistics"""
        if self.accelerator.is_local_main_process:
            stats = {
                'gradient/max_grad': self.grad_stats['max_grad'],
                'gradient/min_grad': self.grad_stats['min_grad'],
                'gradient/vanishing_count': self.grad_stats['gradient_vanishing_count'],
                'gradient/exploding_count': self.grad_stats['gradient_exploding_count']
            }

            if len(self.grad_stats['grad_norm_history']) > 0:
                stats['gradient/mean_norm'] = sum(self.grad_stats['grad_norm_history']) / len(self.grad_stats['grad_norm_history'])

            self.accelerator.log(stats, step=self.step)

    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = DCNModel(
                    self.config,
                    num_cross_layers=3,
                    hidden_size=256*2,
                    dropout_rate=0.1,
                    user_id_embedding_dim=32*2  
                )
        model_path = os.path.join(self.config['model']['model_name_or_path'], 'dcn_model.pt')
        print(f'loading model from {model_path}')
        self.model.load_model(model_path)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)

    def setup_data(self):
        self.load_training_data()
        self.build_evaluator()

    def prepare_for_training(self):
        self.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        if self.config['model']['load_from_new']:
            self._load_latest_checkpoint()
        self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            self._copy_checkpoint_files(best_file_path)

    def _find_best_checkpoint(self, file_paths):
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path

    def _load_latest_checkpoint(self):
        latest_dir = find_latest_dir_with_subdir(self.config['base_project_dir'])
        if latest_dir:
            self._copy_from_dir(latest_dir)

    def _copy_checkpoint_files(self, source_path):
        source_dir = os.path.dirname(source_path)
        self._copy_from_dir(source_dir)
        print(f'Best {self.target_metric} is {self.best_metric}')

    def _copy_from_dir(self, source_dir):
        for cand in ['base_lora_checkpoint_dir']:
            cand_path = self.config['model'][cand]
            print(f'copying {cand_path} from {source_dir} to {self.config["project_dir"]}')
            if os.path.exists(f'{source_dir}/{cand_path}'):
                if self.accelerator.is_main_process:
                    shutil.copytree(
                        f'{source_dir}/{cand_path}', 
                        f"{self.config['project_dir']}/{cand_path}"
                    )
                self.config['model']['model_name_or_path'] = f"{self.config['project_dir']}/{cand_path}"

    def build_evaluator(self):
        self.test_loader = self._create_test_loader()
        self.evaluation_config = self.config['evaluation']
        self.evaluator = DCNEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        return DCNTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def load_training_data(self):
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )

        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)

        pbar.close()

    def _train_step(self, batch):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        query_features, user_features, note_features, labels = batch
        device = self.accelerator.device        
        query_features = {k: v.to(device) for k, v in query_features.items()}
        user_features = {k: v.to(device) for k, v in user_features.items()}
        note_features = {k: v.to(device) for k, v in note_features.items()}
        labels = labels.to(device)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"Warning: Parameter {name} does not require gradients")
        criterion = torch.nn.BCEWithLogitsLoss()
        logits = self.model(query_features, user_features, note_features)
        loss = criterion(logits.squeeze(-1), labels)     

        self.accelerator.backward(loss)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()        
        return loss

    def _update_progress(self, pbar, epoch, step, loss):
        self.step += 1
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _handle_periodic_actions(self, loss, epoch, step):
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}

        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            self.evaluate()

        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)

        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def evaluate(self):
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def save_checkpoint(self, suffix='', is_best=True):
        save_paths = self._get_save_paths(suffix)
        model = self.accelerator.unwrap_model(self.model)
        torch.save(model.state_dict(), os.path.join(save_paths['lora'], 'dcn_model.pt'))

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
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)


class CrossEncoderTrainer(BaseTrainer):
    """Cross-encoder model trainer"""

    def setup_model(self):
        self._handle_previous_checkpoints()
        # 暂时不要用以前的checkpoints
        self.model = CrossEncoderModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model.model)
            # 单独统计并打印 classifier 的可训练参数
            classifier_trainable_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
            print(f"Trainable parameters in classifier: {classifier_trainable_params}")
            encoders_trainable_params = sum(p.numel() for p in self.model.binary_encoders.parameters() if p.requires_grad)
            print(f"Trainable parameters in binary_encoders: {encoders_trainable_params}")
            encoder_grad=True
            for encoder in self.model.binary_encoders.values():
                for param in encoder.parameters():
                    if not param.requires_grad:
                        print(f"{param} grad 为:{param.grad}")
                        encoder_grad=False
            if not encoder_grad:
                print("二进制编码器有被梯度冻结")
            else:
                print("二进制编码器均可训练")

    def setup_data(self):
        self.load_training_data()
        self.build_evaluator()

    def prepare_for_training(self):
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
            if self.config['model']['load_from_new']:
                self._load_latest_checkpoint()
            self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        """Load the best checkpoint"""
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            print(f'Best checkpoint is {best_file_path}')
            self._copy_checkpoint_files(best_file_path)
        else:
            print('No best checkpoint found')

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

    def _load_latest_checkpoint(self):
        """Load the latest checkpoint"""
        latest_dir = find_latest_dir_with_subdir(self.config['base_project_dir'])
        if latest_dir:
            self._copy_from_dir(latest_dir)

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
                shutil.copytree(
                    f'{source_dir}/{cand_path}', 
                    f"{self.config['project_dir']}/{cand_path}"
                )
                self.config['model']['model_name_or_path'] = f"{self.config['project_dir']}/{cand_path}"
                print(f'Copied {cand} from {source_dir}')

    def build_evaluator(self):
        """Build evaluator"""
        self.test_loader = self._create_test_loader()
        self.evaluation_config = self.config['evaluation']
        self.evaluator = CrossEncoderEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        """Create test data loader"""
        return CrossEncoderTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def load_training_data(self):
        """Load training data"""
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        """Train one epoch"""
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )

        for step, batch in enumerate(self.train_data_loader):
            # step为1时打印batch的输入和label

            # if step == 1 and self.accelerator.is_local_main_process:
            #     # 打印前三个数据的输入和标签（每个 batch 中的前三个）
            #     print(f"训练时inp_pos (first 3):")
            #     # 设置 NumPy 打印选项，避免截断大数组
            #     np.set_printoptions(threshold=np.inf)
            #     # 使用 pprint 格式化输出，并只打印前三个输入
            #     pprint(batch['inp_pos'][:3])
                
            #     print(f"训练时inp_neg (first 3):")
            #     np.set_printoptions(threshold=np.inf)
            #     # 使用 pprint 格式化输出，并只打印前三个标签
            #     pprint(batch['inp_neg'][:3])
            if step ==1:
            # 调试时可打印各参数组的实际学习率
                for group in self.optimizer.param_groups:
                    print(f"Params: {len(group['params'])}, LR: {group['lr']}")

            loss = self._train_step(batch)
            # torch.cuda.empty_cache()
            # loss = self._train_step_ranknet(batch)
            # print("正在使用ranknet")
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)

        pbar.close()

    def _train_step_pointwise(self, batch):
        """Train one step"""
        self.model.train()
        inputs = batch['inputs']
        labels = batch['labels']
        print(f"inputs.shape:{inputs.shape}")
        print(f"labels.shape:{labels.shape}")
        # Move data to the correct device
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        labels = labels.to(self.accelerator.device)

        # Use automatic mixed precision
        # Forward pass
        logits = self.model(**inputs)

        # Calculate loss
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.view(-1), labels.view(-1))

        # Backward pass
        self.accelerator.backward(loss)

        # lora，模型参数都没有梯度
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)
                
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return loss

    def _train_step_old(self, batch):
        """Train one pair-wise step with Hinge Loss"""
        self.model.train()

        # 1. 分别取出正、负对的输入，并搬到设备上
        inp_pos = {k: v.to(self.accelerator.device) for k, v in batch["inp_pos"].items()}
        inp_neg = {k: v.to(self.accelerator.device) for k, v in batch["inp_neg"].items()}
        if self.accelerator.is_local_main_process:
            print(f"inp_pos['input_ids'].shape:{inp_pos['input_ids'].shape}")
            print(f"inp_neg['input_ids'].shape:{inp_neg['input_ids'].shape}")

        # 2. 合并正负样本，防止正负样本前向传播模型两次，会影响梯度反传
        combined_input_ids = torch.cat([inp_pos["input_ids"], inp_neg["input_ids"]], dim=0)
        print("运行到这里——2")
        combined_attention_mask = torch.cat([inp_pos["attention_mask"], inp_neg["attention_mask"]], dim=0)
        print("运行到这里——3")
        combined_token_type_ids = torch.cat([inp_pos["token_type_ids"], inp_neg["token_type_ids"]], dim=0)        
        combined_inputs = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "token_type_ids": combined_token_type_ids
        }
        print("运行到这里——4")

        # 2. 前向计算：分别得到正例和负例的 logits
        # combined_inputs.shape:  [num_1+num_2,seq_len],num_1为正样本的个数 与 num_2 负样本的个数相等
        logits = self.model(**combined_inputs)
        # logits.shape:  [num_1+num_2,1]
        logits = logits.view(-1)  # 假设输出logits
        # logits.shape:  [num_1+num_2]
        
        batch_size = inp_pos["input_ids"].shape[0]
        # batch_size:  [num_1] 这里实际上不是 batch size       
        logits_pos = logits[:batch_size]
        # logits_pos.shape: [num_1]
        logits_neg = logits[batch_size:]
        # logits_neg.shape: [num_2]

        # 3. 计算 Hinge Loss
        # 我们希望 logits_pos - logits_neg >= margin
        margin = 0.5
        loss = torch.clamp(margin - (logits_pos - logits_neg), min=0).mean()

        # 4. 反向传播与优化
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def _train_step(self, batch):
        """Train one pair-wise step with Hinge Loss"""
        self.model.train()
        self.model=  self.model.to(self.accelerator.device)

        # 1. 分别取出正、负对的输入，并搬到设备上
        inp_pos = {k: v.to(self.accelerator.device) for k, v in batch["inp_pos"].items()}
        inp_neg = {k: v.to(self.accelerator.device) for k, v in batch["inp_neg"].items()}
        batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["features"].items() }
        user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items() }

        if self.accelerator.is_local_main_process:
            print(f"inp_pos['input_ids'].shape:{inp_pos['input_ids'].shape}")
            print(f"inp_neg['input_ids'].shape:{inp_neg['input_ids'].shape}")

        # 2. 合并正负样本，防止正负样本前向传播模型两次，会影响梯度反传
        combined_input_ids = torch.cat([inp_pos["input_ids"], inp_neg["input_ids"]], dim=0)
        combined_attention_mask = torch.cat([inp_pos["attention_mask"], inp_neg["attention_mask"]], dim=0)
        combined_token_type_ids = torch.cat([inp_pos["token_type_ids"], inp_neg["token_type_ids"]], dim=0)     
        combined_inputs = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask,
            "token_type_ids": combined_token_type_ids
        }
        # 2. 前向计算：分别得到正例和负例的 logits
        # combined_inputs.shape:  [num_1+num_2,seq_len],num_1为正样本的个数 与 num_2 负样本的个数相等
        logits = self.model(batch_features=batch_features,user_feat=user_feat, **combined_inputs)
        # logits.shape:  [num_1+num_2,1]
        logits = logits.view(-1)  # 假设输出logits
        # logits.shape:  [num_1+num_2]
        
        batch_size = inp_pos["input_ids"].shape[0]
        # batch_size:  [num_1] 这里实际上不是 batch size       
        logits_pos = logits[:batch_size]
        # logits_pos.shape: [num_1]
        logits_neg = logits[batch_size:]
        # logits_neg.shape: [num_2]

        # 3. 计算 Hinge Loss
        # 我们希望 logits_pos - logits_neg >= margin
        margin = 1
        loss = torch.clamp(margin - (logits_pos - logits_neg), min=0).mean()

        # 4. 反向传播与优化
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"[No grad] {name}")
        # 梯度裁剪
        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def merge_inputs(self, inputs_i, inputs_j):
        merged = {}
        for key in inputs_i.keys():
            merged[key] = torch.cat([inputs_i[key], inputs_j[key]], dim=0)
        return merged

    def _train_step_ranknet(self, batch):
        """Train one pair-wise step with Hinge Loss"""
        self.model.train()

        # 1. 获取输入与标签
        inp_i = {k: v.to(self.accelerator.device) for k, v in batch["input_i"].items()}
        inp_j = {k: v.to(self.accelerator.device) for k, v in batch["input_j"].items()}
        labels = batch["labels"].to(self.accelerator.device)

        merged_inputs = self.merge_inputs(inp_i, inp_j)
        logits = self.model(**merged_inputs)  # shape: [2*N]
        logits = logits.squeeze()

        batch_size = inp_i["input_ids"].shape[0]
        # batch_size:  [num_1] 这里实际上不是 batch size       
        logits_i = logits[:batch_size]
        # logits_pos.shape: [num_1]
        logits_j = logits[batch_size:]

        logits_diff = logits_i - logits_j  # shape: [N]

        targets = (labels + 1) / 2  # 映射为 0/1 标签
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits_diff, targets)

        # 4. 反向传播与优化
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def _update_progress(self, pbar, epoch, step, loss):
        """Update progress bar and step count"""
        self.step += 1
        print(f"step:{self.step},epoch:{epoch},loss:{loss:.5f}")
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _handle_periodic_actions(self, loss, epoch, step):
        """Handle periodic operations"""
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}

        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            print(f"step:{self.step},epoch:{epoch},start evaluate")
            self.evaluate()

        if self.step % self.config['training']['save_steps'] == 0 or (epoch % self.config['training']['save_epochs'] == 0 and step==0):
            if self.accelerator.is_main_process and not (self.step==0):
                print(f"step:{self.step},epoch:{epoch},start saving")
                self.save_checkpoint(suffix=f"epoch{epoch}_step{step}", is_best=False)
            # self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def evaluate(self):
        """Evaluate model"""
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        """Log evaluation metrics"""
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def save_checkpoint(self, suffix='', is_best=True):
        """Save checkpoint"""
        save_paths = self._get_save_paths(suffix)
        #  用于解除分布式训练框架（如 accelerate）对模型的包装，获取原始模型。
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(save_paths['lora'])

        if is_best:
            self._save_best_metric(save_paths['project'])

    def _get_save_paths(self, suffix):
        """Get save paths"""
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
        """Save best metric"""
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        # Retrieval results directory
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)


class VLMCrossEncoderTrainer(BaseTrainer):
    """VLM cross-encoder model trainer"""

    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = VLMCrossEncoderModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)
            # 单独统计并打印 classifier 的可训练参数
            classifier_trainable_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
            print(f"Trainable parameters in classifier: {classifier_trainable_params}")
            encoders_trainable_params = sum(p.numel() for p in self.model.binary_encoders.parameters() if p.requires_grad)
            print(f"Trainable parameters in binary_encoders: {encoders_trainable_params}")
            encoders_trainable_params = sum(p.numel() for p in self.model.user_binary_encoders.parameters() if p.requires_grad)
            print(f"Trainable parameters in user_binary_encoders: {encoders_trainable_params}")
            encoder_grad=True
            for encoder in self.model.binary_encoders.values():
                for param in encoder.parameters():
                    if not param.requires_grad:
                        print(f"{param} grad 为:{param.grad}")
                        encoder_grad=False
            if not encoder_grad:
                print("二进制编码器有被梯度冻结")
            else:
                print("二进制编码器均可训练")

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
        self.evaluator = VLMCrossEncoderEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        """Create test data loader"""
        accelerator = self.accelerator
        return VLMCrossEncoderTestDataProcessor(
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

        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)

        pbar.close()

    def _train_step(self, batch):
        """Train one step"""
        self.model.train()
        self.accelerator.unwrap_model(self.model).model.enable_input_require_grads()
        # inputs = batch['inputs']
        # labels = batch['labels']

        # 1. 分别取出正、负对的输入，并搬到设备上
        inp_pos = {k: v.to(self.accelerator.device) for k, v in batch["inp_pos"].items()}
        inp_neg = {k: v.to(self.accelerator.device) for k, v in batch["inp_neg"].items()}
        batch_features={k: [singleV.to(self.accelerator.device).to(torch.float16) for singleV in v] for k,v in batch["features"].items() }
        user_feat = {k: [singleV.to(self.accelerator.device).to(torch.float16) for singleV in v] for k,v in batch["user_feat"].items() }

        # 2. 合并正负样本，防止正负样本前向传播模型两次，会影响梯度反传
        combined_input_ids = torch.cat([inp_pos["input_ids"], inp_neg["input_ids"]], dim=0)
        combined_attention_mask = torch.cat([inp_pos["attention_mask"], inp_neg["attention_mask"]], dim=0)    
        combined_inputs = {
            "input_ids": combined_input_ids.to(torch.long),
            "attention_mask": combined_attention_mask.to(torch.long)
        }

        self.model=self.model.to(torch.float16)

        logits = self.model(batch_features=batch_features,user_feat=user_feat, **combined_inputs)
        logits = logits.view(-1)  # 假设输出logits
        # logits.shape:  [num_1+num_2]
        
        batch_size = inp_pos["input_ids"].shape[0]
        # batch_size:  [num_1] 这里实际上不是 batch size       
        logits_pos = logits[:batch_size]
        # logits_pos.shape: [num_1]
        logits_neg = logits[batch_size:]
        # logits_neg.shape: [num_2]

        # 3. 计算 Hinge Loss
        # 我们希望 logits_pos - logits_neg >= margin
        margin = 1
        loss = torch.clamp(margin - (logits_pos - logits_neg), min=0).mean()

        # loss_fn = torch.nn.BCEWithLogitsLoss()
        # loss = loss_fn(logits.view(-1), labels.view(-1))

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"[No grad] {name}")
        # 梯度裁剪
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
            self.evaluate()

        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)

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

class MultiModalTrainer(BaseTrainer):
    """VLM cross-encoder model trainer"""

    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = MultiModalRankModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)
            # 单独统计并打印 classifier 的可训练参数
            # classifier_trainable_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
            # print(f"Trainable parameters in classifier: {classifier_trainable_params}")
            # encoders_trainable_params = sum(p.numel() for p in self.model.binary_encoders.parameters() if p.requires_grad)
            # print(f"Trainable parameters in binary_encoders: {encoders_trainable_params}")
            # encoders_trainable_params = sum(p.numel() for p in self.model.user_binary_encoders.parameters() if p.requires_grad)
            # print(f"Trainable parameters in user_binary_encoders: {encoders_trainable_params}")
            # encoder_grad=True
            # for encoder in self.model.binary_encoders.values():
            #     for param in encoder.parameters():
            #         if not param.requires_grad:
            #             print(f"{param} grad 为:{param.grad}")
            #             encoder_grad=False
            # if not encoder_grad:
            #     print("二进制编码器有被梯度冻结")
            # else:
            #     print("二进制编码器均可训练")

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
        self.evaluator = MultiModalEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        """Create test data loader"""
        accelerator = self.accelerator
        return MultiModalTestDataProcessor(
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

        for step, batch in enumerate(self.train_data_loader):
            # loss = self._train_step(batch)

            if step == 1 and self.accelerator.is_local_main_process:
                freeze1=True
                for name, param in self.model.module.model.named_parameters():
                    if param.requires_grad:
                        freeze1=False
                if not freeze1:
                    print("未冻结 BERT")
                else:
                    print("已冻结Bert")

                freeze2=True
                for name, param in self.model.module.classifier.named_parameters():
                    if param.requires_grad:
                        freeze2=False
                if not freeze2:
                    print("未冻结 classifier")
                else:
                    print("已冻结 classifier")

                # freeze3=True
                # for name, param in self.model.module.binary_encoders.named_parameters():
                #     if param.requires_grad:
                #         freeze3=False
                # if not freeze3:
                #     print("未冻结 binary_encoders")
                # else:
                #     print("已冻结 binary_encoders")

                # freeze4=True
                # for name, param in self.model.module.user_binary_encoders.named_parameters():
                #     if param.requires_grad:
                #         freeze4=False
                # if not freeze4:
                #     print("未冻结 user_binary_encoders")
                # else:
                #     print("已冻结 user_binary_encoders")

                # 设置 NumPy 打印选项，避免截断大数组
                # np.set_printoptions(threshold=np.inf)
                # # 使用 pprint 格式化输出，并只打印前三个输入
                # print(f"训练时inp_pos (first 2):")
                # pprint(batch['inp_pos'][:2])
                # print(f"训练时inp_neg (first 2):")
                # pprint(batch['inp_neg'][:2])

            # if (step % 500) == 1:
            #     print(f"alpha: {self.model.module.alpha.item():.6f}")
            #     print(f"beta: {self.model.module.beta.item():.6f}")
            #     print(f"gamma: {self.model.module.gamma.item():.6f}")

            loss = self._train_step_Multimodal(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)

        pbar.close()

    def _train_step(self, batch):
        """Train one step"""
        self.model.train()
        # self.accelerator.unwrap_model(self.model).model.enable_input_require_grads()

        # Listwise训练

        # 1. 分别取出正、负对的输入，并搬到设备上
        inp_pos = {k: v.to(self.accelerator.device) for k, v in batch["inp_pos"].items()}
        inp_neg = {k: v.to(self.accelerator.device) for k, v in batch["inp_neg"].items()}
        batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["features"].items() }
        user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items() }

        # 2. 合并正负样本，防止正负样本前向传播模型两次，会影响梯度反传
        combined_input_ids = torch.cat([inp_pos["input_ids"], inp_neg["input_ids"]], dim=0)
        combined_attention_mask = torch.cat([inp_pos["attention_mask"], inp_neg["attention_mask"]], dim=0)    
        combined_inputs = {
            "input_ids": combined_input_ids,
            "attention_mask": combined_attention_mask
        }

        logits = self.model(batch_features=batch_features,user_feat=user_feat, **combined_inputs)
        logits = logits.view(-1)  # 假设输出logits
        # logits.shape:  [num_1+num_2]
        
        batch_size = inp_pos["input_ids"].shape[0]
        # batch_size:  [num_1] 这里实际上不是 batch size       
        logits_pos = logits[:batch_size]
        # logits_pos.shape: [num_1]
        logits_neg = logits[batch_size:]
        # logits_neg.shape: [num_2]

        # 3. 计算 Hinge Loss
        # 我们希望 logits_pos - logits_neg >= margin
        margin = 0.5
        loss = torch.clamp(margin - (logits_pos - logits_neg), min=0).mean()

        # loss_fn = torch.nn.BCEWithLogitsLoss()
        # loss = loss_fn(logits.view(-1), labels.view(-1))

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        self.optimizer.step()
        self.scheduler.step()
        # self.optimizer.zero_grad()

        return loss

    def compute_listmle_loss(self, logits): 
        """
        参数说明：
        logits : 模型输出的原始分数 [batch_size, n_items]
        
        返回：
        ListMLE损失值
        """

        #确保logits缩放到[-8, 8]
        logits = logits - logits.max(dim=1, keepdim=True).values  # 稳定性平移
        if self.accelerator.is_local_main_process:
            print(f"平移后logits:{logits}")
        scale =1.0
        logits = scale * torch.tanh(logits / scale)
        #tanh函数超过（-2,2）区间函数值变化不大

        if self.accelerator.is_local_main_process:
            print(f"放缩后logits:{logits}")
        assert not torch.isnan(logits).any(), "Logits contains NaN"
        assert not torch.isinf(logits).any(), "Logits contains Inf"

        # 确保输入为float类型
        logits = logits.float()
        
        note_nums=logits.shape[1]

        # 1. 计算指数值
        exp_logits = torch.exp(logits)  # [batch_size, n_items]
        
        # 2. 反向累积和计算（从右向左）
        reversed_exp = torch.flip(exp_logits, dims=[1])  # 反转第二个维度，倒数第一排到第一
        cumsums = torch.cumsum(reversed_exp, dim=1)      # 累积和 [batch_size, n_items]
        cumsums = torch.flip(cumsums, dims=[1])          # 反转回原始顺序
        
        # 3. 计算对数累积和
        log_cumsums = torch.log(cumsums + 1e-5)        # 防止log(0)
        
        # 4. 计算每个位置的损失项
        loss_per_position = log_cumsums - logits        # [batch_size, n_items]
        loss_per_position = loss_per_position.to(torch.float16)

        if self.accelerator.is_local_main_process:
            print(f"log_cumsums:{log_cumsums}")
            
        # 5. 聚合损失
        return loss_per_position.sum(dim=1).mean() / note_nums      # 批处理平均

    def _train_step_Multimodal(self, batch):
        """Train one step"""
        self.model.train()
        # self.accelerator.unwrap_model(self.model).model.enable_input_require_grads()

        # Listwise训练
        inputs = {k: v.to(self.accelerator.device) for k, v in batch["inputs"].items()}
        # labels = batch["labels"]
        # print(f"trainer labels:{labels}")
        labels = torch.tensor(batch["labels"])
        # labels = [torch.tensor(v).to(self.accelerator.device) for v in batch["labels"]]
        # print(f"trainer labels:{labels}")
        # trainer labels:[0,5,3,1,2,4,6,7,8,9,10]
        sorted_indices = torch.argsort(labels)
        # argsort：按升序排列后，排好的值在原来列表中的索引

        batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["batch_features"].items() }
        user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items() }
        
        assert not torch.isnan(inputs["input_ids"]).any(), "Input contains NaN"
        assert not torch.isinf(inputs["input_ids"]).any(), "Input contains Inf"

        logits = self.model(batch_features=batch_features,user_feat=user_feat, **inputs)
        # print(f"恢复顺序前logits:{logits}")
        # print(f"trainer logits shape:{logits.shape}")
        # trainer logits shape:torch.Size([1, 11, 1])
        logits = logits.squeeze(dim=-1)  # view(-1) 的作用：将张量展平为 1D
        # logits.shape:  [给定quer，查询结果个数]
        # print(f"trainer 展平后 logits.shape:{logits.shape}")
        logits = logits[sorted_indices].unsqueeze(dim=0)  # 保持批次维度
        # print(f"恢复后 logits 形状: {logits.shape}")
        # print(f"恢复顺序后logits:{logits}")

        loss =self.compute_listmle_loss(logits)
        # print(f"loss:{loss}")

        # batch_size = inputs.shape[0]
        # batch_size:  [num_1] 这里实际上不是 batch size       
        # logits_pos = logits[:batch_size]
        # # logits_pos.shape: [num_1]
        # logits_neg = logits[batch_size:]
        # # logits_neg.shape: [num_2]

        # 3. 计算 Hinge Loss
        # # 我们希望 logits_pos - logits_neg >= margin
        # margin = 0.5
        # loss = torch.clamp(margin - (logits_pos - logits_neg), min=0).mean()

        # loss_fn = torch.nn.BCEWithLogitsLoss()
        # loss = loss_fn(logits.view(-1), labels.view(-1))

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
    'cross_encoder_trainer': CrossEncoderTrainer,
    'dense_retrieval_trainer': DenseRetrievalTrainer,
    'dcn_trainer': DCNTrainer,
    'vlm_trainer': VLMCrossEncoderTrainer,
    'multimodal_trainer': MultiModalTrainer
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

