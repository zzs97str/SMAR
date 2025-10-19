import os
os.environ["NCCL_SOCKET_TIMEOUT"] = "1800000"

import sys
from accelerate import Accelerator
from accelerate.utils import set_seed
from evaluator import *
from dataset_factory import *
from utils import *
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
from model_factory import *
from torch.utils.cpp_extension import CUDA_HOME
sys.path.append("../extensions/accelerate")
import logging
import itertools

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

def dataset_class(class_name):
    cls = registry.get_class(class_name)
    if cls:
        return cls
    else:
        raise ValueError(f"Class {class_name} not found")

class BaseTrainer:
    """Base Trainer Class"""

    def __init__(self, config,config1,config2):
        self.config = config
        self.config1 = config1
        self.config2 = config2
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


    def load_optimizer(self):
        """Load optimizer"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']

        Multimodal_params = [
            {'params': [p for p in self.model.model.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.classifier.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.binary_encoders.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.user_binary_encoders.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
            {'params': [p for p in self.model.fusion_module.parameters() if p.requires_grad], 'lr': optimizer_config['kwargs']['lr']},
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

def get_top_p_notes(data, search_idx, p):
    """
    从给定的search_idx中返回得分前p的note_idx
    
    参数:
    data: 原始JSON数据（已解析为字典）
    search_idx: 要查询的search索引
    p: 0~1之间的比例，表示要返回前p的note
    
    返回:
    得分前p的note_idx列表，按得分从高到低排序
    """
    # 检查search_idx是否存在于数据中
    if search_idx not in data:
        return []
    
    # 获取该search_idx下的所有note及其得分
    note_scores = data[search_idx]
    
    # 将note按得分从高到低排序
    sorted_notes = sorted(note_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 计算需要返回的数量
    total = len(sorted_notes)
    if total == 0:
        return []
    
    # 计算要返回的数量（向上取整）
    count = max(1, int(round(total * p)))  # 确保至少返回1个
    count = min(count, total)  # 不超过总数量
    
    # 返回前count个note的索引
    return [note[0] for note in sorted_notes[:count]]


class MultiModalTrainer(BaseTrainer):
    """VLM cross-encoder model trainer"""

    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = MultiModalRankModel(self.config, config1=self.config1, config2=self.config2)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)
            # 单独统计并打印 classifier 的可训练参数
            # classifier_trainable_params = sum(p.numel() for p in self.model.classifier.parameters() if p.requires_grad)
            # print(f"Trainable parameters in classifier: {classifier_trainable_params}")
            # encoders_trainable_params = sum(p.numel() for p in self.model.binary_encoders.parameters() if p.requires_grad)
            # print(f"Trainable parameters in binary_encoders: {encoders_trainable_params}")
            # encoders_trainable_params = sum(p.numel() for p in self.model.user_binary_encoders.parameters() if p.requires_grad)
            # print(f"Trainable parameters in user_binary_encoders: {encoders_trainable_params}")
        
        # 标注策略:取各模态前 30% 的数据进行标注
        # self.top_p = 0.3
        # print(f"self.top_p:{self.top_p}") 

        # 标注策略: 根据实验类型选择不同区间的数据
        self.experiment_type = "middle_1"  # 可选值: "top", "middle_1", "middle_2", "middle_3", "bottom"
        self.top_p = 0.2  # 保持0.3不变，用于计算30%的比例

        print(f"实验类型: {self.experiment_type}, 比例: {self.top_p}")

    def setup_data(self):
        """Set up data loading and evaluator"""
        self.load_training_data()
        self.build_evaluator()
        with open("dataset/ProcessedDataset/MultiModal/ExP5/text_predictions.json") as f:
            self.text_label = json.load(f)

        with open("dataset/ProcessedDataset/MultiModal/ExP5/figure_predictions.json") as f:
            self.figure_label = json.load(f)
        

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


        # itertools.islice(..., start_step, None)：是对数据本身的切片，作用是跳过 self.train_data_loader 中前 start_step 个批次的数据，只保留从 start_step 开始到末尾的批次
        # enumerate(..., start=start_step)：是对迭代序号的起始值设置，不影响数据本身，仅让 step 变量从 start_step 开始计数。
        # start_step = 3990
        # for step, batch in enumerate(itertools.islice(self.train_data_loader, start_step, None), start=start_step):
        
        for step, batch in enumerate(self.train_data_loader):

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


            # if (step % 500) == 1:
            #     print(f"loss_1_factor: {self.model.module.loss_1_factor.item():.6f}")
            #     print(f"loss_2_factor: {self.model.module.loss_2_factor.item():.6f}")
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
        # if self.accelerator.is_local_main_process:
        #     print(f"平移后logits:{logits}")
        scale =1.0
        logits = scale * torch.tanh(logits / scale)
        #tanh函数超过（-2,2）区间函数值变化不大

        # if self.accelerator.is_local_main_process:
        #     print(f"放缩后logits:{logits}")
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

        # if self.accelerator.is_local_main_process:
        #     print(f"log_cumsums:{log_cumsums}")
            
        # 5. 聚合损失
        return loss_per_position.sum(dim=1).mean() / note_nums      # 批处理平均


    def _train_step_Multimodal(self, batch):
        """Train one step"""
        self.model.train()

        # Listwise训练
        inputs = {k: v.to(self.accelerator.device) for k, v in batch["inputs"].items()}
        search_idxs = batch["search_idxs"]
        assert len(search_idxs)==1, "search_idxs长度不为1,search_idxs:{search_idxs}"
        search_idx = search_idxs[0]
        # print(f"search_idx:{search_idx}")
        note_idxs = batch["note_idxs"]
        # 这里的note_idxs包含所有模态的 notes，但是有截断，而单一模态中没有截断，因此要筛选
        labels = torch.tensor(batch["labels"])
        # trainer labels:[0,5,3,1,2,4,6,7,8,9,10]
        # argsort：按升序排列后，排好的值在原来列表中的索引
        sorted_indices = torch.argsort(labels)
        # modal_indexs的顺序是经过 shufle后的顺序
        modal_indexs = batch["modal_indexs"]
        # 根据 模态 拆分 文本模态和图像模态的输入，0 为文本，1 为图像
        text_inputs={}
        figure_inputs={}
        assert len(modal_indexs) == inputs["input_ids"].shape[0], "modal_indexs 与 input_ids batch size 不一致"

        # 这里的text_inputs和figure_inputs显示了原始混排数据中当前 query 下有多少不同模态的 notes
        for index, modal in enumerate(modal_indexs):
            if modal==0:
                for k, v in batch["text_inputs"].items():
                    text_inputs[k] = v
            elif modal==1:
                for k, v in batch["fig_inputs"].items():
                    figure_inputs[k] = v
            else:
                print(f"type(modal):type(modal)")
                raise ValueError("模态不对")

        # 混排listwise
        batch_features={k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["batch_features"].items() }
        user_feat = {k: [singleV.to(self.accelerator.device) for singleV in v] for k,v in batch["user_feat"].items() }
        assert not torch.isnan(inputs["input_ids"]).any(), "Input contains NaN"
        assert not torch.isinf(inputs["input_ids"]).any(), "Input contains Inf"
        logits = self.model(batch_features=batch_features,user_feat=user_feat, **inputs)
        # print(f"multimodal logits shape:{logits.shape}")
        # multimodal logits shape:torch.Size([14, 1])
        logits = logits.squeeze(dim=-1)  # view(-1) 的作用：将张量展平为 1D
        assert len(modal_indexs) == len(logits), "modal_indexs 和 logits 长度不匹配！"
        # 按照 label 的顺序对混排的 logits 进行排序
        logits_sortedby_label = logits[sorted_indices]


        # 将数据喂给单一模态模型前向传播,logits的顺序按照打乱后相应模态的顺序
        # print(f"text_inputs['input_ids'].shape:{text_inputs['input_ids'].shape}")
        # text_inputs['input_ids'].shape:torch.Size([9, 321])

        # text_logits的顺序是 shuffle 后的顺序
        # 单一文本模态给的序，混排蒸馏文本模态
        try:
            text_note2logits = self.text_label[str(search_idx)]
            text_note2logits = {k: v for k, v in text_note2logits.items() if int(k) in note_idxs}
            if len(text_note2logits)>0:
                sorted_text_note_idx=[note_idx for note_idx, _ in sorted(
                    text_note2logits.items(),
                    key=lambda item: item[1],  # 按score排序
                    reverse=True               # 降序排列
                )]
                assert len(note_idxs)==len(logits),"note_idxs和logits不是一一对应的,有问题"
                note_to_logit = dict(zip(note_idxs, logits))
                print(f"len(sorted_text_note_idx):{len(sorted_text_note_idx)}")
                # note_idxs的类型是 int 类型
                # hunpai_text_logits是按照文本模态所给的序而产生的
                hunpai_text_logits = [note_to_logit[int(note_idx)] for note_idx in sorted_text_note_idx]
                # hunpai_text_logits = torch.tensor(hunpai_text_logits).unsqueeze(dim=0)
                hunpai_text_logits = torch.stack(hunpai_text_logits).unsqueeze(dim=0)
                loss_text_kd = self.compute_listmle_loss(hunpai_text_logits)
            else:
                print("当前query 下没有文本模态")
                loss_text_kd = torch.tensor(0.0, device = self.accelerator.device)
        except Exception as e:
            print(f"search_idx:{search_idx}天生全是图像模态")                
            loss_text_kd = torch.tensor(0.0, device = self.accelerator.device)

        if self.accelerator.is_local_main_process:
            print(f"loss_text_kd:{loss_text_kd}")

        # print(f"figure_inputs['input_ids'].shape:{figure_inputs['input_ids'].shape}")
        # figure_inputs['input_ids'].shape:torch.Size([5, 512])

        # 单一图像模态给的序，混排蒸馏图像模态
        # 如果当前query下没有图像模态，没有截断也没有
        try:
            figure_note2logits = self.figure_label[str(search_idx)]
            figure_note2logits = {k: v for k, v in figure_note2logits.items() if int(k) in note_idxs}
            if len(figure_note2logits)>0:
                sorted_figure_note_idx=[note_idx for note_idx, _ in sorted(
                    figure_note2logits.items(),
                    key=lambda item: item[1],  # 按score排序
                    reverse=True               # 降序排列
                )]
                print(f"len(sorted_figure_note_idx):{len(sorted_figure_note_idx)}")
                note_to_logit = dict(zip(note_idxs, logits))
                # hunpai_figure_logits是按照图像模态所给的序而产生的
                hunpai_figure_logits = [note_to_logit[int(note_idx)] for note_idx in sorted_figure_note_idx]
                # hunpai_figure_logits = torch.tensor(hunpai_figure_logits).unsqueeze(dim=0)
                hunpai_figure_logits = torch.stack(hunpai_figure_logits).unsqueeze(dim=0)
                loss_figure_kd = self.compute_listmle_loss(hunpai_figure_logits)
            else:
                print("当前query 下截断后没有图像模态")
                loss_figure_kd = torch.tensor(0.0, device = self.accelerator.device)
        except Exception as e:
            print(f"len(figure_inputs):{len(figure_inputs)}")
            print(f"search_idx:{search_idx}天生全是文本模态")
            loss_figure_kd = torch.tensor(0.0, device = self.accelerator.device)
                    
        if self.accelerator.is_local_main_process:
            print(f"loss_figure_kd:{loss_figure_kd}")


        # loss from multimidal
        # # 混排数据集->logits->只标注各单一模态的前 10%
        # # 单一图像模态给的序
        # if len(figure_inputs)>0:
        #     # 问题：figure_label没有search_idx时混排显示该search_idx下有图像模态
        #     # 混排的search_idx有5个点了，没点的随机模态，依赖于multimodal_train_modal_index
        #     # figure_label怎么区分模态的？ 依赖于multimodal_exp_1_modal_index
        #     # multimodal_exp_1_modal_index是全量训练集的，multimodal_train_modal_index只是混排的，不冲突
        #     n = len(sorted_figure_note_idx)
        #     fig_k = round(n * self.top_p)
        #     fig_k = max(1, fig_k)           # 确保至少取1个元素
        #     sorted_figure_note_idx = sorted_figure_note_idx[:fig_k]
        # else:
        #     print(f"len(figure_inputs):{len(figure_inputs)}")

        # # 单一文本模态给的序
        # if len(text_inputs)>0:
        #     m = len(sorted_text_note_idx)
        #     text_k = round(m * self.top_p)
        #     text_k = max(1, text_k)           # 确保至少取1个元素
        #     sorted_text_note_idx = sorted_text_note_idx[:text_k]
        # else:
        #     print(f"len(text_inputs):{len(text_inputs)}")

        # 处理图像模态数据
        if len(figure_inputs) > 0:
            n = len(sorted_figure_note_idx)
            # 根据实验类型计算不同区间
            if self.experiment_type == "top":
                # 取前20%
                start_idx = 0
                end_idx = round(n * self.top_p)
            elif self.experiment_type == "middle_1":

                start_idx = round(n * self.top_p)
                end_idx = round(2*n * self.top_p)
            elif self.experiment_type == "middle_2":

                start_idx = round(2*n * self.top_p)
                end_idx = round(3*n * self.top_p)
            elif self.experiment_type == "middle_3":

                start_idx = round(3*n * self.top_p)
                end_idx = round(4*n * self.top_p)
            else:  # bottom
                start_idx = round(n * (1 - self.top_p))
                end_idx = n
            
            # 确保至少取1个元素
            end_idx = max(start_idx + 1, end_idx)
            # 确保end不大于总长度
            end_idx = min(n, end_idx)
            sorted_figure_note_idx = sorted_figure_note_idx[start_idx:end_idx]
        else:
            print(f"len(figure_inputs):{len(figure_inputs)}")

        # 处理文本模态数据
        if len(text_inputs) > 0:
            m = len(sorted_text_note_idx)
            # 根据实验类型计算不同区间
            if self.experiment_type == "top":
                # 取前20%
                start_idx = 0
                end_idx = round(m * self.top_p)
            elif self.experiment_type == "middle_1":
                start_idx = round(m * self.top_p)
                end_idx = round(2*m * self.top_p)
            elif self.experiment_type == "middle_2":
                start_idx = round(2*m * self.top_p)
                end_idx = round(3*m * self.top_p)
            elif self.experiment_type == "middle_3":
                start_idx = round(3*m * self.top_p)
                end_idx = round(4*m * self.top_p)
            else:  # bottom
                start_idx = round(m * (1 - self.top_p))
                end_idx = m
            
            # 确保至少取1个元素
            end_idx = max(start_idx + 1, end_idx)
            # 确保end不大于总长度
            end_idx = min(m, end_idx)
            sorted_text_note_idx = sorted_text_note_idx[start_idx:end_idx]
        else:
            print(f"len(text_inputs):{len(text_inputs)}")

        # 混排标注损失
        if len(text_inputs)>0 and len(figure_inputs)>0:
            hunpai_note_idxs= sorted_figure_note_idx + sorted_text_note_idx
            print(f"混排标注量:{len(hunpai_note_idxs)}")
            # print(f"图像前 0.1:{sorted_figure_note_idx}")
            # print(f"文本前 0.1:{sorted_text_note_idx}")

            hunpai_logits = []
            # logits_sortedby_label这里就是混排的 label顺序,需要把note_idxs也要恢复成 label 顺序的 notes 顺序
            note_idxs = torch.tensor(note_idxs)
            note_idxs_sortedby_label= note_idxs[sorted_indices]
            # print(f"note_idxs_sortedby_label:{note_idxs_sortedby_label}")
            # print(f"logits_sortedby_label:{logits_sortedby_label}")
            for note_idx,logit in zip(note_idxs_sortedby_label, logits_sortedby_label):
                if str(note_idx.item()) in hunpai_note_idxs:
                    # print(f"note_idx:{note_idx}")
                    hunpai_logits.append(logit)

            if hunpai_logits:
                # hunpai_logits = torch.tensor(hunpai_logits).unsqueeze(dim=0)
                hunpai_logits = torch.stack(hunpai_logits).unsqueeze(dim=0)
                loss_multimodal =self.compute_listmle_loss(hunpai_logits)
            else:
                loss_multimodal = torch.tensor(0.0, device = self.accelerator.device)
        else:
            loss_multimodal = torch.tensor(0.0, device = self.accelerator.device)
        
        if self.accelerator.is_local_main_process:
            print(f"loss_multimodal:{loss_multimodal}")

        # total loss
        loss = loss_text_kd + loss_figure_kd + loss_multimodal
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
    'cross_encoder_trainer': CrossEncoderTrainer,
    'vlm_trainer': VLMCrossEncoderTrainer,
    'multimodal_trainer': MultiModalTrainer
}

if __name__ == "__main__":
    config_path = sys.argv[1]
    print(f"Starting Training on {config_path}")
    config = get_config(config_path)
    config1 = get_config("config/search_cross_encoder_config.yaml")
    config2 = get_config("config/search_vlm_config.yaml")
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
    trainer = trainer_class[config['trainer']](config,config1=config1,config2=config2)
    print(f"Starting Training")
    trainer.train()

