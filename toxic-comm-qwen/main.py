import os
import sys
import logging
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from torchvision.ops import sigmoid_focal_loss

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset import ToxicDataset
from utils import compute_metrics_simple, compute_metrics_detailed
import swanlab

logger = logging.getLogger(__name__)

class CustomTrainer(Trainer):
    def __init__(self, loss_type="bce", focal_loss_gamma=2.0, focal_loss_alpha=0.25, use_swanlab=False, num_train_epochs=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.focal_loss_gamma = focal_loss_gamma
        self.focal_loss_alpha = focal_loss_alpha
        self.use_swanlab = use_swanlab
        self.num_train_epochs = num_train_epochs
        self.training_metrics = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.loss_type == "focal":
            loss = sigmoid_focal_loss(logits, labels, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="mean")
        else:
            loss = outputs.get("loss")
            if loss is None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if model.training:
            with torch.no_grad():
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits).cpu().numpy()
                y_pred = (probs >= 0.5).astype(int)
                y_true = labels.cpu().numpy()

                # Strict Accuracy: all labels must be correct for a sample to be correct
                strict_acc = np.all(y_pred == y_true, axis=1).mean()

                self.training_metrics = {
                    "acc": float(strict_acc),
                }

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self.training_metrics:
            logs.update(self.training_metrics)
        if self.use_swanlab:
            # Only log to SwanLab from the main process to avoid duplicate experiments/logs
            local_rank = getattr(self.args, "local_rank", -1)
            if local_rank in (-1, 0):
                swanlab.log(logs)
        super().log(logs, *args, **kwargs)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="/home/fwm/projects/Toxic-comment-classification/models/ar_models/qwen/Qwen3-0.6B-Base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    loss_type: str = field(
        default="bce",
        metadata={"help": "Loss function to use: 'bce' or 'focal'"}
    )
    focal_loss_gamma: float = field(
        default=2.0,
        metadata={"help": "Gamma parameter for Focal Loss"}
    )
    focal_loss_alpha: Optional[str] = field(
        default="0.25",
        metadata={"help": "Alpha parameter for Focal Loss. Can be a single float or a comma-separated list."}
    )
    classifier_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for the classification head"}
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the backbone model weights and only train the classification head"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )
    use_swanlab: bool = field(
        default=False,
        metadata={"help": "Whether to use SwanLab for experiment tracking"}
    )
    swanlab_project: str = field(
        default="Toxic-Comment-Classification",
        metadata={"help": "The name of the SwanLab project"}
    )
    swanlab_workspace: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the SwanLab workspace"}
    )
    swanlab_experiment_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the SwanLab experiment"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        default="/home/fwm/projects/Toxic-comment-classification/data/jigsaw-toxic-comment",
        metadata={"help": "The input data dir. Should contain the .csv files."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def parse_config_file():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, help="Path to the JSON config file.")
    args, remaining_args = parser.parse_known_args()
    return args.config_file, remaining_args

def main():
    # 解析 --config_file 参数
    config_file, remaining_args = parse_config_file()

    # 1. 解析参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if config_file:
        # 如果传入的是 JSON 配置文件
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(config_file))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=remaining_args)

    # focal_loss_alpha 应该是一个单一的浮点值
    if isinstance(model_args.focal_loss_alpha, str):
        try:
            model_args.focal_loss_alpha = float(model_args.focal_loss_alpha)
        except ValueError:
            raise ValueError(f"Invalid focal_loss_alpha value: {model_args.focal_loss_alpha}")

    # 2. 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # 设置 transformers 的日志级别
    import transformers
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 3. 设置随机种子
    set_seed(training_args.seed)

    # 4. 加载数据集
    logger.info(f"Loading dataset from {data_args.data_dir}")
    dataset_loader = ToxicDataset(data_args.data_dir)
    raw_datasets = dataset_loader.load_data()
    label_cols = dataset_loader.get_label_cols()
    num_labels = dataset_loader.get_num_labels()
    
    # 5. 加载 Tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        trust_remote_code=True
    )
    
    # Qwen 模型通常没有 pad_token，需要手动设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad_token to eos_token: {tokenizer.pad_token}")

    # 6. 数据预处理
    def preprocess_function(examples):
        # Tokenize the texts
        tokenized_inputs = tokenizer(
            examples["comment_text"],
            padding=False, # 后面使用 DataCollator 动态 padding
            truncation=True,
            max_length=data_args.max_seq_length,
        )
        
        # 处理 Labels: 将多列标签转换为一个 float list
        # examples 是一个 dict of lists
        batch_size = len(examples["comment_text"])
        labels_matrix = np.zeros((batch_size, num_labels), dtype=np.float32)
        
        for i, col in enumerate(label_cols):
            labels_matrix[:, i] = examples[col]
            
        tokenized_inputs["labels"] = labels_matrix.tolist()
        return tokenized_inputs

    with training_args.main_process_first(desc="dataset map pre-processing"):
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
            remove_columns=raw_datasets['train'].column_names # 移除原始列，只保留 input_ids, attention_mask, labels
        )

    # 7. 加载模型
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        problem_type="multi_label_classification", # 指定为多标签分类，会自动使用 BCEWithLogitsLoss
        classifier_dropout=model_args.classifier_dropout,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True
    )
    
    # 确保 pad_token_id 正确
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.use_lora:
        logger.info("Applying LoRA to the model...")
        target_modules = model_args.lora_target_modules.split(",") if model_args.lora_target_modules else None
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["score"] # 确保分类头被训练
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif model_args.freeze_backbone:
        logger.info("Freezing backbone model weights, only training the classification head.")
        # 冻结 base_model 的所有参数
        for param in model.base_model.parameters():
            param.requires_grad = False
            
        # 打印可训练参数信息以确认
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.2f}%)")

    # 自动配置保存最优模型策略
    if training_args.do_eval and not training_args.load_best_model_at_end:
        logger.info("Enabling load_best_model_at_end=True for best model strategy")
        training_args.load_best_model_at_end = True
        if training_args.metric_for_best_model is None:
            training_args.metric_for_best_model = "eval_f1_macro" # 使用 f1_macro 作为最优指标
            training_args.greater_is_better = True
        if training_args.save_total_limit is None:
            training_args.save_total_limit = 2

    # 8. 初始化 Trainer
    callbacks = []
    # Guard SwanLab initialization so only rank 0 (main process) initializes the experiment
    swanlab_inited = False
    if model_args.use_swanlab:
        # Determine whether this process is the main process
        is_main_process = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                is_main_process = (torch.distributed.get_rank() == 0)
            except Exception:
                is_main_process = False

        # 合并所有参数到 swanlab_config
        swanlab_config = {
            **vars(model_args),
            **vars(data_args),
            **vars(training_args),
        }

        if is_main_process:
            swanlab.init(
                project=model_args.swanlab_project,
                workspace=model_args.swanlab_workspace,
                experiment_name=model_args.swanlab_experiment_name,
                config=swanlab_config,
            )
            swanlab_inited = True
        else:
            logger.info("Skipping SwanLab init on non-main process (rank != 0)")

    # Determine eval dataset
    eval_dataset = processed_datasets["val"] if training_args.do_eval else None


    trainer = CustomTrainer(
        loss_type=model_args.loss_type,
        focal_loss_gamma=model_args.focal_loss_gamma,
        focal_loss_alpha=model_args.focal_loss_alpha,
        use_swanlab=model_args.use_swanlab,
        num_train_epochs=training_args.num_train_epochs,
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"] if training_args.do_train else None,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_simple,
        callbacks=callbacks,
    )

    # 9. 训练
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 10. 评估 (Validation)
    if training_args.do_eval:
        logger.info("*** Evaluate (Validation) ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. 评估 Test Set
    if "test" in processed_datasets:
        logger.info("*** Evaluate (Test) ***")
        # Switch to detailed metrics for test
        trainer.compute_metrics = compute_metrics_detailed
        test_metrics = trainer.evaluate(eval_dataset=processed_datasets["test"], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    if model_args.use_swanlab and swanlab_inited:
        swanlab.finish()

if __name__ == "__main__":
    main()