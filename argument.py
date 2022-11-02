from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(default="./output/")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=16)
    overwrite_output_dir: bool = field(default=True)
    load_best_model_at_end: bool = field(default=True)
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    save_strategy: str = field(default="steps")
    metric_for_best_model: str = field(default="f1_score")
    save_total_limit: int = field(default=5)
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={
            "help": "Select evaluation strategy[linear, cosine, cosine_with_restarts, polynomial, constant, constant with warmup]"
        },
    )
    warmup_steps: int = field(default=500)


@dataclass
class TrainModelArguments:
    model_name: str = field(default="bert-base-cased")
    loss_name: str = field(default="focal")
    project_name: str = field(default="baseline")
    data_path: str = field(default="./data/")
    k_fold: bool = field(default=False)
