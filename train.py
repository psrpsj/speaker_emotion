import numpy as np
import os
import pandas as pd
import torch
import wandb

from argument import TrainingArguments, TrainModelArguments
from dataset import CustomDataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from trainer import CustomTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from utils import label_to_num


def compute_metrics(pred):
    label = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(label, preds)
    f1 = f1_score(label, preds, average="macro")
    return {"accuracy": acc, "f1_score": f1}


def train():
    data = pd.read_csv("./data/train.csv")
    data["Target"] = label_to_num(data["Target"])
    parser = HfArgumentParser((TrainingArguments, TrainModelArguments))
    (train_args, model_args) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Current model is {model_args.model_name}")
    print(f"Current device is {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    set_seed(train_args.seed)

    if model_args.k_fold:
        print("### START TRAINING with K-Fold ###")
        fold = 1
        k_fold = StratifiedKFold(n_splits=5, shuffle=False)
        for train_index, valid_index in k_fold.split(data, data["Target"]):
            print(f"--- START Fold {fold} ---")
            output_dir = os.path.join(
                train_args.output_dir,
                model_args.project_name + "_kfold",
                "fold" + str(fold),
            )
            model_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name
            )
            model_config.num_labels = 7
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name, config=model_config
            )
            model.to(device)
            model.train()

            wandb.init(
                entity="psrpsj",
                project="speaker",
                name=model_args.project_name + "_kfold_" + str(fold),
                tags=[model_args.model_name],
            )
            wandb.config.update(train_args)

            train_dataset, valid_dataset = (
                data.iloc[train_index],
                data.iloc[valid_index],
            )

            train = CustomDataset(train_dataset, tokenizer)
            valid = CustomDataset(valid_dataset, tokenizer)

            trainer = CustomTrainer(
                loss_name=model_args.loss_name,
                model=model,
                args=train_args,
                train_dataset=train,
                eval_dataset=valid,
                compute_metrics=compute_metrics,
            )
            trainer.train()
            model.save_pretrained(output_dir)
            wandb.finish()
            print(f"--- Fold {fold} finish! ---")
            fold += 1

    else:
        print("### START TRAINING with Non-KFold ###")
        print(f"Current model is {model_args.model_name}")
        print(f"Current device is {device}")

        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=model_args.model_name
        )
        model_config.num_labels = 7
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name, config=model_config
        )
        model.to(device)
        model.train()

        wandb.init(
            entity="psrpsj",
            project="speaker",
            name=model_args.project_name,
            tags=[model_args.model_name],
        )
        wandb.config.update(train_args)
        train_dataset, valid_dataset = train_test_split(
            data, test_size=0.2, stratify=data["Target"], random_state=42
        )

        train = CustomDataset(train_dataset, tokenizer)
        valid = CustomDataset(valid_dataset, tokenizer)

        trainer = CustomTrainer(
            loss_name=model_args.loss_name,
            model=model,
            args=train_args,
            train_dataset=train,
            eval_dataset=valid,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        model.save_pretrained(
            os.path.join(train_args.output_dir, model_args.project_name)
        )
        wandb.finish()
    print("### TRAINING FINISH ###")


if __name__ == "__main__":
    train()
