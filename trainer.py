import numpy as np
import torch
import torch.nn.functional as F

from loss import create_criterion
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from transformers import (
    Trainer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


class CustomTrainer(Trainer):
    """Custom Loss를 적용하기 위한 Trainer"""

    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False):

        if "labels" in inputs and self.loss_name != "default":
            custom_loss = create_criterion(self.loss_name)
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


class CustomBERTTrainer:
    def __init__(self, model, args, loss_name, train_data, eval_data, device):
        self.model = model
        self.args = args
        self.loss_name = loss_name
        self.train_data = train_data
        self.eval_data = eval_data
        self.device = device

    def compute_metrics(self, preds, label):
        acc = accuracy_score(label, preds)
        f1 = f1_score(label, preds, average="macro")
        return {"accuracy": acc, "f1_score": f1}

    def criterion(self, label, pred):
        custom_loss = create_criterion(self.loss_name)
        loss = custom_loss(pred, label)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs and self.loss_name != "default":
            custom_loss = create_criterion(self.loss_name)
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def get_scheduler(self, optimizer):
        if self.args.lr_scheduler_type == "linear":
            return get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=len(self.train_data) * self.args.num_train_epochs,
            )
        elif self.args.lr_scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=len(self.train_data) * self.args.num_train_epochs,
            )
        elif self.args.lr_scheduler_type == "cosine_with_restarts":
            return get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=len(self.train_data) * self.args.num_train_epochs,
            )

    def validation(self, eval_dataloader: DataLoader):
        self.model.eval()
        val_loss = []
        preds = []
        label = []
        with torch.no_grad():
            for data in tqdm(eval_dataloader):
                outputs = self.model(
                    input_ids=data["input_ids"].to(self.device),
                    attention_mask=data["attention_mask"].to(self.device),
                    token_type_ids=data["token_type_ids"].to(self.device),
                )
                logit = outputs[0]
                logit = logit.detach().cpu()
                loss = self.criterion(data["label"], logit)
                val_loss.append(loss)
                result = np.argmax(logit, axis=-1)
                preds += result.tolist()
                label += data["label"].tolist()
            metrics = self.compute_metrics(preds, label)
            print(
                f"*** Eval_loss: {np.mean(val_loss)}, Accuracy: {metrics['accuracy']}, F1 Score {metrics['f1_score']} ***"
            )
        return val_loss, metrics

    def train(self):
        best_score = 0
        best_model = None
        train_data = ConcatDataset([self.train_data] * self.args.num_train_epochs)
        train_dataload = DataLoader(
            train_data, batch_size=self.args.per_device_train_batch_size, shuffle=True
        )
        eval_dataload = DataLoader(
            self.eval_data,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=5e-5)
        scheduler = self.get_scheduler(optimizer)

        for step, data in tqdm(enumerate(train_dataload)):
            self.model.train()
            optimizer.zero_grad()
            train_loss = []

            outputs = self.model(
                input_ids=data["input_ids"].to(self.device),
                attention_mask=data["attention_mask"].to(self.device),
                token_type_ids=data["token_type_ids"].to(self.device),
            )
            loss = self.criterion(data["label"].to(self.device), outputs[0])
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % self.args.eval_steps == 0 and step != 0:
                val_loss, score = self.validation(eval_dataload)
                compare_score = score[self.args.metric_for_best_model]
                if best_score < compare_score:
                    best_score = compare_score
                    best_model = self.model

        return best_model
