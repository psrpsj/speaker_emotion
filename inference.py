import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from argument import TrainModelArguments
from dataset import CustomDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, HfArgumentParser
from utils import num_to_label


def inference():
    dataset = pd.read_csv("./data/test.csv")
    parser = HfArgumentParser(TrainModelArguments)
    (model_args,) = parser.parse_args_into_dataclasses()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)

    dataset["Target"] = [-1] * len(dataset)
    test_dataset = CustomDataset(dataset, tokenizer)
    dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_path = os.path.join("./output/", model_args.project_name)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    output_prob = []
    output_pred = []

    print("### START INFERENCE ###")
    for data in tqdm(dataloader):
        output = model(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            token_type_ids=data["token_type_ids"].to(device),
        )
        logit = output[0]
        prob = F.softmax(logit, dim=-1).detach().cpu().numpy()
        logit = logit.detach().cpu().numpy()
        result = np.argmax(logit, axis=-1)
        output_pred.append(result)
        output_prob.append(prob)

    pred_answer = np.concatenate(output_pred).tolist()
    output_prob = np.concatenate(output_prob, axis=0).tolist()
    dataset["Target"] = num_to_label(pred_answer)
    submission = pd.DataFrame({"ID": dataset["ID"], "Target": dataset["Target"]})
    submission.to_csv(model_path, index=False)
    print("### INFERENCE FINISH ###")
