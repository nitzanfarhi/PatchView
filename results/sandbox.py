# %%
import wandb

run = wandb.init()
# %%

from matplotlib import pyplot as plt
from datasets import get_commit_from_repo

import argparse
import shap

from models import *
from msd import get_tokenizer
from transformers import pipeline

from results import get_all_predictions


args = argparse.Namespace()
args.code_artifact = "nitzanfarhi/code4/run-ia1zinlc-test_table_0:v0"
args.message_artifact = "nitzanfarhi/message4/run-q8hktc1v-test_table_0:v0"
args.event_artifact = "nitzanfarhi/events4/run-agwcpmq1-test_table_0:v0"
args.message_model = "nitzanfarhi/message4/Message_model_0.bin:v110"
args.fold = 0
args.aggr = "avg"
mall = get_all_predictions(args, run)


args = argparse.Namespace()
# wandb.init()
args.message_model_type = "roberta"
args.multi_message_model_artifact = "nitzanfarhi/message4/Message_model_0.bin:v110"
args.message_tokenizer_name = "roberta-base"
args.message_model_name = "roberta-base"
args.model_cache_dir = "cache_data/models"
args.dropout = 0.1
args.freeze_submodel_layers = True
args.do_lower_case = False
message_model = get_message_model(args)
if args.multi_message_model_artifact:
    initialize_model_from_wandb(args, message_model, args.multi_message_model_artifact)

tokenizer = get_tokenizer(args, args.message_model_type, args.message_tokenizer_name)
model = pipeline(
    "text-classification", model=message_model.encoder, tokenizer=tokenizer, device=0
)
explainer = shap.Explainer(model)

commit_messages = []
for i in range(len(mall)):
    repo_name = mall.iloc[i]["Name"]
    repo_hash = mall.iloc[i]["Hash"]
    commit = get_commit_from_repo(
        os.path.join(r"D:\multisource\commits", repo_name), repo_hash
    )
    if len(commit.msg) < tokenizer.model_max_length:
        commit_messages.append(commit.msg)

# Todo understand which label is which
shap_values = explainer(commit_messages)
# %%

fig = shap.plots.bar(
    shap_values[:, :, 0].mean(0),
    show=False,
    max_display=20,
    order=shap.Explanation.argsort.flip,
)
plt.title("Impactful words for label 0")
# plt.show()
run.log({"Label 0": wandb.Image(plt)})
plt.show()
# %%

shap.plots.bar(
    shap_values[:, :, 1].mean(0),
    show=False,
    max_display=20,
    order=shap.Explanation.argsort.flip,
)
plt.title("Impactful words for label 1")
run.log({"Label 1": wandb.Image(plt)})
plt.show()


# %%
import pickle

with open(r"cache_data\code\commits.json", "rb") as f:
    commit_info = pickle.load(f)
# %%

test_commit_info = []

shap.plots.bar(shap_values.abs.sum(0))

# %%
import importlib
import msd

importlib.reload(msd)
import argparse
import os
import json
import sys
import logging

logger = logging.getLogger(__name__)


sys.argv = ["run.py"]
args = msd.parse_args()
args.cache_dir = "cache_data"
args.model_type = "roberta"
args.n_gpu = 1
args.device = "cpu"
if args.cache_dir:
    args.model_cache_dir = os.path.join(args.cache_dir, "models")


# tokenizer, model = code_training.get_text_model_and_tokenizer(args)

with open(os.path.join(args.cache_dir, "orc", "orchestrator.json"), "r") as f:
    mall = json.load(f)


# %%
with open(
    r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\repo_commits.json", "r"
) as f:
    data = json.load(f)

# %%
import importlib
import msd
import events_datasets

importlib.reload(events_datasets)
importlib.reload(msd)


keys = sorted(list(mall.keys()))

# %%

code_args = argparse.Namespace(**vars(args))
code_args.recreate_cache = True
code_args.code_merge_file = True
code_args.model_type = "roberta_classification"
code_tokenizer = msd.get_tokenizer(code_args)
code_dataset = msd.TextDataset(code_tokenizer, code_args, mall, keys, "train")
code_model = msd.get_text_model(code_args)
args.hidden_size = code_model.encoder.config.hidden_size


# %%
message_args = argparse.Namespace(**vars(args))
message_args.recreate_cache = True
message_args.code_merge_file = True
message_args.model_type = "roberta_classification"
message_tokenizer = msd.get_tokenizer(message_args)
message_dataset = msd.TextDataset(message_tokenizer, message_args, mall, keys, "train")
message_model = msd.get_text_model(message_args)


# %%
events_args = argparse.Namespace(**vars(args))
events_args.model_type = "conv1d"
events_args.recreate_cache = True
events_dataset = events_datasets.EventsDataset(events_args, mall, keys, "train")
events_model = msd.get_events_model(events_args, events_dataset)


# %%
import torch
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    random_split,
    SubsetRandomSampler,
)
import tqdm
from transformers import get_linear_schedule_with_warmup


class MyConcatDataset(torch.utils.data.Dataset):
    def __init__(self, code_dataset, message_dataset, events_dataset):
        self.code_dataset = code_dataset
        self.message_dataset = message_dataset
        self.events_dataset = events_dataset

        self.merged_dataset = []
        self.merged_labels = []
        self.merged_info = []
        code_counter = 0
        message_counter = 0
        events_counter = 0

        in_counter = 0

        while (
            code_counter < len(code_dataset)
            and message_counter < len(message_dataset)
            and events_counter < len(events_dataset)
        ):
            code_commit = code_dataset.final_commit_info[code_counter]
            message_commit = message_dataset.final_commit_info[message_counter]
            events_commit = events_dataset.final_commit_info[events_counter]

            if code_commit["hash"] == message_commit["hash"] == events_commit["hash"]:
                self.merged_dataset.append(
                    (
                        code_dataset[code_counter][0],
                        message_dataset[message_counter][0],
                        events_dataset[events_counter][0],
                    )
                )
                self.merged_labels.append(code_dataset[code_counter][1])
                self.merged_info.append(code_dataset.final_commit_info[code_counter])
                in_counter += 1
                print(in_counter)
                code_counter += 1
                message_counter += 1
                events_counter += 1
            elif code_commit["hash"] < message_commit["hash"]:
                code_counter += 1
            elif message_commit["hash"] < events_commit["hash"]:
                message_counter += 1
            else:
                events_counter += 1

    def __getitem__(self, i):
        return self.merged_dataset[i], self.merged_labels[i]

    def get_info(self, i):
        return self.merged_info[i]

    def __len__(self):
        return len(self.merged_dataset)


concat_dataset = MyConcatDataset(code_dataset, message_dataset, events_dataset)


# %%
from torch import nn


class MultiModel(nn.Module):
    def __init__(self, code_model, message_model, events_model, args):
        super(MultiModel, self).__init__()
        self.code_model = code_model
        self.message_model = message_model
        self.events_model = events_model
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_size * 3, 2)

    def forward(self, data, labels=None):
        code, message, events = data
        code = self.code_model(code)
        message = self.message_model(message)
        events = self.events_model(events)
        x = torch.stack([code, message, events], dim=1)
        x = x.reshape(code.shape[0], -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log(
                (1 - prob)[:, 0] + 1e-10
            ) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


# %%
model = MultiModel(code_model, message_model, events_model, args)

tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
model.zero_grad()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(
    optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.max_steps * 0.1, num_training_steps=args.max_steps
)

args.start_epoch = 0
args.start_step = 0
args.source_model = "Multi"
global_step = args.start_step

train_dataloader = DataLoader(
    concat_dataset, batch_size=2, num_workers=0, pin_memory=True, drop_last=True
)
for idx in range(args.start_epoch, int(args.num_train_epochs)):
    bar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
    tr_num = 0
    train_loss = 0
    for step, batch in enumerate(bar):
        if args.source_model == "Multi":
            inputs = [x.to(args.device) for x in batch[0]]
        else:
            inputs = batch[0].to(args.device)

        labels = batch[1].to(args.device)
        model.train()
        loss, logits = model(inputs, labels)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        tr_loss += loss.item()
        tr_num += 1
        train_loss += loss.item()
        if avg_loss == 0:
            avg_loss = tr_loss
        avg_loss = round(train_loss / tr_num, 5)
        final_train_loss = avg_loss
        bar.set_description("epoch {} loss {}".format(idx, avg_loss))

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            output_flag = True
            avg_loss = round(
                np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
            )
            if (
                args.local_rank in [-1, 0]
                and args.logging_steps > 0
                and global_step % args.logging_steps == 0
            ):
                logging_loss = tr_loss
                tr_nb = global_step
