# %%
import pickle
import pandas as pd
from tqdm import trange
import wandb
import argparse
import os
import requests
import argparse
import shap
import requests

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from functools import reduce
from msd import define_activation, get_tokenizer
from datasets_info import get_commit_from_repo
from transformers import pipeline

from models import *

# %%
try:
    with open(r"C:\secrets\github_token.txt", "r") as f:
        github_token = f.read()
except FileNotFoundError:
    with open(r"/storage/nitzan/code/github_token.txt", "r") as f:
        github_token = f.read()

os.environ["WANDB_NOTEBOOK_NAME"] = "results.ipynb"
run = wandb.init(settings=wandb.Settings(start_method="thread"))


def get_repository_language(repo_owner, repo_name):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
    headers = {"Authorization": f"token {github_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        repository = response.json()
        language = repository["language"]
        return language
    else:
        return None


def get_commit_message(repo_owner, repo_name, commit_sha):
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/{commit_sha}"
    headers = {"Authorization": f"token {github_token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        commit_data = response.json()
        message = commit_data["commit"]["message"]
        return message
    else:
        return None


def wandb_to_df(wandb_json_table):
    df = pd.DataFrame(
        wandb_json_table.data, columns=["Name", "Hash", "Prediction", "Actual"]
    )
    df.index = df["Hash"]
    df = df.drop(columns=["Hash"])
    return df


def merge_rows(df, args):
    def my_avg(x):
        return int(0.5 + (sum(x) / len(x)))

    def my_any(x):
        return int(sum(x) > 0)

    def my_all(x):
        return int(sum(x) == len(x))

    aggr_dict = {"avg": my_avg, "any": my_any, "all": my_all}
    if args.aggr.lower() not in aggr_dict:
        raise ValueError("Aggregation function not found")

    df = df.groupby(["Hash", "Name"])
    if args.aggr.lower() == "any":
        df = df.any()
    elif args.aggr.lower() == "all":
        df = df.all()
    elif args.aggr.lower() == "avg":
        df = df.mean()
    else:
        raise ValueError("Aggregation function not found")
    df = df.reset_index()
    df.index = df["Hash"]
    return df


def get_best_accuracy(mall):
    best_accuracy = 0
    best_threshold = 0
    for i in range(100):
        final_accuracy = accuracy_score(mall["Actual"], mall["predicted_avg"] > i / 100)
        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_threshold = i / 100

    return best_threshold, best_accuracy


repo_cache_dict = {}


def get_language_from_name(name):
    if name in repo_cache_dict:
        return repo_cache_dict[name]

    owner, repo = name.split("_", 1)
    lang = get_repository_language(owner, repo)
    repo_cache_dict[name] = lang
    return lang


def get_all_predictions(args, run):
    code_df = wandb_to_df(
        run.use_artifact(args.code_artifact).get(f"test_table_{args.fold}.table.json")
    )
    message_df = wandb_to_df(
        run.use_artifact(args.message_artifact).get(
            f"test_table_{args.fold}.table.json"
        )
    )
    event_df = wandb_to_df(
        run.use_artifact(args.event_artifact).get(f"test_table_{args.fold}.table.json")
    )

    code_df = merge_rows(code_df, args)
    message_df = merge_rows(message_df, args)
    event_df = merge_rows(event_df, args)

    mall = pd.merge(
        pd.merge(
            code_df.add_suffix("_code"),
            message_df.add_suffix("_message"),
            left_index=True,
            right_index=True,
        ),
        event_df.add_suffix("_time"),
        left_index=True,
        right_index=True,
    )

    mall["predicted_avg"] = mall[
        ["Prediction_code", "Prediction_message", "Prediction_time"]
    ].mean(axis=1)
    return mall


def save_language_compare(mall, run):
    mall["language"] = mall.apply(
        lambda row: get_language_from_name(row["Name"]), axis=1
    )
    langs = mall["language"].unique()
    langs = [lang for lang in langs if lang is not None]
    langs = sorted(langs)
    print(langs)
    return langs


def save_accuracy(args, mall, run):
    best_threshold, best_accuracy = get_best_accuracy(mall)

    print("Final accuracy: ", best_accuracy)
    print("Best threshold: ", best_threshold)
    run.summary["best_threshold"] = best_threshold
    run.summary["best_accuracy"] = best_accuracy

    if args.compare_languages:
        for language in mall["language"].unique():
            (
                run.summary[f"best_threshold_{language}"],
                run.summary[f"best_accuracy_{language}"],
            ) = get_best_accuracy(mall[mall["language"] == language])
            print(
                f"Final accuracy for {language}: ",
                run.summary[f"best_accuracy_{language}"],
            )
            print(
                f"Best threshold for {language}: ",
                run.summary[f"best_threshold_{language}"],
            )


def make_url(row):
    owner, repo = row.get("Name_code").split("_", 1)
    commit_hash = row.get("Hash_code")
    return f"https://github.com/{owner}/{repo}/commit/{commit_hash}"


# %%


import argparse

args = argparse.Namespace()
args.code_artifact = "nitzanfarhi/code4/run-ia1zinlc-test_table_0:v0"
args.message_artifact = "nitzanfarhi/message4/run-q8hktc1v-test_table_0:v0"
args.event_artifact = "nitzanfarhi/events4/run-agwcpmq1-test_table_0:v0"
args.message_model = "nitzanfarhi/message4/Message_model_0.bin:v110"
args.aggr = "avg"
args.local_dir = ""
args.fold = 0
args.compare_languages = True

# %%
# wandb.init()
args.message_model_type = "roberta"
args.message_model_name = "roberta-base"
args.multi_message_model_artifact = "nitzanfarhi/message4/Message_model_0.bin:v110"
args.message_tokenizer_name = "roberta-base"
args.message_model_name = "roberta-base"
args.model_cache_dir = "cache_data/models"
args.dropout = 0.1
args.freeze_submodel_layers = True
args.do_lower_case = False
args.xshape1 = 20
args.xshape2 = 401
args.event_l1 = 883
args.event_l2 = 100
args.event_l3 = 114
args.event_activation = "tanh"
args.event_activation = define_activation(args.event_activation)

args.events_model_type = "conv1d"
args.cut_layers = False
args.event_window_size_before = 15
args.event_window_size_after = 5
args.cache_dir = "cache_data"
args.recreate_cache = False
args.balance_factor = 0.5
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.events_model_artifact = "nitzanfarhi/MSD4/Events_model_0.bin:v5"
# message_model = get_message_model(args)
events_model = get_events_model(args)
# initialize_model_from_wandb(args, events_model,args.events_model_artifact)
# if args.multi_message_model_artifact:
# initialize_model_from_wandb(args, message_model, args.multi_message_model_artifact)


# %%
import json
from datasets_info import EventsDataset

with open(os.path.join(args.cache_dir, "orc", "orchestrator.json"), "r") as f:
    mall = json.load(f)
keys = list(mall.keys())[:1000]
events_dataset = EventsDataset(args, mall, keys[:2000])
from datasets import MyConcatDataset

dataset = MyConcatDataset(args, events_dataset=events_dataset)
dataset.set_hashes(keys[:2000])

# %%
from torch.utils.data import DataLoader, SubsetRandomSampler

args.train_batch_size = 512
train_dataloader = DataLoader(
    dataset, batch_size=32, num_workers=0, pin_memory=True, drop_last=True, shuffle=True
)


batch = next(iter(train_dataloader))
data, _ = batch
data = data[2]
# %%
X_train = torch.tensor([a[0][2] for a in dataset][0])
explainer = shap.DeepExplainer(events_model, data.to(args.device))

batch = next(iter(train_dataloader))
images, _ = batch
e = shap.DeepExplainer(model, images[2].to(args.device))


# %%
batch = next(iter(train_dataloader))
data2, _ = batch
data2 = data2[2]
shap_values = explainer.shap_values(data2.to(args.device))
# draw graph
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0])
# %%
