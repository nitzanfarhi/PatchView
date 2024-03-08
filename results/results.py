import pickle
import pandas as pd
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
from transformers import pipeline

from models import *

with open(r"C:\secrets\github_token.txt", "r") as f:
    github_token = f.read()


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
        pd.merge(code_df, message_df, left_index=True, right_index=True),
        event_df,
        left_index=True,
        right_index=True,
    )

    mall["predicted_avg"] = mall[["Prediction_x", "Prediction_y", "Prediction"]].mean(
        axis=1
    )
    return mall


repo_cache_dict = {}


def get_language_from_name(name):
    if name in repo_cache_dict:
        return repo_cache_dict[name]

    owner, repo = name.split("_", 1)
    lang = get_repository_language(owner, repo)
    repo_cache_dict[name] = lang
    return lang


def save_language_compare(mall, run):
    mall["language"] = mall.apply(
        lambda row: get_language_from_name(row["Name"]), axis=1
    )
    langs = mall["language"].unique()
    langs = [lang for lang in langs if lang is not None]
    langs = sorted(langs)
    print(langs)


def example_explainability(mall, run, message_model_name):
    from msd import get_tokenizer
    from datasets import get_commit_from_repo

    args = argparse.Namespace()
    # wandb.init()
    args.message_model_type = "roberta"
    args.multi_message_model_artifact = message_model_name
    args.message_tokenizer_name = "roberta-base"
    args.message_model_name = "roberta-base"
    args.model_cache_dir = "cache_data/models"
    args.dropout = 0.1
    args.freeze_submodel_layers = True
    args.do_lower_case = False
    message_model = get_message_model(args)
    if args.multi_message_model_artifact:
        initialize_model_from_wandb(
            args, message_model, args.multi_message_model_artifact
        )

    tokenizer = get_tokenizer(
        args, args.message_model_type, args.message_tokenizer_name
    )
    model = pipeline(
        "text-classification",
        model=message_model.encoder,
        tokenizer=tokenizer,
        device=0,
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
    shap.plots.bar(
        shap_values[:, :, 0].mean(0),
        show=False,
        max_display=20,
        order=shap.Explanation.argsort.flip,
    )
    plt.title("Impactful words for label 0")
    run.log({"Label 0": wandb.Image(plt)})
    plt.show()

    shap.plots.bar(
        shap_values[:, :, 1].mean(0),
        show=False,
        max_display=20,
        order=shap.Explanation.argsort.flip,
    )
    plt.title("Impactful words for label 1")
    run.log({"Label 1": wandb.Image(plt)})
    plt.show()


def show_results(args):
    with wandb.init() as run:
        mall = get_all_predictions(args, run)
        if args.compare_languages:
            save_language_compare(mall, run)
        save_accuracy(args, mall, run)
        if args.message_model:
            # message_model = run.use_artifact(args.message_model).file()
            example_explainability(mall, run, args.message_model)


def parse_args():
    parser = argparse.ArgumentParser(description="Show results from wandb")
    parser.add_argument(
        "--code_artifact",
        type=str,
        default="nitzanfarhi/code4/run-ia1zinlc-test_table_0:v0",
        help="Gather tables from wandb artifact",
    )
    parser.add_argument(
        "--message_artifact",
        type=str,
        default="nitzanfarhi/message4/run-q8hktc1v-test_table_0:v0",
        help="Gather tables from wandb artifact",
    )
    parser.add_argument(
        "--event_artifact",
        type=str,
        default="nitzanfarhi/events4/run-agwcpmq1-test_table_0:v0",
        help="Gather tables from wandb artifact",
    )

    parser.add_argument(
        "--message_model",
        type=str,
        default="nitzanfarhi/message4/Message_model_0.bin:v110",
    )

    parser.add_argument(
        "--aggr", type=str, default="avg", help="Aggregation function to use"
    )
    parser.add_argument(
        "--local_dir", type=str, default="", help="Gather tables from local directory"
    )
    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--compare_languages", action="store_true", default=False)

    args = parser.parse_args()
    return args


def generate_easy_graphs():
    code_compare_data = [
        ["Concat Encoding", 0.8253],
        ["Concat with Comment Encoding", 0.8303],
        ["Special Token Encoding", 0.8234],
        ["Sequence Match Encoding", 0.816],
    ]
    plt.barh([x[0] for x in code_compare_data], [x[1] for x in code_compare_data])
    # set y axis to 0.8
    plt.xlim(0.8, 0.835)
    # convert to horizontal graph
    # set image size
    plt.gcf().subplots_adjust(left=0.30)
    plt.gcf().set_size_inches(10, 5)
    plt.title("Code Encoding Comparison")
    plt.xlabel("Accuracy")
    plt.show()

    print("Finished code encoding comparison")


if __name__ == "__main__":
    args = parse_args()
    generate_easy_graphs()
    # show_results(args)
