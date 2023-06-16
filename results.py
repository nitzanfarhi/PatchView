import wandb
import argparse
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from functools import reduce


def wandb_to_df(wandb_json_table):
    df = pd.DataFrame(wandb_json_table.data, columns=[
                      "Name", "Hash", "Prediction", "Actual"])
    df.index = df["Hash"]
    df = df.drop(columns=["Hash"])
    return df


def merge_rows(df, args):
    def my_avg(x): return int(0.5+(sum(x)/len(x)))
    def my_any(x): return int(sum(x) > 0)
    def my_all(x): return int(sum(x) == len(x))
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


def show_results(args):
    with wandb.init() as run:
        if args.local_dir != "":
            code_df = pd.read_csv(os.path.join(
                args.local_dir, "code_results.csv"), index_name="Hash")
            message_df = pd.read_csv(os.path.join(
                args.local_dir, "message_results.csv"), index_name="Hash")
            event_df = pd.read_csv(os.path.join(
                args.local_dir, "event_results.csv"), index_name="Hash")

        else:
            code_df = wandb_to_df(run.use_artifact(
                args.code_artifact).get(f"test_table_{args.fold}.table.json"))
            message_df = wandb_to_df(run.use_artifact(
                args.message_artifact).get(f"test_table_{args.fold}.table.json"))
            event_df = wandb_to_df(run.use_artifact(
                args.event_artifact).get(f"test_table_{args.fold}.table.json"))

            code_df = merge_rows(code_df, args)
            message_df = merge_rows(message_df, args)
            event_df = merge_rows(event_df, args)

            all = pd.merge(pd.merge(code_df, message_df, left_index=True,
                        right_index=True), event_df, left_index=True, right_index=True)

            all['predicted_avg'] = all[['Prediction_x',
                                        'Prediction_y', 'Prediction']].mean(axis=1)

            best_accuracy = 0
            best_threshold = 0
            for i in range(100):
                final_accuracy = accuracy_score(
                    all['Actual'], all['predicted_avg'] > i/100)
                if final_accuracy > best_accuracy:
                    best_accuracy = final_accuracy
                    best_threshold = i/100

            
            print("Best threshold: ", best_threshold)
            run.summary["best_threshold"] = best_threshold
            run.summary["best_accuracy"] = best_accuracy
            print("Final accuracy: ", best_accuracy)


def parse_args():
    parser = argparse.ArgumentParser(description="Show results from wandb")
    parser.add_argument("--code_artifact", type=str, default="nitzanfarhi/MSD2/run-x6yu69pm-test_table:v0",
                        help="Gather tables from wandb artifact")
    parser.add_argument("--message_artifact", type=str,
                        default="nitzanfarhi/MSD2/run-5n24o8xr-test_table:v0", help="Gather tables from wandb artifact")
    parser.add_argument("--event_artifact", type=str,
                        default="nitzanfarhi/MSD2/run-xqc0dddj-test_table:v0", help="Gather tables from wandb artifact")
    parser.add_argument("--aggr", type=str, default="avg",
                        help="Aggregation function to use")
    parser.add_argument("--local_dir", type=str, default="",
                        help="Gather tables from local directory")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    show_results(args)
