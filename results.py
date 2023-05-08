import wandb
import argparse
import pandas as pd
import os


def wandb_to_df(wandb_json_table):
    df = pd.DataFrame(wandb_json_table.data, columns = ["Name","Hash","Label"])
    df.index = df["Hash"]
    df = df.drop(columns=["Hash"])
    return df


def merge_rows(df, arg):
    my_avg = lambda x: int(0.5+(sum(x)/len(x)))
    my_any = lambda x: int(sum(x)>0)
    my_all = lambda x: int(sum(x)==len(x))
    aggr_dict = {"avg":my_avg, "any":my_any, "all":my_all}
    if arg.aggr.lower() not in aggr_dict:
        raise ValueError("Aggregation function not found")
    
    return df.groupby(["Hash","Name"])['Label'].apply(aggr_dict[arg.aggr.lower()])

def show_results(args):

    if args.local_dir!="":
        code_df = pd.read_csv(os.path.join(args.local_dir,"code_results.csv"), index_name="Hash")
        message_df = pd.read_csv(os.path.join(args.local_dir,"message_results.csv"), index_name="Hash")
        event_df = pd.read_csv(os.path.join(args.local_dir,"event_results.csv"), index_name="Hash")

    else:
        with wandb.init() as run:
            code_df = wandb_to_df(run.use_artifact(args.code_artifact).get("test_table.table.json"))
            message_df = wandb_to_df(run.use_artifact(args.message_artifact).get("test_table.table.json"))
            event_df = wandb_to_df(run.use_artifact(args.event_artifact).get("test_table.table.json"))

    code_df = merge_rows(code_df)
    all = pd.merge([code_df, message_df, event_df], on="Hash")
    print(all)

    


def parse_args():
    parser = argparse.ArgumentParser(description="Show results from wandb")
    parser.add_argument("--code_artifact", action="str", default="", help="Gather tables from wandb artifact")
    parser.add_argument("--message_artifact", action="str", default="", help="Gather tables from wandb artifact")
    parser.add_argument("--event_artifact", action="str", default="", help="Gather tables from wandb artifact")
    parser.add_argument("--aggr",type=str, default="avg", help="Aggregation function to use")
    parser.add_argument("--local_dir", action="str", default="", help="Gather tables from local directory")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    show_results(args)