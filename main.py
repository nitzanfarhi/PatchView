""" Main script for training and evaluating models. """
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-not-lazy
# pylint: enable=logging-fstring-interpolation
import logging
import json
import argparse
import os
import random
import numpy as np
import torch
import wandb
import shap


from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
)

from data.orchestator import get_orchestrator
from models.models import get_model
from data.datasets_info import EventsDataset, MyConcatDataset, TextDataset
from data.datasets_info import get_patchdb_repos

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger(__name__)


os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
PROJECT_NAME = "MSD4"


MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "roberta_classification": (RobertaConfig, RobertaModel, RobertaTokenizer),
}


def parse_args():
    """handle user arguments."""
    parser = argparse.ArgumentParser()

    # Data Related arguments

    parser.add_argument("--use_cached_orchestrator",
                        action="store_true",
                        help="predefined orchestration of train/val/test"
    )
    
    parser.add_argument("--split_by_repos",
                        action="store_true",
                        help="orchestration of train/val/test is splitted by different repositories"
    )

    parser.add_argument(
        "--dataset",
        default="/storage/nitzan/dataset/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    parser.add_argument(
        "--cache_dir",
        default="cache_data",
        type=str,
        help=" directory to store cache data",
    )

    parser.add_argument(
        "--block_size",
        default=300,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs "
        "(take into account special tokens).",
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--recreate_cache",
        action="store_true",
        help="recreate the language model cache",
    )

    parser.add_argument(
        "--commit_repos_path", type=str, default=r"D:\multisource\commits"
    )

    parser.add_argument("--filter_repos", type=str, default="")
    parser.add_argument("--use_patchdb_commits", action='store_true', help="use only commits from patchdb")

    # Training parameters
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train"
    )
    parser.add_argument("--pooler_type", type=str, default="cls", help="poller type")
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument("--activation", default="tanh", type=str, help="activation")
    parser.add_argument("--folds", type=int, default=10, help="folds")
    parser.add_argument("--run_fold", type=int, default=-1, help="run_fold")
    parser.add_argument(
        "--balance_factor", type=float, default=1.0, help="balance_factor"
    )
    parser.add_argument(
        "--early_stop_threshold", type=int, default=20, help="early_stop_threshold"
    )

    # Source related arguments
    parser.add_argument(
        "--source_model", type=str, default="Multi", help="source model"
    )

    # Multi related arguments
    parser.add_argument(
        "--multi_model_type", type=str, default="multiv1", help="multi model type"
    )
    parser.add_argument(
        "--freeze_submodel_layers", action="store_true", help="freeze submodel layers"
    )
    parser.add_argument(
        "--cut_layers", action="store_true", help="cut layers for the multi model"
    )
    parser.add_argument(
        "--multi_model_hidden_size_1",
        type=int,
        default=768,
        help="multi model hidden size 1",
    )
    parser.add_argument(
        "--multi_model_hidden_size_2",
        type=int,
        default=64,
        help="multi model hidden size 2",
    )

    parser.add_argument(
        "--multi_code_model_artifact",
        type=str,
        default="",
        help="multi code model artifact",
    )
    parser.add_argument(
        "--multi_events_model_artifact",
        type=str,
        default="",
        help="multi events model artifact",
    )
    parser.add_argument(
        "--multi_message_model_artifact",
        type=str,
        default="",
        help="multi message model artifact",
    )

    # Events related arguments
    parser.add_argument(
        "--events_model_type", type=str, default="conv1d", help="events model type"
    )
    parser.add_argument(
        "--event_window_size_before", type=int, default=10, help="event window size"
    )
    parser.add_argument(
        "--event_window_size_after", type=int, default=10, help="event window size"
    )
    parser.add_argument("--event_l1", type=int, default=1024, help="event l1")
    parser.add_argument("--event_l2", type=int, default=256, help="event l1")
    parser.add_argument("--event_l3", type=int, default=64, help="event l1")
    parser.add_argument(
        "--event_bidirectional", type=int, default=0, help="bidirectional"
    )
    parser.add_argument(
        "--event_activation", type=str, default="tanh", help="activation"
    )

    # Code related arguments
    parser.add_argument(
        "--code_merge_file", action="store_true", help="code merge file"
    )
    parser.add_argument(
        "--code_model_type", type=str, default="roberta", help="code model type"
    )
    parser.add_argument("--code_embedding_type", "-cet", default="simple", type=str)
    parser.add_argument(
        "--code_model_name",
        default="microsoft/codebert-base",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--code_tokenizer_name",
        default="microsoft/codebert-base",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--code_activation", default="tanh", type=str, help="activation"
    )

    # Message related arguments
    parser.add_argument(
        "--message_model_type",
        type=str,
        default="roberta_classification",
        help="message model type",
    )
    parser.add_argument(
        "--message_embedding_type", "-met", default="commit_message", type=str
    )
    parser.add_argument(
        "--message_model_name",
        default="roberta-base",
        type=str,
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--message_tokenizer_name",
        default="roberta-base",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument("--message_l1", type=int, default=1024, help="message l1")
    parser.add_argument("--message_l2", type=int, default=256, help="message l1")
    parser.add_argument("--message_l3", type=int, default=64, help="message l1")
    parser.add_argument("--message_l4", type=int, default=2, help="message l1")
    parser.add_argument(
        "--message_activation", default="tanh", type=str, help="activation"
    )

    return parser.parse_args()


def set_seed(seed=42):
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, dataset):
    """Evaluate the model."""
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(
        dataset, batch_size=args.eval_batch_size, num_workers=0, pin_memory=True
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = []
        for x in batch[0]:
            if len(x) == 0:
                continue
            inputs.append(x.to(args.device))
        if len(inputs) == 1:
            inputs = inputs[0]

        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    best_acc = 0
    best_threshold = 0
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    for i in range(100):
        preds = logits[:, 0] > i / 100
        eval_acc = np.mean(labels == preds)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_threshold = i / 100
    logger.warning("best threshold: %s", best_threshold)
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(best_acc, 4),
        "labels": labels,
        "preds": preds,
    }
    return result


def train(args, train_dataset, model, fold, idx, run, eval_idx=None):
    """Train the model"""

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # print train_dataset label percentages
    negatives = 0
    positives = 0
    train_dataset.is_train = True
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    for a in train_dataloader:
        for b in a[1]:
            if b == 0:
                negatives += 1
            elif b == 1:
                positives += 1
            else:
                raise ValueError("label error")
    logger.warning("dataset balance percentage: %s", positives / (positives + negatives))
    run.summary[f"balance_{fold}"] = positives / (positives + negatives)

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epochs
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
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
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.max_steps * 0.1,
        num_training_steps=args.max_steps,
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    scheduler_last = os.path.join(checkpoint_last, "scheduler.pt")
    optimizer_last = os.path.join(checkpoint_last, "optimizer.pt")
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num GPUS = %d", args.n_gpu)
    logger.info("  Num examples = %d , evaluation = %d", len(idx), len(eval_idx))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    run.summary[f"train_examples_{fold}"] = len(idx)
    run.summary[f"eval_examples_{fold}"] = len(eval_idx)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0

    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        train_dataset.is_train = True
        model.train()
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(progress_bar):
            inputs = []
            for x in batch[0]:
                if len(x) == 0:
                    continue
                inputs.append(x.to(args.device))
            if len(inputs) == 1:
                inputs = inputs[0]

            labels = batch[1].to(args.device)
            loss, _ = model(inputs, labels)

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
            progress_bar.set_description(f"epoch {idx} loss {avg_loss}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    train_dataset.is_train = False
                    results = evaluate(args, model, train_dataset)
                    train_dataset.is_train = True

                    logger.warning(f"eval_loss {float(results['eval_loss'])}")
                    logger.warning(f"train_loss {final_train_loss}")
                    logger.warning(f"eval_acc {round(results['eval_acc'],4)}")

                    if results["eval_acc"] > best_acc:
                        best_epoch = idx
                        best_acc = results["eval_acc"]
                        logger.info("  " + "*" * 20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = "checkpoint-best-acc"
                        output_dir = os.path.join(
                            args.output_dir, "{}".format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )
                        output_dir = os.path.join(
                            output_dir,f"{args.source_model}_model_{fold}.bin",
                        )
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    wandb.log(
                        {
                            "epoch": idx,
                            "train_loss": final_train_loss,
                            "global_step": global_step,
                            "eval_loss": results["eval_loss"],
                            "eval_acc": results["eval_acc"],
                        }
                    )

        if idx - best_epoch > args.early_stop_threshold:
            logger.warning(f"Early stopped training at epoch {idx}")
            break  # terminate the training loop

    wandb.summary["best_epoch"] = best_epoch
    return best_acc


def test(args, model, dataset, idx, fold=0):
    """ Test the model."""
    dataset.is_train = False
    eval_dataloader = DataLoader(
        dataset, batch_size=args.eval_batch_size, num_workers=0, pin_memory=True
    )

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info(f"  Num examples = {len(idx)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = []
        for x in batch[0]:
            if len(x) == 0:
                continue
            inputs.append(x.to(args.device))
        if len(inputs) == 1:
            inputs = inputs[0]
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)

    best_acc = 0
    best_threshold = 0
    for i in range(100):
        preds = logits[:, 0] > i / 100
        eval_acc = np.mean(labels == preds)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_threshold = i / 100

    preds = logits[:, 0] > best_threshold

    res = []
    infos = []
    for i in range(len(dataset)):
        infos.append(dataset.get_info(i))

    for example, pred in zip(infos, preds):
        res.append([example["name"], example["hash"], pred, example["label"]])

    table = wandb.Table(columns=["Name", "Hash", "Prediction", "Actual"], data=res)
    wandb.log({f"test_table_{fold}": table})
    wandb.run.summary[f"test_acc_{fold}"] = best_acc
    wandb.run.summary[f"test_threshold_{fold}"] = best_threshold


def get_tokenizer(args, model_type, tokenizer_name):
    """Gets the relevant tokenizer"""
    _, _, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.model_cache_dir if args.model_cache_dir else None,
    )

    args.block_size = tokenizer.max_len_single_sentence
    return tokenizer


def define_activation(cur_activation):
    """Gets the relevant activation function"""
    if cur_activation == "tanh":
        return torch.nn.Tanh()
    elif cur_activation == "relu":
        return torch.nn.ReLU()
    elif cur_activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif cur_activation == "leakyrelu":
        return torch.nn.LeakyReLU()
    else:
        raise NotImplementedError
    

    
def main(args):
    """Main function"""
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    if args.n_gpu == 0:
        args.n_gpu = 1

    args.eval_batch_size = args.batch_size
    args.train_batch_size = args.batch_size
    filter_repos = ""

    args.per_gpu_train_batch_size = args.batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.batch_size // args.n_gpu
    # Setup logging
    logger.warning(f"Device: {device}, n_gpu: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0

    if args.cache_dir:
        args.model_cache_dir = os.path.join(args.cache_dir, "models")

    if args.filter_repos != "" and args.use_patchdb_commits:
        raise ValueError("Cannot use both filter_repos and use_patchdb_commits")

    if args.filter_repos != "":
        filter_repos = args.filter_repos.split(",")
    
    if args.use_patchdb_commits:
        filter_repos = get_patchdb_repos()

    logger.warning("Training/evaluation parameters %s", args)

    args.code_activation = define_activation(args.code_activation)
    args.message_activation = define_activation(args.message_activation)
    args.event_activation = define_activation(args.event_activation)

    if args.use_cached_orchestrator:
        with open(os.path.join(args.cache_dir, "orc", "orchestrator.json"), "r") as f:
            mall = json.load(f)
    else:
        mall = get_orchestrator(
            os.path.join(args.dataset, "commits"),
            os.path.join(args.dataset, "repo_commits.json"),
            cache_path=args.cache_dir,
            split_by_repos=args.split_by_repos)

    code_tokenizer, message_tokenizer = None, None

    if args.source_model == "Code":
        code_tokenizer = get_tokenizer(
            args, args.code_model_type, args.code_tokenizer_name
        )
        dataset = TextDataset(
            code_tokenizer, args, mall, mall.keys(), args.code_embedding_type, filter_repos
        )
        code_tokenizer = dataset.tokenizer
        args.return_class = True
        dataset = MyConcatDataset(args, code_dataset=dataset)

    elif args.source_model == "Message":
        message_tokenizer = get_tokenizer(
            args, args.message_model_type, args.message_tokenizer_name
        )
        dataset = TextDataset(
            message_tokenizer, args, mall, mall.keys(), args.message_embedding_type, filter_repos
        )
        args.return_class = True
        dataset = MyConcatDataset(args, message_dataset=dataset)

    elif args.source_model == "Events":
        dataset = EventsDataset(args, mall, mall.keys(), filter_repos, balance=True)
        args.xshape1 = dataset[0][0].shape[0]
        args.xshape2 = dataset[0][0].shape[1]
        args.return_class = True
        dataset = MyConcatDataset(args, events_dataset=dataset)

    elif args.source_model in ["Multi", "Multi_Without_Behavioural", "Multi_Without_Code","Multi_Without_Message"]:
        dataset, code_tokenizer, message_tokenizer = get_multi_dataset(args, mall)
        args.return_class = True
    else:
        raise NotImplementedError

    best_acc = 0
    splits = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    best_accs = []
    mall_keys_list = np.array(sorted(mall.keys()))
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(np.arange(len(mall_keys_list)))
    ):
        if args.run_fold != -1 and args.run_fold != fold:
            continue

        logger.warning(f"Running Fold {fold}")
        dataset.set_hashes(mall_keys_list[train_idx], is_train=True)
        dataset.set_hashes(mall_keys_list[val_idx], is_train=False)

        with wandb.init(
            project=PROJECT_NAME, tags=[args.source_model], config=args
        ) as run:
            model = get_model(
                args, message_tokenizer=message_tokenizer, code_tokenizer=code_tokenizer
            )
            run.define_metric("epoch")

            best_acc = train(
                args, dataset, model, fold, train_idx, run, eval_idx=val_idx
            )
            best_accs.append(best_acc)

            dataset.is_train = True
            train_dataloader = DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
                shuffle=True,
            )
            batch = next(iter(train_dataloader))
            if args.source_model == "Multi":
                feature_importance_analysis(dataset, model, batch[0])
            test(args, model, dataset, val_idx, fold=fold)
            run.summary["best_acc"] = max(best_accs)

            model_dir = os.path.join(args.output_dir, "checkpoint-best-acc")

            output_dir = os.path.join(
                model_dir, f"{args.source_model}_model_{fold}.bin"
            )
            artifact = wandb.Artifact(
                f"{args.source_model}_model_{fold}.bin", type="model"
            )
            artifact.add_file(output_dir)
            run.log_artifact(artifact)

    return {}

def feature_importance_analysis(dataset, model, images):
    my_model = model.to("cpu")
    e = shap.DeepExplainer(my_model, images[2].to("cpu"))
    shap_values = e.shap_values(images[2].to("cpu"))
    shap_val = np.array(shap_values)
    shap_val = np.reshape(
                shap_val, (-1, int(shap_val.shape[2]), int(shap_val.shape[3]))
            )
    shap_abs = np.absolute(shap_val)
    sum_0 = np.sum(shap_abs, axis=0)
    f_names = dataset.events_dataset.cur_repo_column_names
    x_pos = [i for i, _ in enumerate(f_names)]

    plt1 = plt.subplot(311)
    plt1.barh(x_pos, sum_0[1])
    plt1.set_yticks(x_pos)
    plt1.set_yticklabels(f_names)
    plt1.set_title("Yesterdays features (time-step 2)")
    plt2 = plt.subplot(312, sharex=plt1)
    plt2.barh(x_pos, sum_0[0])
    plt2.set_yticks(x_pos)
    plt2.set_yticklabels(f_names)
    plt2.set_title("The day before yesterdays features(time-step 1)")
    plt.tight_layout()
    plt.show()


def get_multi_dataset(args, mall):
    """Get the multi dataset."""
    keys = sorted(list(mall.keys()))

    if args.source_model == "Multi_Without_Code":
        code_tokenizer = None
        code_dataset = None
    else:
        code_tokenizer = get_tokenizer(args, args.code_model_type, args.code_tokenizer_name)
        code_dataset = TextDataset(
            code_tokenizer, args, mall, keys, args.code_embedding_type, args.filter_repos
        )

        logger.warning("Overall added lines: %s" ,code_dataset.added_lines_statistics)
        logger.warning("Overall deleted lines: %s" ,code_dataset.deleted_lines_statistics)

        code_tokenizer = code_dataset.tokenizer

    if args.source_model == "Multi_Without_Message":
        message_tokenizer = None
        message_dataset = None
    else:
        message_tokenizer = get_tokenizer(
            args, args.message_model_type, args.message_tokenizer_name
        )
        message_dataset = TextDataset(
            message_tokenizer, args, mall, keys, args.message_embedding_type, args.filter_repos
        )
        message_tokenizer = message_dataset.tokenizer

    if args.source_model == "Multi_Without_Behavioural":
        events_dataset = None
        args.xshape1 = 0
        args.xshape2 = 0
    else:
        events_dataset = EventsDataset(args, mall, keys, args.filter_repos)
        args.xshape1 = events_dataset[0][0].shape[0]
        args.xshape2 = events_dataset[0][0].shape[1]

    concat_dataset = MyConcatDataset(
        args,
        code_dataset=code_dataset,
        message_dataset=message_dataset,
        events_dataset=events_dataset,
    )
    return concat_dataset, code_tokenizer, message_tokenizer


if __name__ == "__main__":
    main(parse_args())
