from __future__ import absolute_import, division, print_function
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss
import copy
from torch.autograd import Variable

import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, random_split, SubsetRandomSampler
from transformers import (get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

import multiprocessing
from tqdm import tqdm
import pickle
import json
import difflib
import argparse
import logging
import os
import random
import numpy as np
import torch
import wandb
import logging
from torch import optim

from events_models import Conv1DTune
from language_models import RobertaClass
from sklearn.model_selection import KFold, train_test_split
import os
os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()

cpu_cont = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'roberta_classification': (RobertaConfig, RobertaModel, RobertaTokenizer)
}


# files bigger than that will be ignored as they are probably binaries / not code
MAXIMAL_FILE_SIZE = 100000
PROJECT_NAME = 'MSD2'

INS_TOKEN = '[INS]'
DEL_TOKEN = '[DEL]'
REP_BEFORE_TOKEN = '[RBT]'
REP_AFTER_TOKEN = '[RAT]'

ADD_TOKEN = '[ADD]'
DELETE_TOKEN = '[DEL]'


def get_commit_from_repo(cur_repo, hash):
    from pydriller import Repository
    return next(Repository(cur_repo, single=hash).traverse_commits())


def convert_to_ids_and_pad(source_tokens, tokenizer, args):
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    if len(source_ids) > args.block_size:
        return None
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)


def embed_file(file, tokenizer, args):
    before, after = "", ""
    if file["before_content"] is not None:
        before = file["before_content"]
    else:
        before = ""

    if file["content"] is not None:
        after = file["content"]

    if len(before) > MAXIMAL_FILE_SIZE or len(after) > MAXIMAL_FILE_SIZE or len(before) == 0 or len(after) == 0:
        return None

    operation_list = []
    opcodes = difflib.SequenceMatcher(a=before, b=after).get_opcodes()

    logger.warning(
        f"Size Before - {len(before)}, Size After - {len(after)}, Opcode Number -  {len(opcodes)}")
    for opp, a1, a2, b1, b2 in opcodes:
        if opp == 'equal':
            continue

        elif opp == 'replace':
            res = REP_BEFORE_TOKEN+" " + \
                before[a1:a2]+" "+REP_AFTER_TOKEN+" "+after[b1:b2]
            res = tokenizer(res, truncation=True,
                            padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        elif opp == 'insert':
            res = INS_TOKEN + " " + after[b1:b2]
            res = tokenizer(res, truncation=True,
                            padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        elif opp == 'delete':
            res = DEL_TOKEN + " " + before[a1:a2]
            res = tokenizer(res, truncation=True,
                            padding='max_length', max_length=args.block_size)
            operation_list.append(res)

        else:
            raise ValueError(f"Unknown operation: {opp}")

    return operation_list


def get_line_comment(language):
    from code_utils import ext_to_comment

    if language in ext_to_comment:
        return ext_to_comment[language]+" "
    else:
        return "// "


def handle_commit(commit, tokenizer, args, language='all', embedding_type='concat'):
    res = []
    for file in commit["files"]:

        if language != "all" and commit["filetype"] != language.lower():
            continue

        elif embedding_type == 'sum':
            embed_file_res = embed_file(file, tokenizer, args)
            if embed_file_res is not None:
                for embed in embed_file_res:
                    res.append(embed)

        elif embedding_type == 'simple':
            added = [diff[1] for diff in file['added']]
            deleted = [diff[1] for diff in file['deleted']]
            if not args.code_merge_file:
                file_res = " ".join(added+deleted)
                file_res = tokenizer(file_res, truncation=True,
                                    padding='max_length', max_length=args.block_size)
                res.append(file_res)
            else:
                res.append((added,deleted))

        elif embedding_type == 'simple_with_tokens':
            added = [ADD_TOKEN]+[diff[1]
                                 for diff in file['added']]+[tokenizer.sep_token]
            deleted = [DEL_TOKEN]+[diff[1] for diff in file['deleted']]
            file_res = " ".join(added+deleted)
            file_res = tokenizer(file_res, truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)

        elif embedding_type == 'simple_with_comments':
            added = [diff[1] for diff in file['added']]
            deleted = [get_line_comment(file["filetype"])+diff[1]
                       for diff in file['deleted']]

            file_res = " \n ".join(added+deleted)

            file_res = tokenizer(file_res, truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)

        elif embedding_type == 'commit_message':
            file_res = tokenizer(commit["message"], truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)
            break
        
    if args.code_merge_file and res != []:
        added_lst = []
        deleted_lst = []
        for added,deleted in res:
            added_lst += added
            deleted_lst += deleted
        file_res = " ".join(added_lst+deleted_lst)
        file_res = tokenizer(file_res, truncation=True, padding='max_length', max_length=args.block_size)
        return [file_res]

    return res


def safe_makedir(path):
    try:
        os.makedirs(path)
    except:
        pass


class TextDataset(Dataset):

    def __init__(self, tokenizer, args, all_json, keys, phase):
        logger.warning(f"Loading dataset {phase} {args.language}")
        self.tokenizer = tokenizer
        self.args = args
        self.phase = phase
        self.cache = not args.recreate_cache
        self.language = args.language
        self.counter = 0
        self.final_list_tensors = []
        self.final_list_labels = []
        self.final_commit_info = []
        self.commit_path = os.path.join(
            args.cache_dir, "code", f"{self.language}_{self.phase}.git")
        self.final_cache_list = os.path.join(
            args.cache_dir, "code", f"{args.embedding_type}_{self.language}_{self.phase}_final_list.pickle")
        self.positive_label_counter = 0
        self.negative_label_counter = 0
        self.commit_repos_path = args.commit_repos_path

        # self.load_commits_and_labels(args, phase)
        self.csv_list = keys

        if self.args.embedding_type == 'simple_with_tokens':
            logger.warning(
                f"Tokenizer size before adding tokens: {len(self.tokenizer)}")
            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': [ADD_TOKEN, DEL_TOKEN]})
        elif self.args.embedding_type == "sum":
            self.tokenizer.add_special_tokens({'additional_special_tokens': [
                                              REP_BEFORE_TOKEN, REP_AFTER_TOKEN, INS_TOKEN, DEL_TOKEN]})

        # commit_list = self.get_commits()
        # logger.warning(f"Number of commits: {len(commit_list)}")
        self.create_final_list(all_json, keys)
        logger.warning(f"Number of instances: {len(self.final_list_tensors)}")
        logger.warning(
            f"Number of positive instances: {sum(self.final_list_labels)}")

    def load_commits_and_labels(self, args, phase):
        if phase == 'train':
            with open(os.path.join(args.cache_dir, "orc", "orchestrator_train.json"), 'r') as f:
                self.csv_list = json.load(f)
        elif phase == 'val':
            with open(os.path.join(args.cache_dir, "orc", "orchestrator_val.json"), 'r') as f:
                self.csv_list = json.load(f)
        elif phase == 'test':
            with open(os.path.join(args.cache_dir, "orc", "orchestrator_test.json"), 'r') as f:
                self.csv_list = json.load(f)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_commits(self):
        result = []
        positives = 0
        negatives = 0
        if os.path.exists(self.commit_path):
            logger.warning("Get Commits from cache")
            with open(self.commit_path, 'rb') as f:
                return pickle.load(f)
        else:
            logger.warning("Get Commits from repos")
            for repo in (bar := tqdm(self.csv_list)):
                for cur_hash, label in self.csv_list[repo]:
                    try:
                        if cur_hash == "":
                            assert False, "shouldnt be empty hashes here"
                            continue

                        bar.set_description(
                            f"Repo - {repo} - Positives: {positives}, Negatives: {negatives}")
                        result.append(self.prepare_dict(
                            repo.replace("/", "_"), label, cur_hash))
                        if label == 1:
                            positives += 1
                        else:
                            negatives += 1
                    except Exception as e:
                        print(e)
                        continue

            with open(self.commit_path, 'wb') as f:
                pickle.dump(result, f)

            logger.warning(f"Positives: {positives}, Negatives: {negatives}")
            return result

    def add_code_data_to_dict(self, file):
        cur_dict = {}
        before = ""
        after = ""
        if file.content_before is not None:
            try:
                before = file.content_before.decode('utf-8')
            except UnicodeDecodeError:
                return None
        else:
            before = ""

        if file.content is not None:
            try:
                after = file.content.decode('utf-8')
            except UnicodeDecodeError:
                return None
        if "." not in file.filename:
            return None
        if len(after) > MAXIMAL_FILE_SIZE or len(before) > MAXIMAL_FILE_SIZE:
            return None

        filetype = file.filename.split(".")[-1].lower()
        cur_dict["filetype"] = filetype
        cur_dict["filename"] = file.filename
        cur_dict["content"] = after
        cur_dict["before_content"] = before
        cur_dict["added"] = file.diff_parsed["added"]
        cur_dict["deleted"] = file.diff_parsed["deleted"]
        return cur_dict

    def prepare_dict(self, repo, label, cur_hash):
        commit = get_commit_from_repo(
            os.path.join(self.commit_repos_path, repo), cur_hash)
        final_dict = {}
        final_dict["name"] = commit.project_name
        final_dict["hash"] = commit.hash
        final_dict["files"] = []
        final_dict["source"] = []
        final_dict["label"] = label
        final_dict["repo"] = repo
        final_dict["message"] = commit.msg

        for file in commit.modified_files:
            cur_dict = self.add_code_data_to_dict(file)
            if cur_dict is not None:
                final_dict["files"].append(cur_dict)

        return final_dict

    def create_final_list(self, all_json, keys):
        if os.path.exists(self.final_cache_list) and self.cache:
            logger.warning("Get final list from cache")
            with open(self.final_cache_list, 'rb') as f:
                cached_list = torch.load(f)
                self.final_list_tensors = cached_list['input_ids']
                self.final_list_labels = cached_list['labels']
                self.final_commit_info = cached_list['commit_info']
            return

        logger.warning("Create final list")
        for commit in (pbar := tqdm(keys[:], leave=False)):
            token_arr_lst = handle_commit(
                all_json[commit],
                self.tokenizer,
                self.args,
                language=self.language,
                embedding_type=self.args.embedding_type)

            for token_arr in token_arr_lst:
                if token_arr is not None:
                    self.final_list_tensors.append(
                        torch.tensor(token_arr['input_ids']))
                    self.final_list_labels.append(
                        torch.tensor(int(all_json[commit]['label'])))
                    self.final_commit_info.append(all_json[commit])
            # print(len(token_arr_lst))
            if int(all_json[commit]['label']) == 1:
                self.positive_label_counter += len(token_arr_lst)
            else:
                self.negative_label_counter += len(token_arr_lst)

            pbar.set_description(
                f"Current Project: {all_json[commit]['name']} Positive: {self.positive_label_counter}, Negative: {self.negative_label_counter}")

        if self.args.balance:
            self.balance_data()

        with open(self.final_cache_list, 'wb') as f:
            torch.save({"input_ids": self.final_list_tensors,
                       "labels": self.final_list_labels, "commit_info": self.final_commit_info}, f)

    def balance_data(self):
        logger.warning("Balance data")
        min_label = min(self.positive_label_counter,
                        self.negative_label_counter)
        wanted_negatives = [i for i, val in enumerate(
            self.final_list_labels) if val == 0][:min_label]
        wanted_positivies = [i for i, val in enumerate(
            self.final_list_labels) if val == 1][:min_label]
        self.final_list_tensors = [self.final_list_tensors[x]
                                   for x in wanted_negatives + wanted_positivies]
        self.final_list_labels = [self.final_list_labels[x]
                                  for x in wanted_negatives + wanted_positivies]
        logger.warning(
            f"Positives: {len(wanted_positivies)}, Negatives: {len(wanted_negatives)}, Total: {len(self.final_list_tensors)}")

    def __len__(self):
        return len(self.final_list_tensors)

    def __getitem__(self, i):
        return self.final_list_tensors[i], self.final_list_labels[i]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding-type", "-et", default="simple", type=str)

    parser.add_argument("--output_dir", default='./cache_data/saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--language", default="all", type=str)
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="cache_data", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=300, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_false',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_false',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_false',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_false',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--generate_data",
                        action='store_true', help="generate data")

    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=50,
                        help="random seed for initialization")
    parser.add_argument('--recreate-cache', action='store_true',
                        help="recreate the language model cache")
    parser.add_argument('--hyperparameter',
                        action='store_true', help="hyperparameter")
    parser.add_argument('--commit_repos_path', type=str,
                        default=r"D:\multisource\commits")
    parser.add_argument('--use_roberta_classifer',
                        action='store_true', help="use roberta classifer")
    parser.add_argument('--pooler_type', type=str,
                        default="cls", help="poller type")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument("--source_model", type=str,
                        default="Text", help="source model")
    parser.add_argument("--balance", action="store_true",
                        help="balance data to keep positives and negatives the same")
    parser.add_argument("--event_window_size", type=int, default=10, help="event window size")
    parser.add_argument("--code_merge_file", action="store_true", help="code merge file")
    parser.add_argument("--folds", type=int, default=5, help="folds")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer, dataset, eval_idx=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SubsetRandomSampler(eval_idx)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=0, pin_memory=True)

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
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "labels": labels,
        "preds": preds,

    }
    return result


def train(args, train_dataset, model, tokenizer, fold, idx, eval_idx=None):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SubsetRandomSampler(idx)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True)

    args.max_steps = args.epoch*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num GPUS = %d", args.n_gpu)
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0

    model.zero_grad()

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            final_train_loss = avg_loss
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(
                            args, model, tokenizer, train_dataset, eval_idx=eval_idx)
                        logger.warning(
                            f"eval_loss {float(results['eval_loss'])}")
                        logger.warning(
                            f"train_loss {final_train_loss}")
                        logger.warning(
                            f"eval_acc {round(results['eval_acc'],4)}")

                        # Save model checkpoint

                    if results['eval_acc'] > best_acc:
                        best_acc = results['eval_acc']
                        logger.info("  "+"*"*20)
                        logger.info("  Best acc:%s", round(best_acc, 4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format(f'model_{fold}.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)

                    wandb.log({ f"Fold {fold} Epoch" : idx, f"{fold}_train_loss": final_train_loss, "global_step": global_step, f"{fold}_eval_loss": results['eval_loss'], f"{fold}_eval_acc": best_acc})

    return best_acc


def test(args, model, tokenizer, eval_dataset):

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(
        eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5

    res = []
    for example, pred in zip(eval_dataset.final_commit_info, preds):
        res.append([example['name'], example['hash'], pred, example['label']])

    table = wandb.Table(
        columns=["Name", "Hash", "Prediction", "Actual"], data=res)
    wandb.log({"test_table": table})


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = outputs
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0]+1e-10)*labels + \
                torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob


def main(args):
    if args.hyperparameter:
        wandb.init(config=args, tags=[args.embedding_type, args.language])

        args.learning_rate = wandb.config.lr
        args.train_batch_size = wandb.config.batch_size
        args.eval_batch_size = wandb.config.batch_size
        args.epochs = wandb.config.epochs
        args.weight_decay = wandb.config.weight_decay
        args.max_grad_norm = wandb.config.max_grad_norm

    args.epoch = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    if args.n_gpu == 0:
        args.n_gpu = 1

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    args.output_dir = args.cache_dir
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(
            checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(
            checkpoint_last, args.start_epoch))

    if args.cache_dir:
        args.model_cache_dir = os.path.join(args.cache_dir, "models")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.model_cache_dir if args.model_cache_dir else None, num_labels=2)
    config.num_labels = 2
    config.hidden_dropout_prob = args.dropout
    config.classifier_dropout = args.dropout
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.model_cache_dir if args.model_cache_dir else None)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool(
                                                '.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.model_cache_dir if args.model_cache_dir else None)
    else:
        model = model_class(config)

    if args.model_type == "roberta_classification":
        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout
        model = RobertaClass(model, args)
        logger.warning("Using RobertaClass")
    else:
        model = Model(model, config, tokenizer, args)

    if args.local_rank == 0:
        # End of barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    logger.warning("Training/evaluation parameters %s", args)
    with open(os.path.join(args.cache_dir,"orc", "orchestrator.json"), "r") as f:
        mall = json.load(f)

    if args.source_model == "Text":
        args.Dataset = TextDataset
        dataset = TextDataset(tokenizer, args, mall, mall.keys(), "train")
        model.encoder.resize_token_embeddings(len(tokenizer))

    elif args.source_model == "Events":
        from events_datasets import EventsDataset
        args.Dataset = EventsDataset
        dataset = EventsDataset(args, mall, mall.keys(), "train")

        xshape1 = dataset[0][0].shape[0]
        xshape2 = dataset[0][0].shape[1]
        events_config = {'l1': 64, 'l2': 32, 'l3': 16,
                        'l4': 128, 'lr': 0.01, 'dropout': 0.2, 'batch_size': 512,
                        'optimizer': optim.Adam}
        model = Conv1DTune(args,
                        xshape1, xshape2,  l1=events_config["l1"], l2=events_config["l2"], l3=events_config["l3"], l4=events_config["l4"])
        model = model.to(args.device)

    else:
        raise NotImplementedError
    
    best_acc = 0
    fold_best_acc = 0

    splits=KFold(n_splits=args.folds,shuffle=True,random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))

        # Training
        if args.do_train:
            if args.local_rank not in [-1, 0]:
                # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
                torch.distributed.barrier()

            if args.local_rank == 0:
                torch.distributed.barrier()

            acc = train(args, dataset, model, tokenizer, fold, train_idx, eval_idx=val_idx)

            if acc > best_acc:
                best_acc = acc
                fold_best_acc = fold

            print(f"Current acc: {acc}, best acc: {best_acc}, best fold: {fold_best_acc}")
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoint_prefix = 'checkpoint-best-acc/model_{fold}.bin'
    #     output_dir = os.path.join(
    #         args.output_dir, '{}'.format(checkpoint_prefix))
    #     model.load_state_dict(torch.load(output_dir))
    #     model.to(args.device)
    #     results = evaluate(args, model, tokenizer, eval_dataset=eval_idx)
    #     logger.info("***** Eval results *****")
    #     logger.warning(f"eval_loss {float(results['eval_loss'])}")
    #     logger.warning(f"eval_acc {round(results['eval_acc'],4)}")

    # if args.do_test and args.local_rank in [-1, 0]:
    #     checkpoint_prefix = 'checkpoint-best-acc/model_{fold}.bin'
    #     output_dir = os.path.join(
    #         args.output_dir, '{}'.format(checkpoint_prefix))
    #     model.load_state_dict(torch.load(output_dir))
    #     model.to(args.device)
    #     test(args, model, tokenizer, test_dataset)
    #     wandb.log_artifact(output_dir, type='model')

    return {}


def initalize_wandb(args):
    import wandb
    # Define sweep config
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'eval_loss'},
        'parameters':
        {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [5, 10, 15, 20]},
            'lr': {'max': 0.1, 'min': 1e-5},
            'weight_decay': {'max': 0.1, 'min': 1e-5},
            'max_grad_norm': {'max': 5, 'min': 0},
        }
    }

    from functools import partial
    if args.hyperparameter:
        sweep_id = wandb.sweep(
            sweep=sweep_configuration,
            project=PROJECT_NAME
        )
        wandb.agent(sweep_id, function=partial(main, args), count=20)

    else:
        wandb.init(project=PROJECT_NAME,
                   # name = args.model_type + '-' + args.embedding_type + '-' + args.language,
                   tags=[args.embedding_type, args.language],
                   config=args
                   )


if __name__ == "__main__":

    args = parse_args()

    initalize_wandb(args)

    if not args.hyperparameter:
        main(args)
