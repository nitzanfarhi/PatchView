from __future__ import absolute_import, division, print_function
from torch.nn import CrossEntropyLoss, MSELoss
import copy
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

import multiprocessing
from tqdm import tqdm
import csv
import pickle
import json
import difflib
import argparse
import logging
import os
import random
import numpy as np
import torch
import datetime

import traceback
import warnings

import wandb

from code_utils import ext_to_comment

# _old_warn = warnings.warn
# def warn(*args, **kwargs):

#     tb = traceback.extract_stack()
#     _old_warn(*args, **kwargs)
#     print("".join(traceback.format_list(tb)[:-1]))
# warnings.warn = warn



cpu_cont = multiprocessing.cpu_count()

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}



COMMITS_PATH = r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\commits"
MAXIMAL_FILE_SIZE = 100000 # files bigger than that will be ignored as they are probably binaries / not code

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

    logging.warning(f"Size Before - {len(before)}, Size After - {len(after)}, Opcode Number -  {len(opcodes)}")
    for opp, a1, a2, b1, b2 in opcodes:
        if opp == 'equal':
            continue

        elif opp == 'replace':
            res = REP_BEFORE_TOKEN+" "+before[a1:a2]+" "+REP_AFTER_TOKEN+" "+after[b1:b2]
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
            file_res = " ".join(added+deleted)
            file_res = tokenizer(file_res, truncation=True,
                                 padding='max_length', max_length=args.block_size)
            res.append(file_res)

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

    return res
def safe_makedir(path):
    try:
        os.makedirs(path)
    except:
        pass


class TextDataset(Dataset):

    def __init__(self, tokenizer, args, phase):
        logging.warning(f"Loading dataset {phase} {args.language}")
        self.tokenizer = tokenizer
        self.args = args
        self.phase = phase
        self.cache = not args.recreate_cache
        self.language = args.language
        self.counter = 0
        self.final_list_tensors = []
        self.final_list_labels = []
        self.final_commit_info = []
        self.commit_path = os.path.join(args.cache_dir,f"{self.language}_{self.phase}.git")
        self.final_cache_list = os.path.join(args.cache_dir,f"{args.embedding_type}_{self.language}_{self.phase}_final_list.pickle")
        self.positive_label_counter = 0
        self.negative_label_counter = 0
        self.commit_repos_path = args.commit_repos_path

        if phase == 'train':
            with open(os.path.join(args.cache_dir,"train_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        elif phase == 'val':
            with open(os.path.join(args.cache_dir,"validation_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        elif phase == 'test':
            with open(os.path.join(args.cache_dir,"test_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        else:
            raise ValueError(f"Unknown phase: {phase}")


        if self.args.embedding_type == 'simple_with_tokens':
            logging.warning(f"Tokenizer size before adding tokens: {len(self.tokenizer)}")
            self.tokenizer.add_special_tokens({'additional_special_tokens': [ADD_TOKEN, DEL_TOKEN]})
        elif self.args.embedding_type == "sum":
            self.tokenizer.add_special_tokens({'additional_special_tokens': [REP_BEFORE_TOKEN, REP_AFTER_TOKEN, INS_TOKEN, DEL_TOKEN]})


        commit_list = self.get_commits()
        logging.warning(f"Number of commits: {len(commit_list)}")
        self.create_final_list(commit_list)
        logging.warning(f"Number of instances: {len(self.final_list_tensors)}")

    def get_commits(self):
        result = []

        if os.path.exists(self.commit_path):
            logging.warning("Get Commits from cache")
            with open(self.commit_path, 'rb') as f:
                return pickle.load(f)
        else:
            logging.warning("Get Commits from repos")
            for mdict in tqdm(self.csv_list[:]):
                try:
                    repo = mdict["repo_name"]
                    label = mdict["label"]
                    cur_hash = mdict["hash"]
                    if cur_hash == "":
                        continue
                    result.append(self.prepare_dict(repo, label, cur_hash))
                except ValueError:
                    continue

            with open(self.commit_path, 'wb') as f:
                pickle.dump(result, f)

            return result

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
            cur_dict = {}
            before = ""
            after = ""
            if file.content_before is not None:
                try:
                    before = file.content_before.decode('utf-8')
                except UnicodeDecodeError:
                    continue
            else:
                before = ""

            if file.content is not None:
                try:
                    after = file.content.decode('utf-8')
                except UnicodeDecodeError:
                    continue
            if "." not in file.filename:
                continue
            if len(after) > MAXIMAL_FILE_SIZE or len(before) > MAXIMAL_FILE_SIZE:
                continue

            filetype = file.filename.split(".")[-1].lower()
            cur_dict["filetype"] = filetype
            cur_dict["filename"] = file.filename
            cur_dict["content"] = after
            cur_dict["before_content"] = before
            cur_dict["added"] = file.diff_parsed["added"]
            cur_dict["deleted"] = file.diff_parsed["deleted"]
            final_dict["files"].append(cur_dict)

        return final_dict

    def create_final_list(self, commit_list):
        if os.path.exists(self.final_cache_list) and self.cache:
            logging.warning("Get final list from cache")
            with open(self.final_cache_list, 'rb') as f:
                cached_list = torch.load(f)
                self.final_list_tensors = cached_list['input_ids']
                self.final_list_labels = cached_list['labels']
                self.final_commit_info = cached_list['commit_info']
            return

        logging.warning("Create final list")
        for commit in (pbar := tqdm(commit_list[:], leave=False)):
            token_arr_lst = handle_commit(
                commit,
                self.tokenizer,
                self.args,
                language=self.language,
                embedding_type=self.args.embedding_type)

            for token_arr in token_arr_lst:
                if token_arr is not None:
                    self.final_list_tensors.append(torch.tensor(token_arr['input_ids']))
                    self.final_list_labels.append(torch.tensor(int(commit['label'])))
                    self.final_commit_info.append(commit)
            if int(commit['label']) == 1:
                self.positive_label_counter += len(token_arr_lst)
            else:
                self.negative_label_counter += len(token_arr_lst)
            

            pbar.set_description(f"Current Project: {commit['name']} Positive: {self.positive_label_counter}, Negative: {self.negative_label_counter}")


        with open(self.final_cache_list, 'wb') as f:
            torch.save({"input_ids":self.final_list_tensors, "labels":self.final_list_labels, "commit_info": self.final_commit_info}, f)

        
    def __len__(self):
        return len(self.final_list_tensors)

    def __getitem__(self, i):
        return self.final_list_tensors[i], self.final_list_labels[i]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding-type","-et", default="simple", type=str)

    parser.add_argument("--output_dir", default='./saved_models', type=str,
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

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
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
    parser.add_argument('--hyperparameter', action='store_true', help="hyperparameter")
    parser.add_argument('--commit_repos_path', type=str, default=r"D:\multisource\commits")

    return parser.parse_args()


# def main():
#     args = parse_args()
#     print("Loading model")
#     from transformers import AutoTokenizer, AutoModelForSequenceClassification

#     from transformers import pipeline
#     # model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
#     # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

#     model = AutoModelForSequenceClassification.from_pretrained(
#         'mrm8488/codebert-base-finetuned-detect-insecure-code', cache_dir=args.cache_dir)
#     tokenizer = AutoTokenizer.from_pretrained(
#         'mrm8488/codebert-base-finetuned-detect-insecure-code', cache_dir=args.cache_dir)

#     import numpy as np
#     from datasets import load_metric
#     import wandb
#     wandb.init(project="msd",
#                 tags = [args.embedding_type, args.language],
#                 config = args
#                 )
    
#     load_accuracy = load_metric("accuracy")

#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred
#         predictions = np.argmax(predictions, axis=1)
#         return load_accuracy.compute(predictions=predictions, references=labels)

#     from transformers import TrainingArguments, Trainer


#     train_dataset = TextDataset(tokenizer, args, 'train')
#     eval_dataset = TextDataset(tokenizer, args, 'val')

#     if args.embedding_type == "simple_with_tokens":
#         model.resize_token_embeddings(len(tokenizer))


#     trng_args = TrainingArguments(output_dir=args.embedding_type,
#                                   evaluation_strategy="epoch",
#                                   num_train_epochs=args.epochs,
#                                   resume_from_checkpoint=True,
#                                   learning_rate=args.learning_rate,
#                                   report_to="wandb")
#     trainer = Trainer(model=model, args=trng_args, train_dataset=train_dataset,
#                       eval_dataset=eval_dataset, compute_metrics=compute_metrics)
#     trainer.train()

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
def evaluate(args, model, tokenizer,eval_dataset=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=0,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 :
        model = torch.nn.DataParallel(model)


    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
            
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
        "labels": labels,
        "preds": preds,

    }
    return result

def train(args, train_dataset, model, tokenizer, eval_dataset=None):
    """ Train the model """ 

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=0,pin_memory=True)
    
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
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
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_acc=0.0

    model.zero_grad()
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,eval_dataset = eval_dataset)
                        logging.warning(f"eval_loss {float(results['eval_loss'])}")
                        logging.warning(f"eval_acc {round(results['eval_acc'],4)}")

                        # Save model checkpoint
                        
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best acc:%s",round(best_acc,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    wandb.log({"train_loss":avg_loss, "global_step":global_step, "epoch":idx, "eval_loss": results['eval_loss'], "eval_acc": results['eval_acc']})

                        

from torch.utils.data.distributed import DistributedSampler




def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args,'test')


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
    logits=[]   
    labels=[]
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)        
        label=batch[1].to(args.device) 
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5

    res = []
    for example,pred in zip(eval_dataset.final_commit_info,preds):
        res.append([example['name'], example['hash'], pred])

    table = wandb.Table(columns=["Name", "Hash", "Prediction"], data=res)
    wandb.log({"test_table": table})



        

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob




def main2(args):
    if args.hyperparameter:
        wandb.init(config=args, tags = [args.embedding_type, args.language])

        args.learning_rate = wandb.config.lr
        args.batch_size = wandb.config.batch_size
        args.epochs = wandb.config.epochs
        args.weight_decay = wandb.config.weight_decay
        args.max_grad_norm = wandb.config.max_grad_norm

    args.epoch = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)


    args.start_epoch = 0
    args.start_step = 0
    args.output_dir = args.cache_dir
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    eval_dataset = TextDataset(tokenizer, args,'val') if args.evaluate_during_training else None

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, "train")

        logging.warning(f"Tokenizer size after : {len(tokenizer)}")
        model.encoder.resize_token_embeddings(len(tokenizer))
    

        if args.local_rank == 0:
            torch.distributed.barrier()
            
        train(args, train_dataset, model, tokenizer, eval_dataset=eval_dataset)



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            results=evaluate(args, model, tokenizer, eval_dataset=eval_dataset)
            logger.info("***** Eval results *****")
            logging.warning(f"eval_loss {float(results['eval_loss'])}")
            logging.warning(f"eval_acc {round(results['eval_acc'],4)}")

            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test(args, model, tokenizer)

    return results

def initalize_wandb(args):
    import wandb
    # Define sweep config
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'eval_acc'},
        'parameters': 
        {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 1e-5},
            'weight_decay': {'max': 0.1, 'min': 1e-5},
            'max_grad_norm' : {'max': 5, 'min': 0},


        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='msd'
    )


    from functools import partial
    if args.hyperparameter:
        wandb.agent(sweep_id, function=partial(main2,args), count=4)

    else:
        wandb.init(project="msd",
            # name = args.model_type + '-' + args.embedding_type + '-' + args.language,
            tags = [args.embedding_type, args.language],
            config = args
            )


if __name__ == "__main__":


    args = parse_args()


    initalize_wandb(args)

    if not args.hyperparameter:
        main2(args)




