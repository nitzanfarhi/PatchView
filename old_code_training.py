# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
from collections import defaultdict
import pickle
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from pydriller import Repository
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from events_datasets import GeneralDataset
from code_model import *
import multiprocessing
from tqdm import tqdm, trange
import csv
import git

import json
import difflib
import argparse
import logging
import os
import random
import numpy as np
import torch
import datetime

from misc import set_seed
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter


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
MAXIMAL_FILE_SIZE = 1000000
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:128'


INS_TOKEN='[INS]'
DEL_TOKEN='[DEL]'
REP_BEFORE_TOKEN='[RBT]'
REP_AFTER_TOKEN='[RAT]'

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'])


def get_commit_from_repo(cur_repo, hash):
    return next(Repository(cur_repo, single=hash).traverse_commits())


def concat_tokenize(cur_txt, tokenizer, args):
    code_tokens = tokenizer.tokenize(cur_txt)
    source_tokens = [tokenizer.cls_token] + \
        code_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(
        source_tokens)
    if len(source_ids) > args.block_size:
        return []
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return source_ids

def embed_txt_and_pad(txt, tokenizer, args):
    code_tokens = tokenizer.tokenize(txt)
    source_tokens = [tokenizer.cls_token]
    source_tokens += code_tokens+[tokenizer.sep_token]

    source_ids = tokenizer.convert_tokens_to_ids(
        source_tokens)
    if len(source_ids) > args.block_size:
        return None
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)


def convert_to_ids_and_pad(source_tokens, tokenizer, args):
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    if len(source_ids) > args.block_size:
        return None
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id]*padding_length
    return torch.tensor(source_ids)



def embed_file(file, tokenizer, args):
    before, after = "",""
    if file.content_before is not None:
        before = file.content_before.decode('utf-8')
    else:
        before = ""

    if file.content is not None:
        after = file.content.decode('utf-8')

    if len(before) > MAXIMAL_FILE_SIZE or len(after) > MAXIMAL_FILE_SIZE:
        return None

    operation_list = []
    for opp,a1,a2,b1,b2 in difflib.SequenceMatcher(a=before,b=after).get_opcodes():
        match opp:
            case 'equal':
                continue

            case 'replace':
                after_tokens = tokenizer.tokenize(after[b1:b2])
                before_tokens = tokenizer.tokenize(before[a1:a2])
                res = []
                res += [tokenizer.cls_token]
                res += [tokenizer.sep_token]
                res += before_tokens
                res += [tokenizer.sep_token]
                res += after_tokens
                res += [tokenizer.sep_token]
                final_tensor = convert_to_ids_and_pad(res, tokenizer, args)
                operation_list.append(final_tensor)
      
                # after_tokens = embed_txt_and_pad(after[b1:b2], tokenizer, args)
                # before_tokens = embed_txt_and_pad(
                #     before[a1:a2], tokenizer, args
                # )
                # final_tensor = torch.sub(after_tokens,before_tokens)
                # final_tensor[final_tensor == 0] = tokenizer.pad_token_id


            case 'insert':
                after_tokens = tokenizer.tokenize(after[b1:b2])
                res = []
                res += [tokenizer.cls_token]
                res += [tokenizer.sep_token]
                res += after_tokens
                res += [tokenizer.sep_token]
                final_tensor = convert_to_ids_and_pad(res, tokenizer, args)
                operation_list.append(final_tensor)

                # final_tensor = embed_txt_and_pad(after[b1:b2], tokenizer, args)
                # if final_tensor is None:
                #     continue

            case 'delete':
                before_tokens = tokenizer.tokenize(before[a1:a2])
                res = []
                res += [tokenizer.cls_token]
                res += [tokenizer.sep_token]
                res += before_tokens
                res += [tokenizer.sep_token]
                final_tensor = convert_to_ids_and_pad(res, tokenizer, args)
                operation_list.append(final_tensor)

                # final_tensor = torch.sub(0,final_tensor)
                # final_tensor[
                #     final_tensor == -tokenizer.pad_token_id
                # ] = tokenizer.pad_token_id

            case _:
                raise ValueError(f"Unknown operation: {opp}")

    return operation_list

def handle_commit(commit, tokenizer, args, language='all', add_sep_between_lines=True, embedding_type='concat'):
    res = []
    for file in commit.modified_files:
        if "." not in file.filename:
            continue
        filetype = file.filename.split(".")[-1].lower()
        if language != "all" and filetype != language.lower():
            continue
        if embedding_type == 'concat':
            source_tokens = [tokenizer.cls_token]

            for line in file.diff_parsed['added']:
                source_tokens += tokenizer.tokenize(line[1])
                if add_sep_between_lines:
                    source_tokens += [tokenizer.sep_token]
            
            if not add_sep_between_lines:
                source_tokens += [tokenizer.sep_token]

            for line in file.diff_parsed['deleted']:
                source_tokens += tokenizer.tokenize(line[1])
                if add_sep_between_lines:
                    source_tokens += [tokenizer.sep_token]

           
            if len(source_tokens) > args.block_size:
                source_tokens = source_tokens[:args.block_size]
            else:
                padding_length = args.block_size - len(source_tokens)
                source_tokens += [tokenizer.pad_token_id]*padding_length

            res.append(torch.tensor(tokenizer.convert_tokens_to_ids(source_tokens)))

        elif embedding_type == 'sum':
            embed_file_res = embed_file(file, tokenizer, args)
            if embed_file_res is not None:
                res += embed_file_res

    return res


def filter_repos(csv_list, file_path, language):
    filtered_csv_list = []
    for repo, cur_hash, label in tqdm(csv_list):
        try:
            commit = get_commit_from_repo(
                os.path.join(file_path, repo), cur_hash)
            for file in commit.modified_files:
                if "." not in file.filename:
                    continue
                filetype = file.filename.split(".")[-1].lower()
                if filetype == language.lower():
                    filtered_csv_list.append([repo, cur_hash, label])
                    break
        except Exception:
            continue
    return filtered_csv_list


def safe_makedir(path):
    try:
        os.makedirs(path)
    except:
        pass


class TextDataset(GeneralDataset):
    def __init__(self, tokenizer, args, phase, csv_list_dir=r"C:\Users\nitzan\local\analyzeCVE"):
        
        safe_makedir("languages_cache")
        self.csv_list_path = f"languages_cache\\{args.language}.csv"
        self.tokenizer = tokenizer
        self.args = args
        self.cache = args.cache
        self.language = args.language
        self.counter = 0
        self.current_path = f"languages_cache\\{self.language}_{self.args.embedding_type}_{phase}.json"
        self.final_list = []
        self.hash_list = []


        if phase == 'train':
            with open(os.path.join(csv_list_dir,"train_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        elif phase == 'val':
            with open(os.path.join(csv_list_dir,"validation_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        elif phase == 'test':
            with open(os.path.join(csv_list_dir,"test_details.pickle"), 'rb') as f:
                self.csv_list = pickle.load(f)
        else:
            raise ValueError(f"Unknown phase: {phase}")


        print(f"Caching: {self.cache} , Path to cache: {self.current_path}: {os.path.exists(self.current_path)}")
        if self.cache and os.path.exists(self.current_path):
            self.final_list = torch.load(self.current_path)[:]
        else:
            self.create_final_list()
            torch.save(self.final_list, self.current_path)

    def create_final_list(self):
        counter = 0
        for repo, _, label, cur_hash in tqdm(self.csv_list[:], leave=False):
            cur_hash = cur_hash.values[0]
            if cur_hash == '':
                counter+=1
                continue
            commit = get_commit_from_repo(
                os.path.join(COMMITS_PATH, repo), cur_hash)

            try:
                token_arr_lst = handle_commit(
                    commit,
                    self.tokenizer,
                    self.args,
                    language="",
                    embedding_type=self.args.embedding_type)

                for token_arr in token_arr_lst:
                    if token_arr is not None:
                        self.hash_list.append(f"{repo}/{cur_hash}")
                        self.final_list.append((token_arr, int(label)))
            except Exception as e:
                print(e)
                continue


    def __len__(self):
        return len(self.final_list)

    def __getitem__(self, i):
        source_ids, label = self.final_list[i]
        return source_ids, torch.tensor(int(label))




def oldtrain(train_dataset, model, name, args, eval_dataset=None, tokenizer=None):
    #  todo tensorboard handling https://discuss.pytorch.org/t/how-to-plot-train-and-validation-accuracy-graph/105524/2
    writer = SummaryWriter(f"log/{name}/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    writer.add_text("args",json.dumps(args.__dict__, default=lambda o: '<not serializable>'))

    # add graph to tensorboard

    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True)

    args.max_steps = args.epochs*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epochs
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)

    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0

    model.zero_grad()


    # print("Started")
    # history = defaultdict(list)
    for idx in trange(args.epochs, leave=False):
        tr_num = 0
        train_loss = 0
        for batch in tqdm(train_dataloader, leave=False):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, _ = model(inputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            avg_loss = round(
                np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logging_loss = tr_loss
                tr_nb = global_step

        print(f"Epoch: {idx} Loss: {avg_loss}")
        # if args.save_steps > 0 and global_step % args.save_steps == 0:
        #     path = os.path.join(args.output_dir, f'model_{args.model_name_or_path.replace("/","_")}_{args.embedding_type}_{args.language}.pth')
        #     torch.save({
        #                 'epoch': idx,
        #                 'model_state_dict': model.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'loss': train_loss,
        #                 }, path)
        #     writer.add_scalar("Loss/train", train_loss, idx)
        #     if args.evaluate_during_training:
        #         results = evaluate(
        #             args, model, tokenizer, eval_when_training=True)
        #         for key, value in results.items():
        #             logger.warn("  %s = %s", key, round(value, 4))
        #         # Save model checkpoint
        #         writer.add_scalar("Loss/eval", results['eval_loss'], idx)
        #         writer.add_scalar("Acc/eval", results['eval_acc'], idx)

        #         bar.set_description(f"epoch {idx} loss {results['eval_loss']}, acc {results['eval_acc']} ")

        #     if results['eval_acc'] > best_acc:
        #         best_acc = results['eval_acc']
        #         imporoved_accuracy(args, model, results)
            
        #     writer.flush()

    writer.close()



def imporoved_accuracy(args, model, results):
    best_acc = results['eval_acc']
    # logger.warn("  "+"*"*20)
    # logger.warn("  Best acc:%s", round(best_acc, 4))
    # logger.warn("  "+"*"*20)

    checkpoint_prefix = f'checkpoint-best-acc-{args.language}'
    output_dir = os.path.join(args.output_dir, f'{checkpoint_prefix}')
    safe_makedir(output_dir)
    model_to_save = model.module if hasattr(
        model, 'module') else model
    output_dir = os.path.join(output_dir, 'model.bin')
    torch.save(model_to_save.state_dict(), output_dir)
    logger.warn(
        "Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    safe_makedir(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
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

    return {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }


def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
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
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx+'\t1\n')
            else:
                f.write(example.idx+'\t0\n')


def generate_data(language):
    csv_list_path = r"C:\Users\nitzan\local\code_embed\repo_commits_with_label.csv"
    language = language
    with open(csv_list_path, 'r') as f:
        csv_list = list(csv.reader(f))
    csv_list = csv_list[1:]
    filtered_csv_list = filter_repos(
        csv_list, r"C:\Users\nitzan\local\analyzeCVE\data_collection\data\commits", language)
    with open(f"languages_cache\\{language}.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(filtered_csv_list)


def main():
    # Parse Args
    args = parse_args()

    # Setup CUDA
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARN)
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    # do_checkpoint(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

    roberta = model_class(config)


    model = Model(roberta, config, tokenizer, args)

    logger.info("Training/evaluation parameters %s", args)

    if args.generate_data:
        generate_data(args.language)

    if args.do_train:
        from events_training import train as mytrain
        mconfig = { 'batch_size': args.batch_size, "optimizer":AdamW, "lr":0.1}
        model.to(args.device)
        # beep_train(model, TextDataset(tokenizer,args, 'train'), args)
        mytrain(model,mconfig, args, TextDataset(tokenizer,args, 'train'), TextDataset(tokenizer,args, 'val'), name ='code')
        # oldtrain(TextDataset(tokenizer, args, 'train'), model, 'code', args, tokenizer=tokenizer)

    if args.do_eval:
        load_model(args, model, 'code')
        result = evaluate(args, model, tokenizer)
        logger.warning("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.warning("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        load_model(args, model, 'code')
        test(args, model, tokenizer, TextDataset(tokenizer, args, 'test'), 'code')


def load_model(args, model, name):
    checkpoint_prefix = f'checkpoint-best-acc-{name}.bin'
    output_dir = os.path.join(args.output_dir, f'{checkpoint_prefix}')
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)


def do_checkpoint(args):
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

        logger.info(
            f"reload model from {checkpoint_last}, resume on {args.start_epoch}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding-type", default="concat", type=str)

    parser.add_argument("--output_dir", default='./saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--language", default="all", type=str)

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
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=300, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_false', 
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_false', 
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_false', 
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--generate_data",
                        action='store_true', help="generate data")

    parser.add_argument("--batch-size", default=8, type=int, help= "batch size")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
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

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=50,
                        help="random seed for initialization")
    parser.add_argument('--recreate-cache', action='store_true', help="recreate the language model cache")
    parser.add_argument('--cache', action='store_true', help="cache old data")

    return parser.parse_args()


if __name__ == "__main__":
    main()
