import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)


from torch.utils.data import DataLoader, SubsetRandomSampler
from models import get_model
from tqdm import tqdm
from sklearn.model_selection import KFold
from datasets import EventsDataset, MyConcatDataset, TextDataset
import wandb
import torch
import numpy as np
import random
import os
import argparse
import sys
import json
from transformers import (get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)


os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()
PROJECT_NAME = 'MSD2'


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'roberta_classification': (RobertaConfig, RobertaModel, RobertaTokenizer),
}


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def parse_args():
    parser = argparse.ArgumentParser()

    # Data Related arguments
    parser.add_argument("--output_dir", default='./cache_data/saved_models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--cache_dir", default="cache_data", type=str,
                        help=" directory to store cache data")

    parser.add_argument("--block_size", default=300, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--recreate_cache', action='store_true',
                        help="recreate the language model cache")

    parser.add_argument('--commit_repos_path', type=str,
                        default=r"D:\multisource\commits")

    # Training parameters
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout")
    parser.add_argument('--epochs', type=int, default=50,
                        help="number of epochs to train")
    parser.add_argument('--pooler_type', type=str,
                        default="cls", help="poller type")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--folds", type=int, default=5, help="folds")
    parser.add_argument("--patience", type=int, default=100, help="patience")
    parser.add_argument(
        "--return_class", action="store_true", help="return class")

    # Source related arguments
    parser.add_argument("--source_model", type=str,
                        default="Multi", help="source model")

    # Multi related arguments
    parser.add_argument("--multi_model_type", type=str,
                        default="multiv1", help="multi model type")

    # Events related arguments
    parser.add_argument("--events_model_type", type=str,
                        default="conv1d", help="events model type")
    parser.add_argument("--event_window_size", type=int,
                        default=10, help="event window size")

    # Code related arguments
    parser.add_argument("--code_merge_file",
                        action="store_true", help="code merge file")
    parser.add_argument("--code_model_type", type=str,
                        default="roberta_classification", help="code model type")
    parser.add_argument("--code_embedding_type", "-cet",
                        default="simple", type=str)
    parser.add_argument("--code_model_name", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--code_tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    # Message related arguments
    parser.add_argument("--message_model_type", type=str,
                        default="roberta_classification", help="message model type")
    parser.add_argument("--message_embedding_type", "-met",
                        default="commit_message", type=str)
    parser.add_argument("--message_model_name", default="roberta-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--message_tokenizer_name", default="roberta-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, dataset, eval_idx=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
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
        if args.source_model == "Multi":
            inputs = [x.to(args.device) for x in batch[0]]
        else:
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
    best_acc = 0
    best_threshold = 0
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    for i in range(100):
        preds = logits[:, 0] > i/100
        eval_acc = np.mean(labels == preds)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_threshold = i/100
    logger.info("best threshold: {}".format(best_threshold))
    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(best_acc, 4),
        "labels": labels,
        "preds": preds,

    }
    return result


def train(args, train_dataset, model, fold, idx, run, eval_idx=None):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = SubsetRandomSampler(idx)

    # print train_dataset label percentages
    negatives = 0
    positives = 0

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True, drop_last=True)

    for a in train_dataloader:
        for b in a[1]:
            if b == 0:
                negatives += 1
            elif b == 1:
                positives += 1
            else:
                raise ValueError("label error")
    logger.warning("dataset balance percentage: {}".format(
        positives/(positives+negatives)))
    run.summary[f"balance_{fold}"] = positives/(positives+negatives)

    args.max_steps = args.epochs*len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
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
    logger.info("  Num examples = %d , evaluation = %d",
                len(idx), len(eval_idx))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0

    model.zero_grad()

    early_stopper = EarlyStopper(patience=args.patience, min_delta=0)

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
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
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    results = evaluate(
                        args, model, train_dataset, eval_idx=eval_idx)
                    logger.warning(
                        f"eval_loss {float(results['eval_loss'])}")
                    logger.warning(
                        f"train_loss {final_train_loss}")
                    logger.warning(
                        f"eval_acc {round(results['eval_acc'],4)}")

                    if early_stopper.early_stop(results['eval_acc']):
                        logger.info("Early stopping")
                        return best_acc

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

                    wandb.log({f"epoch": idx, f"train_loss": final_train_loss, "global_step": global_step,
                              f"eval_loss": results['eval_loss'], f"eval_acc": results['eval_acc']})

    return best_acc


def test(args, model, dataset, idx, fold=0):

    train_sampler = torch.utils.data.Subset(dataset, idx)
    eval_dataloader = DataLoader(
        train_sampler, batch_size=args.train_batch_size, num_workers=0, pin_memory=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(idx))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        if args.source_model == "Multi":
            inputs = [x.to(args.device) for x in batch[0]]
        else:
            inputs = batch[0].to(args.device)

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
        preds = logits[:, 0] > i/100
        eval_acc = np.mean(labels == preds)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_threshold = i/100

    preds = logits[:, 0] > best_threshold

    res = []
    infos = [dataset.final_commit_info[x] for x in idx]
    for example, pred in zip(infos, preds):
        res.append([example['name'], example['hash'], pred, example['label']])

    table = wandb.Table(
        columns=["Name", "Hash", "Prediction", "Actual"], data=res)
    wandb.log({f"test_table_{fold}": table})
    wandb.run.summary[f"test_acc_{fold}"] = best_acc
    wandb.run.summary[f"test_threshold_{fold}"] = best_threshold


def get_tokenizer(args, model_type, tokenizer_name):
    _, _, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.model_cache_dir if args.model_cache_dir else None)
    if args.block_size <= 0:
        # Our input block size will be the max possible for the model
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    return tokenizer


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    if args.n_gpu == 0:
        args.n_gpu = 1

    args.per_gpu_train_batch_size = args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size//args.n_gpu
    # Setup logging
    logger.warning(f"Device: {device}, n_gpu: {args.n_gpu}")

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    args.output_dir = args.cache_dir

    if args.cache_dir:
        args.model_cache_dir = os.path.join(args.cache_dir, "models")

    logger.warning("Training/evaluation parameters %s", args)

    with open(os.path.join(args.cache_dir, "orc", "orchestrator.json"), "r") as f:
        mall = json.load(f)

    if args.source_model == "Code":
        tokenizer = get_tokenizer(
            args, args.code_model_type, args.code_tokenizer_name)
        dataset = TextDataset(tokenizer, args, mall,
                              mall.keys(), args.code_embedding_type)

    elif args.source_model == "Message":
        tokenizer = get_tokenizer(
            args, args.message_model_type, args.message_tokenizer_name)
        dataset = TextDataset(tokenizer, args, mall,
                              mall.keys(), args.message_embedding_type)

    elif args.source_model == "Events":
        tokenizer = None
        dataset = EventsDataset(args, mall, mall.keys(), "train")

    elif args.source_model == "Multi":
        tokenizer = None
        dataset = get_multi_dataset(args, mall)
    else:
        raise NotImplementedError

    best_acc = 0
    splits = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    best_accs = []
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        print('Fold {}'.format(fold + 1))
        with wandb.init(project=PROJECT_NAME, tags=[args.source_model],  config=args, name=f"{args.source_model}_{fold}") as run:
            model = get_model(args, dataset, tokenizer)
            run.define_metric("epoch")
            best_acc = train(args, dataset, model, fold,
                             train_idx, run, eval_idx=val_idx)
            best_accs.append(best_acc)
            test(args, model, dataset, val_idx, fold=fold)

            if fold == args.folds - 1:
                wandb.alert(title="Finished last run",
                            text=f"Run {max(best_accs)}: {' '.join(sys.argv)}")

    return {}


def get_multi_dataset(args, mall):
    keys = sorted(list(mall.keys()))

    code_tokenizer = get_tokenizer(
        args, args.code_model_type, args.code_tokenizer_name)
    code_dataset = TextDataset(
        code_tokenizer, args, mall, keys, args.code_embedding_type)

    message_tokenizer = get_tokenizer(
        args, args.message_model_type, args.message_tokenizer_name)
    message_dataset = TextDataset(
        message_tokenizer, args, mall, keys, args.message_embedding_type)

    events_dataset = EventsDataset(args, mall, keys)
    args.xshape1 = events_dataset[0][0].shape[0]
    args.xshape2 = events_dataset[0][0].shape[1]
    concat_dataset = MyConcatDataset(
        code_dataset, message_dataset, events_dataset)

    return concat_dataset


if __name__ == "__main__":
    args = parse_args()
    main(args)
