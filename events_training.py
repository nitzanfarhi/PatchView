import argparse
import datetime
from functools import partial
import json
import logging
import os
import random
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from events_datasets import EventsDataset, create_datasets
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset

from data_utils import extract_dataset, Aggregate, pad_and_fix, split_into_x_and_y, split_repos
from events_models import Conv1D, Conv1DTune, LSTMClassification, RNNModel
from torch.utils.tensorboard import SummaryWriter


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner


SEED = 1337

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
train_losses = []
valid_losses = []


def accuracy(model, device, dataloader):
    model.eval()
    max_acc = 0
    final_threshold = 0
    for i in range(1,100):
        total_correct = 0
        total_instances = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                classifications = 1*(model(images).squeeze(1) > i*0.01)
                correct_predictions = sum(classifications == labels).item()
                total_correct += correct_predictions
                total_instances += len(images)
        if total_correct / total_instances > max_acc:
            max_acc = total_correct / total_instances
            final_threshold = i*0.01
    # print(f"Max accuracy: {max_acc} with threshold: {final_threshold}")
    return max_acc


def train(model, config, args=None, train_dataset=None, validation_dataset=None, name='events', calc_acc=False):

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True)
    if validation_dataset is not None:
        valid_loader = DataLoader(
            validation_dataset, batch_size=config["batch_size"], shuffle=True)

    writer = SummaryWriter(
        f"log/{name}/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    writer.add_text("args", json.dumps(
        args.__dict__, default=lambda o: '<not serializable>'))

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = config["optimizer"](model.parameters(), lr=config["lr"])

    history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_accuracy = 0
    for epoch in trange(args.epochs):
        losses = []
        valid_losses = []
        model.train()

        # (pbar := tqdm(list(trainloader)[:], leave=False)):
        for data in (pbar := tqdm(train_loader, leave=False)):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            tag_scores = model(inputs)
            labels = labels.unsqueeze(1)

            loss = loss_function(tag_scores.float(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
            pbar.set_description(f"curloss - {loss}")

        avg_loss = np.mean(losses)

        if validation_dataset is not None:
            model.eval()
            for data in valid_loader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                labels = labels.unsqueeze(1)
                tag_scores = model(inputs)
                loss = loss_function(tag_scores.float(), labels.float())
                valid_losses.append(float(loss))
            avg_valid_loss = np.mean(valid_losses)
        else:
            avg_valid_loss = 0

        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_valid_loss)
        if calc_acc:
            history['acc'].append(accuracy(model, args.device, train_loader))
            history['val_acc'].append(accuracy(model, args.device, valid_loader))
        else:
            history['acc'].append(0)
            history['val_acc'].append(0)

            print(
                 f'Epoch {epoch} - loss: {avg_loss} - val_loss: {avg_valid_loss} - acc: {history["acc"][-1]} - val_acc: {history["val_acc"][-1]}')

        writer.add_scalars('Loss', {
                           "train": history['loss'][-1],
                           "validation": history['val_loss'][-1]},
                           epoch)

        writer.add_scalars('Accuracy', {
                           "train": history['acc'][-1],
                           "validation": history['val_acc'][-1]},
                           epoch)
        
        writer.add_scalar("loss/eval", history['val_loss'][-1], epoch)
        writer.add_scalar("acc/eval", history['val_acc'][-1], epoch)
        writer.add_scalar("loss/train", history['loss'][-1], epoch)
        writer.add_scalar("acc/train", history['acc'][-1], epoch)

        if history['val_acc'][-1] > best_val_accuracy:
            best_val_accuracy = history['val_acc'][-1]
            checkpoint_prefix = f'checkpoint-best-acc-events.bin'
            output_dir = os.path.join(args.output_dir, f'{checkpoint_prefix}')
            torch.save(model.state_dict(), output_dir)


    return history


def evaluate(args, model, valid_loader, criterion, optimizer, device):
    running_loss = .0

    model.eval()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(inputs.float())
            loss = criterion(preds, labels)
            running_loss += loss
            # print(idx)

        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        print(f'valid_loss {valid_loss}')

    return valid_losses / len(valid_loader) 


def test(args, model, dataset, name):
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), leave=False):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    with open(os.path.join("saved_models", f"{name}_predictions.txt"), 'w') as f:
        for example, pred in zip(dataset.info, preds):
            if pred:
                f.write(str(example)+'\t1\n')
            else:
                f.write(str(example)+'\t0\n')

def load_model(args, model, name):
    checkpoint_prefix = f'checkpoint-best-acc-{name}.bin'
    output_dir = os.path.join(args.output_dir, f'{checkpoint_prefix}')
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)



def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # Parse Args
    args = parse_args()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARN)
    # Set seed
    set_seed(SEED)


    train_dataset, validation_datatest, test_dataset = create_datasets(EventsDataset, backs=args.backs, cache = args.cache)
    xshape1 = train_dataset[0][0].shape[0]
    xshape2 = train_dataset[0][0].shape[1]

    args.start_epoch = 0
    args.start_step = 0

    config = {'l1': 64, 'l2': 32, 'l3': 16,
                'l4': 128, 'lr': 0.01, 'dropout': 0.2, 'batch_size': 512,
                'optimizer': optim.Adam}

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = Conv1DTune(
        xshape1, xshape2,  l1=config["l1"], l2=config["l2"], l3=config["l3"], l4=config["l4"])
    model = model.to(args.device)

    if args.do_train:
        train(model, config, args=args,
              train_dataset=train_dataset, validation_dataset=validation_datatest, name='events')

    if args.do_eval:
        load_model(args, model, 'events')
        optimizer = config['optimizer'](model.parameters(), lr=config['lr'])
        criterion = nn.BCEWithLogitsLoss()
        device = args.device
        evaluate(args, model, validation_datatest, criterion, optimizer, device)

    if args.do_test:
        load_model(args, model, 'events')
        test(args, model, test_dataset, 'events')


    if args.hypertune:
        hypertune(train_dataset, validation_datatest, xshape1, xshape2, args)


def hypertune(train_dataset, validation_datatest, xshape1, xshape2, args):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l4": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.choice([0.1, 0.01, 0.001, 0.0001, 0.00001]),
        "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        "batch_size": tune.choice([32, 64, 128, 256, 512, 1024]),
        "optimizer": tune.choice([optim.SGD, optim.Adam, optim.RMSprop, ])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=100,
        grace_period=100,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])

    tune_func = tune.with_parameters(train, xshape1=xshape1, xshape2=xshape2,
                                     args=args, train_dataset=train_dataset, validation_dataset=validation_datatest)

    analysis = tune.run(
        tune_func,
        resources_per_trial={
            "cpu": 8,
            "gpu": 1,
        },
        stop={"training_iteration": 100},
        config=config,
        max_failures=10,
        num_samples=400,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    print(analysis)
    analysis.results_df.to_csv("results.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do-train", action="store_false",
                        help="Whether to run training.")
    parser.add_argument("--hypertune", action="store_true",
                        help="Whether to run hyperparameter tuning.")
    parser.add_argument("--do-eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do-test", action="store_true",
                        help="Whether to run test on the test set.")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--backs", default=10, type=int,
                        help="Number of back commits to use for training.")
    parser.add_argument('--cache', action='store_true', help="cache old data")
    parser.add_argument("--output_dir", default="saved_models", type=str)
                        

    return parser.parse_args()


if __name__ == "__main__":
    main()
