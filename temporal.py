import argparse
import datetime
from functools import partial
import json
import logging
import os
import random
import numpy as np
import torch
from enum import Enum
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from SecurityPatchDetection.dataset_utils import extract_dataset, Aggregate, pad_and_fix, split_into_x_and_y, split_repos
from temporal_models import Conv1D, Conv1DTune, LSTMClassification, RNNModel
from torch.utils.tensorboard import SummaryWriter


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Tuner


class Set(Enum):
    train = 1
    validation = 2
    test = 3


SEED = 42

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
train_losses = []
valid_losses = []


class TemporalDataset(Dataset):
    def __init__(self, x_set, y_set, details=None, set: Set = Set.train):
        self.x_set = x_set
        self.y_set = y_set
        self.set = set
        self.details = details

    def __len__(self):
        return len(self.x_set)

    def __getitem__(self, idx):
        item = self.x_set[idx]
        label = self.y_set[idx]
        item = torch.from_numpy(item).float()
        return item, label


def accuracy(model, device, dataloader):
    model.eval()
    total_correct = 0
    total_instances = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            classifications = 1*(model(images).squeeze(1) > 0.5)
            correct_predictions = sum(classifications == labels).item()
            total_correct += correct_predictions
            total_instances += len(images)
    return total_correct / total_instances


def train( config, args=None, xshape1=0, xshape2=0, train_dataset=None, validation_dataset=None):

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)

    model = Conv1DTune(xshape1, xshape2,  l1 = config["l1"],l2 = config["l2"], l3 = config["l3"], l4 = config["l4"])
    # writer = SummaryWriter(
    #     f"log/temporal/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    # writer.add_text("args", json.dumps(
    #     args.__dict__, default=lambda o: '<not serializable>'))

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = config["optimizer"](model.parameters(), lr=config["lr"])
    
    model.cuda()

    history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(args.epochs):
        losses = []
        valid_losses = []
        model.train()

        for data in train_loader: # (pbar := tqdm(list(trainloader)[:], leave=False)):
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
            # pbar.set_description(f"Epoch {epoch} - loss: {np.mean(losses)}")

        avg_loss = np.mean(losses)

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

        history['loss'].append(avg_loss)
        history['val_loss'].append(avg_valid_loss)
        history['acc'].append(accuracy(model, args.device, train_loader))
        history['val_acc'].append(accuracy(model, args.device, valid_loader))
        # print(
        #     f'Epoch {epoch} - loss: {avg_loss} - val_loss: {avg_valid_loss} - acc: {history["acc"][-1]} - val_acc: {history["val_acc"][-1]}')

        # writer.add_scalars('Loss', {
        #                    "train": history['loss'][-1],
        #                    "validation": history['val_loss'][-1]},
        #                    epoch)

        # writer.add_scalars('Accuracy', {
        #                    "train": history['acc'][-1],
        #                    "validation": history['val_acc'][-1]},
        #                    epoch)

        # with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #     path = os.path.join(checkpoint_dir, "checkpoint")
        #     torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=max(history['val_loss']), accuracy=max(history["val_acc"]))

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
        # print(f'valid_loss {valid_loss}')


def load_model(args, model):
    checkpoint_prefix = f'checkpoint-best-acc-{args.language}/model.bin'
    output_dir = os.path.join(args.output_dir, f'{checkpoint_prefix}')
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)


def create_datasets(backs = 20, aggr_options = Aggregate.before_cve, metadata = True, cache = True, cache_location = None):
    all_repos, exp_name, columns = extract_dataset(
        aggr_options=Aggregate.before_cve,
        backs=backs,
        cache=True,
        metadata=True,
        cache_location=r"C:\Users\nitzan\local\analyzeCVE\ready_data")

    all_repos, num_of_vulns = pad_and_fix(all_repos)

    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.1

    train_size = int(TRAIN_SIZE * num_of_vulns)
    validation_size = int(VALIDATION_SIZE * num_of_vulns)
    test_size = num_of_vulns - train_size - validation_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {validation_size}")
    logger.info(f"Test size: {test_size}")

    train_and_val_repos, test_repos, _ = split_repos(
        all_repos, train_size + validation_size)
    train_repos, val_repos, num_of_train_repos = split_repos(
        train_and_val_repos, train_size)
    X_train, y_train = split_into_x_and_y(train_repos)
    X_val, y_val = split_into_x_and_y(val_repos)

    X_test, y_test, test_details = split_into_x_and_y(
        test_repos, with_details=True)

    train_dataset = TemporalDataset(X_train, y_train, set=Set.train)
    validation_dateset = TemporalDataset(X_val, y_val, set=Set.validation)
    test_dataset = TemporalDataset(X_test, y_test, set=Set.test)


    return train_dataset, validation_dateset, test_dataset, (X_train.shape[1], X_train.shape[2])


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

    args.start_epoch = 0
    args.start_step = 0

    train_dataset, validation_datatest, test_dataset, (xshape1, xshape2) = create_datasets(backs = args.backs)

    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    model = Conv1D(xshape1, xshape2)
    model = model.to(args.device)

    if args.do_train:
        config = {"l1": 1024, "l2": 256, "l3": 256,
                  "l4": 64, "lr": 0.0001, "dropout": 0.1, "batch_size": 32}

        train(config, xshape1=xshape1, xshape2=xshape2, model=model, args=args, train_dataset=train_dataset, validation_datatest=validation_datatest)

    if args.hypertune:

        hypertune(train_dataset,validation_datatest, xshape1, xshape2, args)


def hypertune(train_dataset,validation_datatest, xshape1, xshape2, args):
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l3": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l4": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.choice([0.1,0.01, 0.001, 0.0001, 0.00001]),
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
    parser.add_argument("--do-train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--hypertune", action="store_true",
                        help="Whether to run hyperparameter tuning.")
    parser.add_argument("--do-eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do-test", action="store_true",
                        help="Whether to run test on the test set.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--batch-size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--backs", default=10, type=int, help="Number of back commits to use for training.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
