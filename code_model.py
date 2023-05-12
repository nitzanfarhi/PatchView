# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


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
        if labels is None:
            return prob
        labels = labels.float()
        loss = torch.log(prob[:, 0]+1e-10)*labels + \
            torch.log((1-prob)[:, 0]+1e-10)*(1-labels)
        loss = -loss.mean()
        return loss, prob


class Model2(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model2, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.fc1 = nn.Linear(config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 1)
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids=None, labels=None):
        x = self.encoder(input_ids=input_ids, attention_mask=input_ids.ne(
            1), output_hidden_states=True)[0]
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class DistillBERTClass(torch.nn.Module):
    def __init__(self, encoder, config):
        super(DistillBERTClass, self).__init__()
        self.l1 = encoder
        self.pre_classifier = torch.nn.Linear(
            config.hidden_size, config.hidden_size)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = torch.sigmoid(output)
        return output


def calcuate_accu(big_idx, targets):
    return (big_idx == targets).sum().item()


def beep_train(model, training_loader, args):
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
    for epoch in range(100):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for _, data in enumerate(training_loader, 0):
            ids, targets = data
            ids = ids.to(args.device, dtype=torch.long)
            ids = ids.unsqueeze(0)
            targets = targets.to(args.device, dtype=torch.long)

            outputs = model(ids, ids.ne(1))
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Training Loss per 5000 steps: {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

        print(
            f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")

    return
