import datetime
import json
import logging
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


def accuracy(model,device, dataloader):
    CM=0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images) #file_name
            preds = torch.argmax(outputs.data, 1)
            CM+=confusion_matrix(labels.cpu(), preds.cpu(),labels=[0,1])
            
        tn=CM[0][0]
        tp=CM[1][1]
        fp=CM[0][1]
        fn=CM[1][0]
        acc=np.sum(np.diag(CM)/np.sum(CM))
        # sensitivity=tp/(tp+fn)
        # precision=tp/(tp+fp)
        
        # print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        # print()
        # print('Confusion Matirx : ')
        # print(CM)
        # print('- Sensitivity : ',(tp/(tp+fn))*100)
        # print('- Specificity : ',(tn/(tn+fp))*100)
        # print('- Precision: ',(tp/(tp+fp))*100)
        # print('- NPV: ',(tn/(tn+fn))*100)
        # print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        # print()
                
    return acc

def train(train_dataset, model, name, opt,  args, evaulate_loader=None):
    """
    Train the model
    """
    writer = SummaryWriter(f"log/{name}/train/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    criterion = torch.nn.BCEWithLogitsLoss()
    num_epochs = args.epochs
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=True)
    total_step = len(train_loader)

    model.to(args.device)
    model.zero_grad()
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    global_step = 0

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True)

    if evaulate_loader is not None:
        valid_loader = DataLoader(
            evaulate_loader, batch_size=args.train_batch_size, shuffle=True)

    writer = SummaryWriter(
        f"log/temporal/{datetime.datetime.now().strftime('%b%d_%H-%M-%S')}")
    writer.add_text("args", json.dumps(
        args.__dict__, default=lambda o: '<not serializable>'))

    loss_function = nn.BCEWithLogitsLoss()

    model.to(args.device)

    history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }


    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_acc = 0.0


    for epoch in range(args.epochs):
        losses = []
        model.train()

        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            loss, _ = model(inputs, labels)
            loss.backward()

            opt.step()
            opt.zero_grad()

            losses.append(float(loss))

        history['loss'].append(np.mean(losses))
        history['acc'].append(accuracy(model, args.device, train_loader))

        print(f"Epoch {epoch} - loss: {history['loss'][-1]} - acc: {history['acc'][-1]}")
        if evaulate_loader is not None:
            model.eval()
            valid_losses = []
            for data in valid_loader:
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                labels = labels.unsqueeze(1)
                tag_scores = model(inputs)
                loss = loss_function(tag_scores.float(), labels.float())
                valid_losses.append(float(loss))
            history['val_loss'].append(np.mean(valid_losses))
            history['val_acc'].append(accuracy(model, args.device, valid_loader))



            writer.add_scalars('Loss', {
                               "train": history['loss'][-1],
                               "validation": history['val_loss'][-1]},
                               epoch)

            writer.add_scalars('Accuracy', {
                               "train": history['acc'][-1],
                               "validation": history['val_acc'][-1]},
                               epoch)

    print(history)

    return history    