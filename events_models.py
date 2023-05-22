import torch
from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, args, xshape1, xshape2):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(xshape2, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.lstm3 = nn.LSTM(50, 25, batch_first=True)
        self.lstm4 = nn.LSTM(25, 12, batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(12, 2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, labels):
        x, (h, c) = self.lstm1(x)
        x, (h, c) = self.lstm2(x)
        x, (h, c) = self.lstm3(x)
        x, (h, c) = self.lstm4(x)
        x = self.fc(x[:,-1])
        x = self.sigmoid(x)

        if labels is None:
            return x
        labels=labels.float()
        loss=torch.log(x[:,0]+1e-10)*labels+torch.log((1-x)[:,0]+1e-10)*(1-labels)
        loss=-loss.mean()

        return loss, x
    
class GRU(nn.Module):

    def __init__(self, args, xshape1, xshape2):
        super(LSTM, self).__init__()
        self.gru1 = nn.GRU(xshape2, 100, batch_first=True)
        self.gru2 = nn.GRU(100, 50, batch_first=True)
        self.gru3 = nn.GRU(50, 25, batch_first=True)
        self.gru4 = nn.GRU(25, 12, batch_first=True)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(12, 2)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, labels):
        x, (h, c) = self.gru1(x)
        x, (h, c) = self.gru2(x)
        x, (h, c) = self.gru3(x)
        x, (h, c) = self.gru4(x)
        x = self.fc(x[:,-1])
        x = self.sigmoid(x)

        if labels is None:
            return x
        labels=labels.float()
        loss=torch.log(x[:,0]+1e-10)*labels+torch.log((1-x)[:,0]+1e-10)*(1-labels)
        loss=-loss.mean()

        return loss, x
    

class Conv1D(nn.Module):
    def __init__(self, xshape1, xshape2):
        super(Conv1D, self).__init__()
        self.xshape1 = xshape1
        self.xshape2 = xshape2

        conv1d_shape = 256
        self.conv = nn.Conv1d(xshape1, conv1d_shape, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm1d(conv1d_shape)
        self.batchnorm2 = nn.BatchNorm1d(1024)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(conv1d_shape * ((self.xshape2 // 2) ), 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.conv(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.batchnorm2(x)

        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x
    



class Conv1DTune(nn.Module):
    def __init__(self, args, xshape1, xshape2, l1 = 1024,l2 = 256, l3 = 256, l4 = 64):
        super(Conv1DTune, self).__init__()
        self.args = args
        self.xshape1 = xshape1
        self.xshape2 = xshape2

        self.conv = nn.Conv1d(xshape1, l2, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm1d(l2)
        self.batchnorm2 = nn.BatchNorm1d(l1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc1 = nn.Linear(l2 * ((self.xshape2 // 2) ), l1)
        if not args.return_class:
            self.fc2 = nn.Linear(l1, args.hidden_size)
        else:
            self.fc2 = nn.Linear(l1, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.fc4 = nn.Linear(l4, 2)
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, labels=None):
        x = self.activation(self.conv(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.batchnorm2(x)

        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        if not self.args.return_class:
            # Returning for the multimodel to 
            return x
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))

        if labels is None:
            return x
        labels=labels.float()
        loss=torch.log(x[:,0]+1e-10)*labels+torch.log((1-x)[:,0]+1e-10)*(1-labels)
        loss=-loss.mean()

        return loss, x
    