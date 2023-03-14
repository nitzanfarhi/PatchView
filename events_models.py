import torch
from torch import nn
import torch.nn.functional as F

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        return self.fc(out)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        return self.fc(out)


class LSTMClassification(nn.Module):

    def __init__(self, input_dim):
        super(LSTMClassification, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, 100, batch_first=True)
        self.lstm2 = nn.LSTM(100, 50, batch_first=True)
        self.lstm3 = nn.LSTM(50, 25, batch_first=True)
        self.lstm4 = nn.LSTM(25, 12, batch_first=True)
        self.fc = nn.Linear(12, 1)

    def forward(self, input_):
        lstm_out, (h, c) = self.lstm1(input_)
        lstm_out, (h, c) = self.lstm2(lstm_out)
        lstm_out, (h, c) = self.lstm3(lstm_out)
        lstm_out, (h, c) = self.lstm4(lstm_out)
        logits = self.fc(lstm_out[:,-1])
        return torch.tanh(logits)
    

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
    def __init__(self, xshape1, xshape2, l1 = 1024,l2 = 256, l3 = 256, l4 = 64, dropout = 0.5):
        super(Conv1DTune, self).__init__()
        self.xshape1 = xshape1
        self.xshape2 = xshape2

        self.conv = nn.Conv1d(xshape1, l2, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm1d(l2)
        self.batchnorm2 = nn.BatchNorm1d(l1)

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(l2 * ((self.xshape2 // 2) ), l1)
        self.fc2 = nn.Linear(l1, l3)
        self.fc3 = nn.Linear(l3, l4)
        self.fc4 = nn.Linear(l4, 1)
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
    