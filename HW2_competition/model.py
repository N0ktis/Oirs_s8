import pickle as pk
import torch
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from torch import nn
import datetime as dt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim

        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=embedding_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))

        x, (_, _) = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        return h_n.reshape((self.n_features, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = 2 * input_dim
        self.n_features = n_features

        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)

        self.output = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.input_dim))

        x, (h_n, c_n) = self.lstm1(x)
        x, (h_n, c_n) = self.lstm2(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        return self.output(x)


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(DEVICE)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(DEVICE)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_model(model, train_dataset, check_dataset, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)

    best_model_w = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(epochs):
        model = model.train()
        train_loss_list = list()
        for train_d in train_dataset:
            optimizer.zero_grad()
            train_d = train_d.to(DEVICE)
            pred_d = model(train_d)
            loss = criterion(pred_d, train_d)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())

        model = model.eval()
        val_loss_list = list()
        with torch.no_grad():
            for check_d in check_dataset:
                check_d = check_d.to(DEVICE)
                seq_pred = model(check_d)
                loss = criterion(seq_pred, check_d)
                val_loss_list.append(loss.item())
        train_loss = np.mean(train_loss_list)
        val_loss = np.mean(val_loss_list)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_w = copy.deepcopy(model.state_dict())
        print('Epoch {epoch}: train loss {train_loss} val loss {val_loss}'.format(epoch=epoch, train_loss=train_loss,
                                                                                  val_loss=val_loss))
    model.load_state_dict(best_model_w)
    return model.eval()


def predict(model, dataset):
    predictions = list()
    loss_list = list()
    criterion = nn.L1Loss(reduction='sum').to(DEVICE)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(DEVICE)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            loss_list.append(loss.item())
    return predictions, loss_list
