import torch
import torchmetrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import inspect
from tqdm import tqdm
import ordinal_losses

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_toy_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    labels = ['unacc', 'acc', 'good', 'vgood']
    df = pd.read_csv(url, header=None)
    X = df.drop(columns=df.columns[-1])
    X = pd.get_dummies(X).to_numpy(np.float32)
    X = (X-X.mean(0)) / X.std(0)  # z-normalization
    Y = df.iloc[:, -1]
    Y = np.array([labels.index(y) for y in Y], np.int64)
    return X, Y

def train_model(loss, X, Y):
    model = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, loss.how_many_outputs()),
    ).to(device)
    opt = torch.optim.Adam(model.parameters())
    model.train()
    ds = torch.utils.data.DataLoader(list(zip(X, Y)), 64, True)
    for _ in range(100):
        for x, y in ds:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            loss_value = loss(yhat, y)
            opt.zero_grad()
            loss_value.backward()
            opt.step()
    return model

def eval_model(model, metric, X, Y):
    model.eval()
    ds = torch.utils.data.DataLoader(list(zip(X, Y)), 64, True)
    for x, y in ds:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        _, yhat = loss.to_proba_and_classes(yhat)
        metric.update(yhat, y)
    return metric.compute()

if __name__ == '__main__':
    X, Y = load_toy_data()
    K = Y.max()+1
    nfolds = 10
    kfold = StratifiedKFold(nfolds, shuffle=True, random_state=123)
    metric = torchmetrics.MeanAbsoluteError().to(device)
    losses = inspect.getmembers(ordinal_losses, inspect.isclass)
    for name, loss in losses:
        loss = loss(K)
        avg_result = 0
        for train_ix, test_ix in tqdm(kfold.split(X, Y), leave=False, total=nfolds):
            model = train_model(loss, X[train_ix], Y[train_ix])
            result = eval_model(model, metric, X[test_ix], Y[test_ix])
            avg_result += float(result)/nfolds
        print('%-25s %f' % (name, avg_result))
