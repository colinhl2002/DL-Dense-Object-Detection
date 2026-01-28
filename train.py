import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm

def setup_training(model, learning_rate, momentum):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4
    )

    return (criterion, optimizer)


def train_one_epoch(model, criterion, optimizer, train_loader, DEVICE):
    epoch_loss = []

    for i, batch in enumerate(tqdm(train_loader)):
        # get one batch
        x, y_true = batch
        x = x.to(DEVICE)
        y_true = y_true.to(DEVICE)

        # format the y_true so that it is compatible with the loss
        y_true = y_true.squeeze().long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        y_pred = model(x)

        # compute loss
        loss = criterion(y_pred, y_true)

        # backward
        loss.backward()

        # update parameters
        optimizer.step()

        # save statistics
        epoch_loss.append(loss.item())

        if i % 10 == 0:
            print(f"Batch {i}, curr loss = {loss.item():.03f}")

    return np.asarray(epoch_loss).mean()