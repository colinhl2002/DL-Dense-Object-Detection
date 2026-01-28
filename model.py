import torch
from torch import nn


def _init_weights(model):
    # about weight initialization
    # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    # https://www.pyimagesearch.com/2021/05/06/understanding-weight-initialization-for-neural-networks/
    for m in model.modules():
        # Initialize all convs
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def cnn_model(num_classes):
    model = nn.Sequential(
    # A first convolution block
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # Another stack of these
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    # A final classifier
    nn.Flatten(),
    nn.Linear(in_features=86528, out_features=64),
    nn.ReLU(),
    nn.Dropout(p=0.1),
    nn.Linear(in_features=64, out_features=num_classes),
    nn.Sigmoid(),
    )

    _init_weights(model)

    return model