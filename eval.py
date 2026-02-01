import torch
import numpy as np
from tqdm import tqdm


def predict_one_epoch(model, test_loader, DEVICE):

    y_true = []
    y_pred = []

    # Send model to correct device
    model.to(DEVICE)

    # Put model in evaluation mode (very important)
    model.eval()

    # Disable all gradients things
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            # get one batch
            images, labels = batch
            images = images.to(DEVICE).float()
            labels = labels.to(DEVICE)

            y = model.forward(images)

            predicted = (y.squeeze() > 0).float()
            
            predicted = predicted.to("cpu").numpy()
            labels = labels.to("cpu").numpy()

            y_pred.append(predicted)
            y_true.append(labels)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    return(y_true, y_pred)