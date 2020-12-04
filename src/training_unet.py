import torch
import numpy as np
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(prediction, label):
    #TODO: to be implemented
    return 0.0 

def dice_coef(prediction, target, smooth = 1, class_weights = [0.5, 0.5]):
    coef = 0.
    for c in range(target.shape[1]):
           pflat = prediction[:, c].contiguous().view(-1)
           tflat = target[:, c].contiguous().view(-1)
           intersection = (pflat * tflat).sum()
           
           w = class_weights[c]
           coef += w*((2. * intersection + smooth) /
                             (pflat.sum() + tflat.sum() + smooth))
    return coef

def dice_loss(prediction, target, smooth = 1, class_weights = [0.5, 0.5]):
    return 1 - dice_coef(prediction, target, smooth, class_weights)

def train(model, dataset_train, dataset_test, optimizer, num_epochs):
    """
    Train the given model
    
    @param model         : torch.nn.Module
    @param criterion     : torch.nn.modules.loss._Loss
    @param dataset_train : torch.utils.data.DataLoader
    @param dataset_test  : torch.utils.data.DataLoader
    @param optimizer     : torch.optim.Optimizer
    @param num_epochs    : int
    """
    print("Starting training")
    model.to(device)
    for epoch in range(num_epochs):
        # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            batch_pred = model(batch_x)
            loss = dice_loss(batch_pred, batch_y)

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Test the quality on the test set
        model.eval()
        accuracies_test = []
        f1_scores_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(accuracy(prediction, batch_y))
            f1_scores_test.append(dice_coef(prediction, batch_y).item())

        print(f"Epoch {epoch + 1 : 2} | Test accuracy : {np.mean(accuracies_test):.5} | Test F1 : {np.mean(f1_scores_test):.5}")

