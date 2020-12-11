import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy(prediction, label):
    """
    Compute the accuracy of the prediction
    
    @param prediction : the prediction of the model, int64 tensor of shape (batch_size), either 0 or 1
    @param label      : the labels of the data     , int64 tensor of shape (batch_size), either 0 or 1
    """
    
    batch_size = label.size(0)
    correct = torch.sum(prediction == label)
    return (correct / batch_size).cpu()
    
def accuracy_unet(prediction, mask):
    correct = torch.sum(torch.round(prediction).int() == mask.int())
    return (correct / (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])).item()

def F1_score(prediction, label):
    """
    Compute the F1-score of the prediction
    
    @param prediction : the prediction of the model, int64 tensor of shape (batch_size), either 0 or 1
    @param label      : the labels of the data     , int64 tensor of shape (batch_size), either 0 or 1
    """
    
    batch_size = label.size(0)
    
    precision = (torch.sum(prediction * label) / torch.sum(prediction))
    recall = (torch.sum(prediction * label) / torch.sum(label))
    
    F1 = 2 * precision * recall / (precision + recall)
    return F1.cpu().item()
    
def dice_coef(prediction, target, smooth = 1, class_weights = [0.5, 0.5]):
    coef = 0.
    for c in range(target.shape[1]):
           pflat = prediction[:, c].contiguous().view(-1)
           tflat = target[:, c].contiguous().view(-1)
           intersection = (pflat * tflat).sum()
           
           w = class_weights[c]
           coef += w*((2. * intersection + smooth) /
                             (pflat.sum() + tflat.sum() + smooth))
    return coef.item()

def iou(prediction, target, smooth = 1e-6):
    
    # take only predictions of road channel
    outputs = prediction[:, 0] > 0.5
    # gt of road channel
    labels = target[:, 0] > 0.5
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds
    
    return thresholded.mean().item()

def train(model, criterion, dataset_train, dataset_test, optimizer, scheduler, num_epochs):
    """
    Train the given model
    
    @param model         : torch.nn.Module
    @param criterion     : torch.nn.modules.loss._Loss
    @param dataset_train : torch.utils.data.DataLoader
    @param dataset_test  : torch.utils.data.DataLoader
    @param optimizer     : torch.optim.Optimizer
    @param scheduler     : torch.optim.lr_scheduler
    @param num_epochs    : int
    """
    print("Starting training")
    model.to(device)
    global_accuracies_test = []
    global_f1_scores_test = []
    global_iou_scores_test = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        begin = time.time()
        # Train an epoch
        model.train()
        for batch_x, batch_y in dataset_train:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            batch_pred = model(batch_x)
            loss = criterion(batch_pred, batch_y.float())
            epoch_loss += loss

            # Compute the gradient
            optimizer.zero_grad()
            loss.backward()

            # Update the parameters of the model with a gradient step
            optimizer.step()

        # Test the quality on the test set
        model.eval()
        accuracies_test = []
        f1_scores_test = []
        iou_scores_test = []
        for batch_x, batch_y in dataset_test:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Evaluate the network (forward pass)
            prediction = model(batch_x)
            accuracies_test.append(accuracy_unet(prediction, batch_y))
            f1_scores_test.append(dice_coef(prediction, batch_y))
            iou_scores_test.append(iou(prediction, batch_y))
        scheduler.step()
        accuracies_test_mean = np.mean(accuracies_test)
        global_accuracies_test.append(accuracies_test_mean)
        f1_scores_test_mean = np.mean(f1_scores_test)
        global_f1_scores_test.append(f1_scores_test_mean)
        iou_scores_test_mean = np.mean(iou_scores_test)
        global_iou_scores_test.append(iou_scores_test_mean)
        print(f"Epoch {epoch + 1 : 2} | Training loss : {epoch_loss:.5} | Test accuracy : {accuracies_test_mean:.5} | Test F1 : {f1_scores_test_mean:.5} | Test IoU : {iou_scores_test_mean:.5} | In {time.time() - begin:.1} s")
    return global_accuracies_test, global_f1_scores_test, global_iou_scores_test
