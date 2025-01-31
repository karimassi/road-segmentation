import torch
from torch import nn

"""
    Classes and functions used as performance metrics and loss functions. 
"""

def accuracy_cnn(prediction, label):
    """
        Computes the accuracy of the prediction

        @param prediction : the prediction of the model, int64 tensor of shape (batch_size), either 0 or 1
        @param label      : the labels of the data     , int64 tensor of shape (batch_size), either 0 or 1
    """
    
    batch_size = label.size(0)
    correct = torch.sum(prediction == label)
    return (correct / batch_size).cpu()
    
def accuracy_unet(prediction, mask):
    """
        Computes the accuracy of the prediction
    """
    correct = torch.sum(torch.round(prediction).int() == mask.int())
    return (correct / (mask.shape[0] * mask.shape[1] * mask.shape[2] * mask.shape[3])).item()

def F1_score(prediction, label):
    """
        Computes the F1-score of the prediction

        @param prediction : the prediction of the model, int64 tensor of shape (batch_size), either 0 or 1
        @param label      : the labels of the data     , int64 tensor of shape (batch_size), either 0 or 1
    """
    
    batch_size = label.size(0)
    
    precision = (torch.sum(prediction * label) / torch.sum(prediction))
    recall = (torch.sum(prediction * label) / torch.sum(label))
    
    F1 = 2 * precision * recall / (precision + recall)
    return F1.cpu().item()
    
def dice_coeficient(prediction, target, smooth = 1, class_weights = [0.5, 0.5]):
    """
        Computes the dice coefficient of the prediction (interchangeable with 
        F1-score)
    """
    coef = 0.
    for c in range(target.shape[1]):
        pflat = prediction[:, c].contiguous().view(-1)
        tflat = target[:, c].contiguous().view(-1)
        intersection = (pflat * tflat).sum()

        w = class_weights[c]
        coef += w*((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))
    return coef.item()

def iou_score(prediction, target, smooth = 1e-6):
    """
        Computes the IoU score (or Jaccard index) of the prediction
    """
    
    # take only predictions of road channel
    outputs = prediction[:, 0] > 0.5
    # gt of road channel
    labels = target[:, 0] > 0.5
    
    intersection = (outputs * labels).float().sum()  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum()         # Will be zero if both are 0
    
    iou = (intersection + smooth) / (union + smooth)  # We smooth our devision to avoid 0/0
    
    return iou.item() 
        
class DiceLoss(nn.Module):
    """
        Class representing a Dice Loss criterion to be 
        used as a loss function
    """
    def __init__(self, class_weights = [0.5, 0.5]):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, prediction, target, smooth = 1):
        dice = dice_coefficient(prediction, target, smooth, self.class_weights)
        return 1 - dice

