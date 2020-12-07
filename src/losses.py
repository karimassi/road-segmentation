from torch import nn

"""
    A set of classes that represent loss functions to be 
    used with a neural network
    
    Most methods borrowed from:
        https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
"""

class IoULoss(nn.Module):
    """
        Intersection over Union loss, or Jaccard Index
        commonly used for evaluating the performance of pixel segmentation models.
    """
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU
    
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
        
class Dice(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, smooth = 1, class_weights = [0.5, 0.5]):
      dice = 0.
      for c in range(target.shape[1]):
            pflat = prediction[:, c].contiguous().view(-1)
            tflat = target[:, c].contiguous().view(-1)
            intersection = (pflat * tflat).sum()
            
            w = class_weights[c]
            dice += w*((2. * intersection + smooth) /
                              (pflat.sum() + tflat.sum() + smooth))
      return 1 - dice

ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    """
        Loss function that penalises false positives and false negatives 
        by alpha and beta respectively
    """
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA): 
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
