import torch
import numpy as np
import time

from metrics import accuracy_unet, dice_coeficient, iou_score 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, criterion, dataset_train, dataset_test, optimizer, scheduler=None, num_epochs=10):
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
            f1_scores_test.append(dice_coeficient(prediction, batch_y))
            iou_scores_test.append(iou_score(prediction, batch_y))
            
        if scheduler is not None :
            scheduler.step()
            
        accuracies_test_mean = np.mean(accuracies_test)
        global_accuracies_test.append(accuracies_test_mean)
        f1_scores_test_mean = np.mean(f1_scores_test)
        global_f1_scores_test.append(f1_scores_test_mean)
        iou_scores_test_mean = np.mean(iou_scores_test)
        global_iou_scores_test.append(iou_scores_test_mean)
        print(f"Epoch {epoch + 1 : 2} | Training loss : {epoch_loss:.5} | Test accuracy : {accuracies_test_mean:.5} | Test F1 : {f1_scores_test_mean:.5} | Test IoU : {iou_scores_test_mean:.5} | In {time.time() - begin:.5} s")
        
    return global_accuracies_test, global_f1_scores_test, global_iou_scores_test
