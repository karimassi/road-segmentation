import torch

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

def train(model, criterion, dataset_train, dataset_test, optimizer, num_epochs):
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
            loss = criterion(batch_pred, batch_y.float())

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
            f1_scores_test.append(F1_score(prediction, batch_y))

        print(f"Epoch {epoch + 1 : 2} | Test accuracy : {np.mean(accuracies_test):.5} | Test F1 : {np.mean(f1_scores_test):.5}")