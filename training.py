from tqdm import tqdm
import torch

def model_train(model, 
                data_loader, 
                criterion, 
                optimizer, 
                device, 
                scheduler=None, 
                tqdm_disable=False):
    """
    Model train (for multi-class classification)

    Args:
        model (torch model)
        data_loader (torch dataLoader)
        criterion (torch loss)
        optimizer (torch optimizer)
        device (str): 'cpu' / 'cuda' / 'mps'
        scheduler (torch scheduler, optional): lr scheduler. Defaults to None.
        tqdm_disable (bool, optional): if True, tqdm progress bars will be removed. Defaults to False.

    Returns:
        loss, accuracy: Avg loss, acc for 1 epoch
    """
    model.train()

    running_loss = 0
    correct = 0

    for X, y in tqdm(data_loader, disable=tqdm_disable):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # multi-class classification
        _, pred = output.max(dim=1)
        correct += pred.eq(y).sum().item()
        running_loss += loss.item() * X.size(0)

    if scheduler:
        scheduler.step()

    accuracy = correct / len(data_loader.dataset) # Avg acc
    loss = running_loss / len(data_loader.dataset) # Avg loss

    return loss, accuracy


def model_evaluate(model, 
                   data_loader, 
                   criterion, 
                   device):
    """
    Model validate (for multi-class classification)

    Args:
        model (torch model)
        data_loader (torch dataLoader)
        criterion (torch loss)
        device (str): 'cpu' / 'cuda' / 'mps'

    Returns:
        loss, accuracy: Avg loss, acc for 1 epoch
    """
    model.eval()

    with torch.no_grad():
        running_loss = 0
        correct = 0

        sample_batch = []
        sample_label = []
        sample_prediction = []

        for i, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)

            output = model(X)

            # multi-class classification
            _, pred = output.max(dim=1)
            correct += torch.sum(pred.eq(y)).item()
            running_loss += criterion(output, y).item() * X.size(0)

            if i == 0:
                sample_batch.append(X)
                sample_label.append(y)
                sample_prediction.append(pred)

        accuracy = correct / len(data_loader.dataset) # Avg acc
        loss = running_loss / len(data_loader.dataset) # Avg loss

        return loss, accuracy, sample_batch[0][:16], sample_label[0][:16], sample_prediction[0][:16]