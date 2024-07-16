import os
import random
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as tr

from model import MLPNet, ConvNet
from training import model_train, model_evaluate

def seed_everything(seed = 21):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Loading Data - MNIST dataset
def make_loader(batch_size, train=True, shuffle=True):
    full_dataset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=train,
                                              download=True,
                                              transform=tr.ToTensor())
    
    loader = DataLoader(dataset=full_dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True)

    return loader  

def map_dict_to_str(config):
    config_str = ', '.join(f"{key}: {value}" for key, value in config.items() if key not in ['dataset', 'epochs', 'batch_size'])
    return config_str

def run(config, project_name = 'wandb_tutorial_test'):
    wandb.init(project=project_name, config=config)
    wandb.run.name = map_dict_to_str(config)

    print('------')
    print(map_dict_to_str(config))
    print('------\n')

    config = wandb.config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")

    train_loader = make_loader(batch_size=config.batch_size, train=True)
    test_loader = make_loader(batch_size=config.batch_size, train=False)
    
    if config.model == 'CNN':
        model = ConvNet().to(device)
    if config.model == 'MLP':
        model = MLPNet().to(device)

    criterion = nn.CrossEntropyLoss()

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    wandb.watch(model, criterion, log="all")

    max_loss = np.inf

    for epoch in range(0, config.epochs):
        train_loss, train_acc = model_train(model, train_loader, criterion, optimizer, device, None)
        val_loss, val_acc, sample_batch, sample_label, sample_prediction = model_evaluate(model, test_loader, criterion, device)

        wandb.log({"Train Loss": train_loss}, step=epoch+1)
        wandb.log({"Train Accuracy": train_acc}, step=epoch+1)
        wandb.log({"Validation Loss": val_loss}, step=epoch+1)
        wandb.log({"Validation Accuracy": val_acc}, step=epoch+1)

        wandb.log({"examples": [wandb.Image(image, caption=f"Pred: {pred}, Label: {label}") for image, pred, label in zip(sample_batch, sample_prediction, sample_label)]}, step=epoch+1)

        if val_loss < max_loss:
            print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
            max_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/Best_Model.pth')

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, acc: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f} \n')
      
    if config.model == 'CNN':
        model = ConvNet().to(device)
        wandb.log({'Total Params': 30762})
    if config.model == 'MLP':
        model = MLPNet().to(device)      
        wandb.log({'Total Params': 53018})
  
    model.load_state_dict(torch.load('Best_Model.pth', map_location=device))
    model.eval()
    val_loss, val_acc, _, _, _ = model_evaluate(model, test_loader, criterion, device)

    print('Test Loss: %s'%val_loss)
    print('Test Accuracy: %s'%val_acc)
    print()
    
    wandb.log({"Best Test Loss": val_loss})
    wandb.log({"Best Test Accuracy": val_acc})

    return 'Done'