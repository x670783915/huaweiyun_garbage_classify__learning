import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import time
import os
from torchvision import datasets, transforms
from torch.optim import lr_scheduler

__all__ = ['train_model']

def train_model(model, dataloader, criterion, optimizer, lr_scheduler=None, device=None, epochs=20, save_name='default'):

    # -------- create save  enc ---------
    model_path_dir = './models'
    save_name = (save_name + '_' + str(time.time()) + '.pth') if save_name == 'default' else save_name + '.pth'
    if os.path.exists(model_path_dir):
        print('Model will save in {} with name {}'.format(model_path_dir, save_name))
        torch.save(model.state_dict(), os.path.join(model_path_dir, save_name))
    else:
        os.makedirs(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, save_name))
    # -----------------------------------

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    train_start_time = time.time()
    # -------- train --------
    model.train()
    for epoch in range(epochs):
        tot_batchs = len(dataloader)
        tot_nums = 0
        running_corrects = 0
        running_loss = 0.0
        since = time.time()
        for batch_idx, data in enumerate(dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * inputs.size(0)
            tot_nums += inputs.size(0)
            loss.backward()
            optimizer.step()
            # print('{} / {} in this epoch'.format(batch_idx, tot_batchs))

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        print('-' * 20)
        print('Train Epoch {}: Loss = {:.4}, Acc = {:.4f}, {}s'.format(
            epoch, 
            running_loss / tot_nums, 
            running_corrects.double() / tot_nums, 
            time.time() - since))
        print('-' * 20)
    # -------- end for train --------

    spend_time = time.time() - train_start_time
    torch.save(model.state_dict(), os.path.join(model_path_dir, save_name))

    print('Train Model spend {}m:{}s'.format(spend_time // 60, spend_time % 60))
