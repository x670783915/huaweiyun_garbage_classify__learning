import os
import torch
import time
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from dataset import Resize, GarbageDataset

img_size = 224

means = [0.20903548, 0.21178319, 0.21442725]
stds = [0.12113936, 0.12205944, 0.12315971]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transforms = transforms.Compose([
    Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

dataset = GarbageDataset(
            './train_data_v2', 
            './test.csv', 
            data_transforms)
# '../../input/garbage_dataset/garbage_classify_v2/train_data_v2/' 服务器上
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


num_classes = 40
model_ft = models.resnet50(pretrained=True)

for param in model_ft.parameters():
    param.requires_grad = False

# print(model_ft.fc)
num_fc_ftr = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_fc_ftr, num_classes)
model_ft.to(device)


# f_list = []
# def hook(module, input, output):
#     for i in range(input[0].size(0)):
#         f_list.append(input[i][0].cpu().numpy())
# model_ft.avgpool.register_forward_hook(hook) 
# 这一段是为特征层添加hook将网络输出的特征保存下来

# print('-' * 20)
# print(model_ft)

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    tot_batchs = len(train_loader)
    corrects = 0
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        corrects += torch.sum(preds == labels.data)
        total_loss += loss.item() * inputs.size(0)
        loss.backward()
        optimizer.step()
        print('{} / {} in this epoch'.format(batch_idx, tot_batchs))
    return total_loss, corrects.double()


for epoch in range(1, 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([ {'params': model_ft.fc.parameters() }], lr=1e-3)
    since = time.time()
    print('{} Epoch start\n'.format(epoch), '-' * 20)
    loss, corrects = train(model_ft, device, dataloader, criterion, optimizer)
    print('Train Epoch {}: Loss = {:.4}, Acc = {:.4f}, {}s'.format(epoch, loss / len(dataset), corrects / len(dataset), time.time() - since))
    
torch.save(model_ft.state_dict(), './models/resnet_ft.pth')
