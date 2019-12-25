import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import time

from dataset import GarbageDataset
from utils import Bar, Logger, AverageMeter, accuracy, savefig, get_optimizer, save_checkpoint

def train(train_loader, model, criterion, optimizer, epoch, device=None, writer=None):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    if device is None:
        device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs, targets = torch.autograd.Variable(inputs.to(device)), torch.autograd.Variable(targets.to(device))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        print(bar.suffix)
        if writer:
            writer.add_scalar('trainning loss', losses.avg, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('top1 acc', top1.avg, epoch * len(train_loader) + batch_idx)
            writer.add_scalar('top5 acc', top5.avg, epoch * len(train_loader) + batch_idx)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def test(val_loader, model, criterion, epoch, device=None, writer=None):
    
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    if device is None:
        device = torch.device('cude:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)

        inputs, targets = torch.autograd.Variable(inputs.to(device)), torch.autograd.Variable(targets.to(device))

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        print(bar.suffix)
        if writer:
            writer.add_scalar('trainning loss', losses.avg, epoch * len(val_loader) + batch_idx)
            writer.add_scalar('top1 acc', top1.avg, epoch * len(val_loader) + batch_idx)
            writer.add_scalar('top5 acc', top5.avg, epoch * len(val_loader) + batch_idx)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def test_a(test_loader, model, device=None, writer=None):
    class_probs = []
    class_preds = []

    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            class_probs_batch = [F.softmax(el, dim=0) for el in outputs]
            _, class_preds_batch = torch.max(outputs, 1)
            class_preds.append(class_preds_batch)
            class_probs.append(class_probs_batch)

            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
    
    return class_probs, class_preds, top1.avg, top5.avg
            