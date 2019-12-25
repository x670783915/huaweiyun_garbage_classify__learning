import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from .misc import *   

def select_n_random(data, labels, n=100):
    assert len(data) == len(labels)

    perm = torch.randomperm(len(data))
    return data[perm][:n], labels[perm][:n]

def imshow_in_project(data, labels, writer, classes, img_size, n=100):

    assert writer is not None
    assert classes is not None

    imges, labels = select_n_random(data, labels)

    class_labels = [classes[lab] for lab in labels]

    features = image.view(-1, img_size[0] * img_size[1])
    writer.add_embedding(features, metadata=class_labels, label_img=imges.unsqueeze(1))
    writer.close()
    
def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, le in zip(preds, output)]

def matplot_imshow(img, mean, std, one_chennel=False):
    if one_chennel:
        img = img.mean(dim=0)
    img = img * std + mean
    npimg = img.numpy()
    if one_chennel:
        plt.imshow(npimg, cmap='Greys')
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(net, images, labels, classes, mean, std, nums=4):
    preds, probs = images_to_probs(net, images)

    fig = plt.figure(figsize=(12, 48))
    for idx in range(nums):
        ax = fig.add_subplot(1, nums, idx + 1, xticks=[], yticks=[])
        matplot_imshow(images[idx], mean, std, one_chennel=True)
        ax.set_title('{0}, {1:1f}%\n(label: {2})'.format(
            classes[preds[idx]], 
            probs[idx] * 100.0, 
            classes[labels[idx]]
        ), color=('green' if preds[idx] == labels[idx].item() else 'red'))
    return fig

def add_pr_curve_tensorboard(writer, classes, class_index, test_probs, test_preds, global_step=0):
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]
    writer.add_pr_curve(
        classes[class_index],
        tensorboard_preds,
        tensorboard_probs,
        global_step=global_step
    )
    writer.close()


