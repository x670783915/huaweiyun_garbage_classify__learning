import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()

        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.modules._modules.items():
            if name in self.extracted_layers:
                if name is 'fc': 
                    x = x.view(x.size(0), -1)
                x = module(x)
                if name in self.extracted_layers:
                    outputs.append(name)
        return outputs

if __name__ == '__main__':
    extract_list = ['conv1', 'maxpool', 'layer1', 'avgpool', 'fc']
    img_path = './test.jpg'
    saved_path = './test.txt'
    resnet = models.resnet50(pretrained=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    img = Image.open(img_path)
    img = transforms(img)

    x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

    if use_gpu:
        x = x.cuda()
        resnet = resnet.cuda()

    extractor = FeatureExtractor(resnet, extract_list)
    print(extractor(x)[4]) # [0]:conv1  [1]:maxpool  [2]:layer1  [3]:avgpool
    
