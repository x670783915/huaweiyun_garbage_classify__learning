# Huaweiyun Garbage Classify Learning
 The topic was from huawei cloud garbage classification competition. 
 To learn how to use pytorch and test the effect of the backbone(Resnet, ResNext, Se_ResNext, etc).
 

 ## Dataset
 In order to better experience the learning process, I also extended the dataset by downloading pictures of corresponding categories at Google and Baidu.

### raw dataset
![raw_dataset_cnt](./images/raw_dataset_cnt.png)


## Models

### requirements
- pytorch 1.3.1 or above
- numpy
- PIL
- [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)



## Results

Model | Train Acc | Val Acc
-|-|-
se_resnext101_32x4d + ce | 0.9825 | 0.9005
se_resnext101_32x4d + fc | 0.9920 | 0.8986
se_resnext101_32x4d + cbam + fc | 0.9993 | 0.9032



