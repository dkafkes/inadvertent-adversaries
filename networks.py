import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# class DeepMerge(nn.Module):
#     def __init__(self, use_bottleneck=False, new_cls=False, class_num=3):
#         super(DeepMerge, self).__init__()
#         self.class_num = class_num
#         self.use_bottleneck = use_bottleneck
#         self.new_cls = new_cls
#         self.in_features = 32 * 12 * 12
#         self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.batchn1 = nn.BatchNorm2d(8)
#         self.batchn2 = nn.BatchNorm2d(16)
#         self.batchn3 = nn.BatchNorm2d(32)
#         #### cut here
#         self.fc1 = nn.Linear(32 * 12 * 12, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, class_num)
#         self.relu =  nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#     def forward(self, x):
#         x = self.maxpool(self.relu(self.batchn1(self.conv1(x))))
#         x = self.maxpool(self.relu(self.batchn2(self.conv2(x))))
#         x = self.maxpool(self.relu(self.batchn3(self.conv3(x))))
#         ### cut here, feed through dense layers
#         x = x.view(-1, 32 * 12 * 12) #vector here
#         y = F.relu(self.fc1(x))
#         y = F.relu(self.fc2(y))
#         y = self.fc3(y)
#         return x, y
        
#     def output_num(self):
#     	return self.in_features #(edited)
    
# #half network to move around embedding
# class EndDM(nn.Module):
#     def __init__(self, use_bottleneck=False, new_cls=False, class_num=3):
#         super(EndDM, self).__init__()
#         self.class_num = class_num
#         #### cut here to pass the embedding in
#         self.fc1 = nn.Linear(32 * 12 * 12, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, class_num)
#         self.relu =  nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#     def forward(self, x):
#         ### cut here, feed through dense layers
#         x = x.view(-1, 32 * 12 * 12) #vector here
#         y = F.relu(self.fc1(x))
#         y = F.relu(self.fc2(y))
#         y = self.fc3(y)
#         return y
        
#     def output_num(self):
#     	return self.in_features


class ResNetFc(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=3, class_num=3):
        super(ResNetFc, self).__init__()
        #model_resnet = resnet_dict[resnet_name](pretrained=False)
        model_resnet = models.resnet18(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.dropout4 = nn.Dropout(0.5)
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                             self.layer1, self.layer2, self.layer3, self.layer4, self.dropout4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features
    def forward(self, x):        
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
            y = self.fc(x)
        else:
            y = self.fc(x)
        return x, y
    
    def output_num(self):
        return self.__in_features
    
class EndResNetFc(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=3, class_num=3):
        super(EndResNetFc, self).__init__()
        model_resnet = models.resnet18(pretrained=False)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls

        if new_cls:
            if self.use_bottleneck:
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features
    
    def forward(self, x): 
        y = self.fc(x)
        return y


class DeepMergeV3(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=3, class_num=3):
        super(DeepMergeV3, self).__init__()
        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.in_features = 32 * 12 * 12
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(8)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)

        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Linear(32 * 12 * 12, bottleneck_dim)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.0)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.batchn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

class EndDMV3(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=3, class_num=3):
        super(EndDMV3, self).__init__()
        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        y = self.fc(x)
        return y
