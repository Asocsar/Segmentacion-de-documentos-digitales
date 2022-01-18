import torch.nn as nn
import torchvision.models as tvmodels
import torch
import re
import os

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        self.model = tvmodels.vgg16(pretrained=False)
        self.model.load_state_dict(torch.load(os.path.join(
                                                os.sep.join(os.getcwd().split(os.sep)[:-1]),
                                                'Weights',
                                                'vgg16.pth')))
        self.features = self.model.features

        '''for name, parameter in self.features.named_parameters():
            parameter.requires_grad = False
            match = re.match(r"^(\d+).*$", name)
            if match:
                name_start = int(match[1])
                if name_start >= 28:
                    parameter.requires_grad = True'''
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512*6*6,512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.LeakyReLU(), 
            nn.Linear(256, n_classes),
            nn.LeakyReLU()
        )

        self._fc = nn.Sigmoid()

    def forward(self, prevPage, targPage):
        x1 = self.features(prevPage)
        x1 = self.classifier(x1)
        x2 = self.features(targPage)
        x2 = self.classifier(x2)        
        x = torch.cat((x1, x2), dim=1)
        x = self.final(x)
        x = self._fc(x)
        
        return x

class VGG_ThreePages(nn.Module):
    def __init__(self, n_classes):
        super(VGG_ThreePages, self).__init__()
        self.model = tvmodels.vgg16(pretrained=False)
        self.model.load_state_dict(torch.load(os.path.join(
                                                os.sep.join(os.getcwd().split(os.sep)[:-1]),
                                                'Weights',
                                                'vgg16.pth')))
        self.features = self.model.features

        '''for ch in self.features.children():
            layer_count = 0
            for layer in ch.children():
                if layer_count<=23:
                    for param in layer.parameters():
                        param.requires_grad = False
                layer_count += 1            
            break'''
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512*6*6,512),
            nn.LeakyReLU(),
            nn.Dropout(0.5)
        )
        
        self.final = nn.Sequential(
            nn.Linear(512*3, 256),
            nn.LeakyReLU(), 
            nn.Linear(256, n_classes),
            nn.LeakyReLU(),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )

    def forward(self, prev_page, targ_page, next_page):
        x1 = self.features(prev_page)
        x1 = self.classifier(x1)
        x2 = self.features(targ_page)
        x2 = self.classifier(x2)        
        x3 = self.features(next_page)
        x3 = self.classifier(x3)        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.final(x)
        return x