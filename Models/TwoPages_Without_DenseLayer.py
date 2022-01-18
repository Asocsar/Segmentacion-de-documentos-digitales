from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import EfficientNet


import os


class Efficient_Bert_OneCall(nn.Module):
    def __init__(self, name):
        super(Efficient_Bert_OneCall, self).__init__()
        configuration = DistilBertConfig()
        self.bert = DistilBertModel(configuration)

        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'Weights', 'bert_weights.pth'))
        self.bert.load_state_dict(pretrained_weights, strict=False)

        self.efficient = EfficientNet.from_name(name)
        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]),  'Weights', name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)

        self._fc = torch.nn.Linear(656128, 1)



    def forward(self, image1, image2, tokens, attention_mask):
        output_bert = self.bert(input_ids=tokens, attention_mask=attention_mask).last_hidden_state[:,0,:]
        output_efficient1 = torch.flatten(self.efficient.extract_features(image1), start_dim=1)
        output_efficient2 = torch.flatten(self.efficient.extract_features(image2), start_dim=1)
        concatenation = torch.cat((output_bert, output_efficient1, output_efficient2), 1)
        output =  self._fc(concatenation)
        return torch.sigmoid(output)



class Efficient_Bert_TwoCall(nn.Module):
    def __init__(self, name):
        super(Efficient_Bert_TwoCall, self).__init__()
        
        configuration = DistilBertConfig()
        self.bert = DistilBertModel(configuration)

        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'Weights', 'bert_weights.pth'))
        self.bert.load_state_dict(pretrained_weights, strict=False)

        self.efficient = EfficientNet.from_name(name)
        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]),  'Weights', name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)


        self._fc = torch.nn.Linear(656896, 1)



    def forward(self, image1, image2, tokens1, tokens2, attention_mask1, attention_mask2):
        output_bert1 = self.bert(input_ids=tokens1, attention_mask=attention_mask1).last_hidden_state[:,0,:]
        output_bert2 = self.bert(input_ids=tokens2, attention_mask=attention_mask2).last_hidden_state[:,0,:]
        output_efficient1 = torch.flatten(self.efficient.extract_features(image1), start_dim=1)
        output_efficient2 = torch.flatten(self.efficient.extract_features(image2), start_dim=1)
        concatenation = torch.cat((output_bert1, output_bert2, output_efficient1, output_efficient2), 1)
        output =  self._fc(concatenation)
        return torch.sigmoid(output)
