from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import EfficientNet


import os


class Efficient_Bert_ThreePages_OneCall_V2(nn.Module):
    def __init__(self, name, num_features_efficient, feature_concatenation):
        super(Efficient_Bert_ThreePages_OneCall_V2, self).__init__()
        configuration = DistilBertConfig()
        self.bert = DistilBertModel(configuration)

        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'Weights', 'bert_weights.pth'))
        self.bert.load_state_dict(pretrained_weights, strict=False)

        self.efficient = EfficientNet.from_name(name)
        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]),  'Weights', name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)

        num_ftrs = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(num_ftrs, num_features_efficient)

        self.feature_fc = torch.nn.Linear(1536, feature_concatenation)
        self.output_fc = torch.nn.Linear(feature_concatenation, 1)



    def forward(self, image1, image2, image3, tokens, attention_mask):

        output_bert = self.bert(input_ids=tokens, attention_mask=attention_mask).last_hidden_state[:,0,:]
        output_efficient1 = self.efficient(image1)
        output_efficient2 = self.efficient(image2)
        output_efficient3 = self.efficient(image3)
        concatenation = torch.cat((output_bert, 
                                    output_efficient1, output_efficient2, output_efficient3), 1)
        
        

        output =  self.feature_fc(concatenation)
        output = self.output_fc(output)
        return torch.sigmoid(output)



class Efficient_Bert_ThreePages_TwoCall_V2(nn.Module):
    def __init__(self, name, num_features_efficient, feature_concatenation):
        super(Efficient_Bert_ThreePages_TwoCall_V2, self).__init__()
        
        configuration = DistilBertConfig()
        self.bert = DistilBertModel(configuration)

        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'Weights', 'bert_weights.pth'))
        self.bert.load_state_dict(pretrained_weights, strict=False)

        self.efficient = EfficientNet.from_name(name)
        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]),  'Weights', name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)


        num_ftrs = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(num_ftrs, num_features_efficient)

        self.feature_fc = torch.nn.Linear(2304, feature_concatenation)
        self.output_fc = torch.nn.Linear(feature_concatenation, 1)





    def forward(self, image1, image2, image3, tokens1, tokens2, attention_mask1, attention_mask2):
        output_bert1 = self.bert(input_ids=tokens1, attention_mask=attention_mask1).last_hidden_state[:,0,:]
        output_bert2 = self.bert(input_ids=tokens2, attention_mask=attention_mask2).last_hidden_state[:,0,:]
        output_efficient1 = self.efficient(image1)
        output_efficient2 = self.efficient(image2)
        output_efficient3 = self.efficient(image3)
        concatenation = torch.cat((output_bert1, output_bert2, 
                                output_efficient1, output_efficient2, output_efficient3), 1)

        output =  self.feature_fc(concatenation)
        output = self.output_fc(output)
        return torch.sigmoid(output)


class Efficient_Bert_ThreePages_ThreeCall_V2(nn.Module):
    def __init__(self, name, num_features_efficient, feature_concatenation):
        super(Efficient_Bert_ThreePages_ThreeCall_V2, self).__init__()
        
        configuration = DistilBertConfig()
        self.bert = DistilBertModel(configuration)

        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]), 'Weights', 'bert_weights.pth'))
        self.bert.load_state_dict(pretrained_weights, strict=False)

        self.efficient = EfficientNet.from_name(name)
        pretrained_weights = torch.load(os.path.join(os.sep.join(os.getcwd().split(os.sep)[:-1]),  'Weights', name + '.pth'))
        pretrained_weights.pop('_fc.weight')
        pretrained_weights.pop('_fc.bias')
        self.efficient.load_state_dict(pretrained_weights, strict=False)


        num_ftrs = self.efficient._fc.in_features
        self.efficient._fc = nn.Linear(num_ftrs, num_features_efficient)

        self.feature_fc = torch.nn.Linear(3072, feature_concatenation)
        self.output_fc = torch.nn.Linear(feature_concatenation, 1)





    def forward(self, image1, image2, image3, tokens1, tokens2, tokens3, attention_mask1, attention_mask2, attention_mask3):
        output_bert1 = self.bert(input_ids=tokens1, attention_mask=attention_mask1).last_hidden_state[:,0,:]
        output_bert2 = self.bert(input_ids=tokens2, attention_mask=attention_mask2).last_hidden_state[:,0,:]
        output_bert3 = self.bert(input_ids=tokens3, attention_mask=attention_mask3).last_hidden_state[:,0,:]
        output_efficient1 = self.efficient(image1)
        output_efficient2 = self.efficient(image2)
        output_efficient3 = self.efficient(image3)
        concatenation = torch.cat((output_bert1, output_bert2, output_bert3, 
                                    output_efficient1, output_efficient2, output_efficient3), 1)

        output =  self.feature_fc(concatenation)
        output = self.output_fc(output)
        return torch.sigmoid(output)

