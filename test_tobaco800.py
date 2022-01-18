#%%
from torch.utils.tensorboard import SummaryWriter

from utils.tools import obtain_folders
from train_validation_test import test_tobacco800

from Models.TwoPages_Without_DenseLayer import *
from Models.TwoPages_With_DenseLayer import *


from Models.ThreePages_Without_DenseLayer import *
from Models.ThreePages_With_DenseLayer import *

from Models.VGG16 import *

from Models.LayoutLM_v2 import LayoutLM_v2, LayoutLM_Three_v2

import argparse
import json
import os
import torch
import sys




def model_election(metadata):
    if "LayoutLM_v2" in metadata.keys() and metadata['LayoutLM_v2']:
        if metadata['num_pages'] == 2:
            model = LayoutLM_v2()
        elif metadata['num_pages'] == 3:
            model = LayoutLM_Three_v2()
        return model 

    if "VGG16" in metadata.keys() and metadata['VGG16']:
        if metadata['num_pages'] == 2:
            model = VGG16(metadata['num_clases'])
        elif metadata['num_pages'] == 3:
            model = VGG_ThreePages(metadata['num_clases'])
        return model 

    if metadata['num_pages'] == 2:
        if metadata['BertCalls'] == 1:
            if metadata['Version'] == 1:
                model = Efficient_Bert_OneCall(metadata['name_efficient'])
            else:
                model = Efficient_Bert_OneCall_V2(metadata['name_efficient'], metadata['num_features'], metadata['feature_concatenation'])
        else:
            if metadata['Version'] == 1:
                model = Efficient_Bert_TwoCall(metadata['name_efficient'])
            else:
                model = Efficient_Bert_TwoCall_V2(metadata['name_efficient'], metadata['num_features'], metadata['feature_concatenation'])

    elif metadata['num_pages'] == 3:
        if metadata['BertCalls'] == 1:
            if metadata['Version'] == 1:
                model = Efficient_Bert_ThreePages_OneCall(metadata['name_efficient'])
            else:
                model = Efficient_Bert_ThreePages_OneCall_V2(metadata['name_efficient'], metadata['num_features'], metadata['feature_concatenation'])
        elif metadata['BertCalls'] == 2:
            if metadata['Version'] == 1:
                model = Efficient_Bert_ThreePages_TwoCall(metadata['name_efficient'])
            else:
                model = Efficient_Bert_ThreePages_TwoCall_V2(metadata['name_efficient'], metadata['num_features'], metadata['feature_concatenation'])
        elif metadata['BertCalls'] == 3:
            if metadata['Version'] == 1:
                model = Efficient_Bert_ThreePages_ThreeCall(metadata['name_efficient'])
            else:
                model = Efficient_Bert_ThreePages_ThreeCall_V2(metadata['name_efficient'], metadata['num_features'], metadata['feature_concatenation'])
    
    return model


#%%
def main(model, metadata):

    model = model.cuda()


    tokenizer_file = open('./tokenizer_saved/tokenizer_config.json')
    tokenizer_config = json.load(tokenizer_file)
    tokenizer_file.close()
    metadata["TokenizerMaxLength"] = tokenizer_config["model_max_length"]


    TensorLog = SummaryWriter(metadata['TensorBoardFold'])
    with open(os.path.join(metadata['TensorBoardFold'], 'config.json'), 'w+') as out_conf:
        json.dump(metadata, out_conf)

    test_tobacco800(model, metadata, TensorLog)
#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tobacco800_conf", type=str,
                    help="Indicate file with configuration for Test")
    parser.add_argument("--select_epoch", type=int,
                    help="Indicate what epoch do the user wants to load", default=-1)
    parser.add_argument("--select_file", type=str,
                    help="Indicate what epoch do the user wants to load", default=None)
    parser.add_argument("--filtered", type=bool,
                    help="Indicate if user wants to load a filtered version from an epoch", default=False)
    parser.add_argument("--fine_tune", type=bool,
                    help="Indicate if user wants to train on a section of Tobacco800 dataset (test will be performed on an independent section)", default=False)
    parser.add_argument("--full_train", type=bool,
                    help="Indicate if user wants to train on a section of Tobacco800 dataset without loading any save weight(test will be performed on an independent section)", default=False)

    args = parser.parse_args()

    assert (not (args.select_epoch == '') or args.filtered), "If user wants to select filtered version, it has to select and epoch"


    conf_file = os.path.join("Main_conf_files", args.tobacco800_conf)
    f = open(conf_file)
    metadata = json.load(f)
    f.close()

    metadata['tobacco800_conf'] = args.tobacco800_conf
    metadata['select_epoch'] = args.select_epoch
    metadata['filtered'] = args.filtered
    metadata['fine_tune'] = args.fine_tune
    metadata['full_train'] = args.full_train
    metadata['select_file'] = args.select_file

    if metadata['full_train']:
        metadata['BATCH'] = 32


    model = model_election(metadata)

    name = model.__class__.__name__
    

    if metadata['AdLR_SLTR']:
        name += '_AdLR_SLTR'


    TensorDir,CheckPointDir = obtain_folders(name, metadata)
    if True or (CheckPointDir != '' and not metadata['full_train']):
        print('Loading Checkpoint located in', CheckPointDir)
        state_dict = torch.load(CheckPointDir)
        state_dict_aux = {}
        for k, v in state_dict.items():
            state_dict_aux['.'.join(k.split('.')[1:])] = v
        state_dict = state_dict_aux
        model.load_state_dict(state_dict, strict=True)

    else:  
        print('Loading Checkpoint located in', CheckPointDir)
        model.load_state_dict(torch.load(CheckPointDir), strict=True)

    print('Checkpoint loaded')

    metadata["TensorBoardFold"] = TensorDir
    metadata["CheckPointLoader"] = CheckPointDir
    metadata["name"] = name


    print('Model Name:', model.__class__.__name__)
    print(metadata)

    main(model, metadata)




# %%
