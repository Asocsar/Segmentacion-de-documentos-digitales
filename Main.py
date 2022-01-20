#%%
from torch.utils.tensorboard import SummaryWriter

from utils.tools import create_folders, obtain_folders


from utils.parallelize import setup

from train_validation_test import *

from Models.TwoPages_Without_DenseLayer import *
from Models.TwoPages_With_DenseLayer import *


from Models.ThreePages_Without_DenseLayer import *
from Models.ThreePages_With_DenseLayer import *

from Models.VGG16 import *



import json
import os

import sys

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def load_checkpoint(model):
    TensorDir,CheckPointDir = obtain_folders(name, metadata)
    if CheckPointDir != '':
        print('Loading Checkpoint located in', CheckPointDir)
        state_dict = torch.load(CheckPointDir)
        
        state_dict_aux = {}
        if ("VGG16" not in metadata.keys() or not metadata["VGG16"]):
            for k in state_dict.keys():
                state_dict_aux['.'.join(k.split('.')[1:])] = state_dict[k]
            state_dict = state_dict_aux
        model.load_state_dict(state_dict, strict=True)
        print('Checkpoint loaded')
    
    return model, TensorDir, CheckPointDir


def model_election(metadata):
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
def main(rank, model, metadata):
    if metadata['Parallel']:
        setup(rank, metadata["num_GPUs"])
    
    model = model.to(rank)
    
    if metadata['Parallel']:
        if (('Version' in metadata.keys() and metadata['Version'] == 1) 
            or ('VGG16' in metadata.keys() and metadata['VGG16'])
            or ("LayoutLM_v2" in metadata.keys() and metadata['LayoutLM_v2'])):
            ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        else:
            ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        ddp_model = model
    tokenizer_file = open('./tokenizer_saved/tokenizer_config.json')
    tokenizer_config = json.load(tokenizer_file)
    tokenizer_file.close()
    metadata["TokenizerMaxLength"] = tokenizer_config["model_max_length"]

    
    TensorLog = None
    if rank == 0:
        TensorLog = SummaryWriter(metadata['TensorBoardFold'])
        with open(os.path.join(metadata['TensorBoardFold'], 'config.json'), 'w+') as out_conf:
            json.dump(metadata, out_conf)
    
    
    if metadata['Parallel']:
        
        if metadata['AdLR_SLTR']:
            train_validation_test_parallel_AdaptativeLR_and_SlantedTriangular(ddp_model, TensorLog, metadata, rank)
        else:
            train_validation_test_parallel(ddp_model, TensorLog, metadata, rank)
    else:
        
        train_validation_test(ddp_model, TensorLog, metadata)

#%%
if __name__ == '__main__':
    conf_file = os.path.join("Main_conf_files", sys.argv[1])


    f = open(conf_file)
    metadata = json.load(f)
    f.close()
    
    
    model = model_election(metadata)

    name = model.__class__.__name__

    if metadata['AdLR_SLTR']:
        name += '_AdLR_SLTR'

    if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
        model, TensorDir, CheckPointDir = load_checkpoint(model)
    else:
        if "tags" in metadata.keys():
            TensorDir,CheckPointDir = create_folders(name, metadata["tags"])
        else:
            TensorDir,CheckPointDir = create_folders(name)

    metadata["TensorBoardFold"] = TensorDir
    metadata["CheckPointDir"] = CheckPointDir
    metadata["name"] = name


    print('Model Name:', model.__class__.__name__)
    print(metadata)

    if ('Parallel' in metadata.keys() and not metadata['Parallel']) or ('Only_Test' in metadata.keys() and metadata['Only_Test']):
        
        main(0, model, metadata)
    else:
        mp.spawn(
            main,
            args=(model, metadata,),
            nprocs=metadata["num_GPUs"],
            join=True
        )




# %%
