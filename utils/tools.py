import os
from datetime import datetime
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score
import time
from datetime import timedelta
from os import listdir
from pathlib import Path
from os.path import isfile, join


def create_folders(name, tags='', rank=None):
    if rank == 0 or rank == None:  
        checkpoints_dir = os.path.join(os.getcwd(), 'checkpoint_three', name)
        Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)



        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S").replace('/','_').replace(' ', '_')

        name_date = name + tags + '[' + dt_string + ']'




        path = os.getcwd()
        path = '/'.join(path.split('/')[:-1])

        dire = os.path.join(path, 'TensorBoard', name_date)
        Path(dire).mkdir(parents=True, exist_ok=True)


        return dire, checkpoints_dir
    
    else:
        return '', ''


def obtain_folders(name, metadata, tags='', rank=None):
    checkPointfile = None
    if 'select_file' in metadata.keys() and metadata['select_file']:
        checkPointfile = os.path.join(os.getcwd(), 'checkpoint_three', name, metadata['select_file'])
    else:
        if not 'full_train' in metadata.keys() or not metadata['full_train']:
            checkpoints_dir = os.path.join(os.getcwd(), 'checkpoint', name)
            if not os.path.exists(checkpoints_dir):
                print('No checkpoint found')
                exit(0)
            else:
                checkPointfile = [os.path.join(checkpoints_dir, f) for f in listdir(checkpoints_dir) if isfile(join(checkpoints_dir, f)) and f[0] != '.' and tags in f]
                if len(checkPointfile) == 0:
                    print('No checkpoint found')
                    checkPointfile = ''
                else:
                    if 'select_epoch' not in metadata.keys() or metadata['select_epoch'] == -1:
                        checkPointfile = [f for f in checkPointfile if 'Epoch' not in f and f.split(os.sep)[-1][0] != '.'][0]
                    else:
                        checkPointfile.sort(key=lambda x: float(os.path.splitext(x)[0].split(':')[-1]))
                        if not metadata['filtered']:
                            checkPointfile = [f for f in checkPointfile if  'Epoch' in f and
                                                                            'BigTobaccoFiltered' not in f and
                                                                            str(metadata['select_epoch']) in f.split('Epoch')[1]
                                                                            and f.split(os.sep)[-1][0] != '.']
                        elif metadata['filtered']:
                            checkPointfile = [f for f in checkPointfile if  'Epoch' in f and
                                                                            'BigTobaccoFiltered' in f and
                                                                            str(metadata['select_epoch']) in f.split('Epoch')[1]
                                                                            and f.split(os.sep)[-1][0] != '.']
                        
                        #checkPointfile.sort(key=lambda x: float(os.path.splitext(x)[0].split('_Epoch:')[-2].split(':')[-1]), reverse=True)
                        checkPointfile.sort(key=lambda x: float(os.path.splitext(x)[0].split('_Epoch:')[-1]), reverse=True)
                        checkPointfile = checkPointfile[0]        
                


    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S").replace('/','_').replace(' ', '_')

    if 'Only_Test' in metadata.keys()  and metadata['Only_Test']:
        name_date = name + '[' + dt_string + ']'
    else:
        name_date = name + '_Tobacco800_' + '[' + dt_string + ']'


    path = os.getcwd()
    path = '/'.join(path.split('/')[:-1])

    dire = os.path.join(path, 'TensorBoard', name_date)
    Path(dire).mkdir(parents=True, exist_ok=True)


    return dire, checkPointfile
    

def print_remaining_time(epoch_start, epoch, EPOCHS):
    epoch_time = time.time() - epoch_start
    epoch_timef = timedelta(seconds=epoch_time)
    time_left = epoch_time*(EPOCHS - (epoch + 1))
    time_left = timedelta(seconds=time_left)
    print('Epoch time:', epoch_timef)
    print('Remaining estimated time', time_left)


def checkPoint(metadata, model, Max_AUROC_avg, AUROC_avg, epoch=None):
    if Max_AUROC_avg < AUROC_avg or epoch is not None:
        onlyfiles = [os.path.join(metadata["CheckPointDir"], f) for f in listdir(metadata["CheckPointDir"]) if os.path.isfile(os.path.join(metadata["CheckPointDir"], f)) and metadata["name"] == f.split('_AUROC')[0]]
        if len(onlyfiles) > 0 and epoch is None:
            os.remove(onlyfiles[0])

        if 'tags' in metadata.keys() and metadata['tags'] is not None:
            name = metadata['name'] + metadata['tags']
        
        else:
            name = metadata['name']

        if epoch is None:
            name_checkpoint = os.path.join(metadata["CheckPointDir"], name + '_AUROC:' + str(AUROC_avg)+'.pkl')
        else:
            name_checkpoint = os.path.join(metadata["CheckPointDir"], name + '_AUROC:' + str(AUROC_avg) + '_Epoch:' + str(epoch) + '.pkl')

        
        metadata["CheckPointLoad"] = name_checkpoint
        torch.save(model.state_dict(), name_checkpoint)
        return AUROC_avg
    
    else:
        return Max_AUROC_avg
