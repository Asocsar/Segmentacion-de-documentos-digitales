from utils.Bert.BertMask import get_features


from datetime import timedelta
import torch.nn as nn
import torch
import time
from H5DFDataset.H5DFDataset import *
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
from utils.tools import checkPoint
from sklearn.metrics import f1_score as f1
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from types import SimpleNamespace
import pandas as pd
import numpy


from os import listdir
from os.path import isfile, join

from utils.Schedulers.allen_scheduler.slanted_triangular import SlantedTriangular

def select_loss(metadata):
    if "VGG16_Loss" in metadata.keys() and metadata['VGG16_Loss']:
        loss_func = nn.CrossEntropyLoss()

    if "SequenceClassification" in metadata.keys() and metadata['SequenceClassification']:
        loss_func = nn.BCEWithLogitsLoss()

    else:
        loss_func = nn.BCELoss()
    
    return loss_func


def train_validation_test(model, TensorLog, metadata):
    device = 0
    loss_func = select_loss(metadata)

    
    optimizer = optim.Adam(model.parameters(), lr=metadata['LR'])
    root_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
        folders = [ os.path.join(root_path,  metadata["directory_h5df_files"], 'test') ]
    else:
        folders = [ os.path.join(root_path, metadata["directory_h5df_files"], 'train'), 
                    os.path.join(root_path,  metadata["directory_h5df_files"], 'validation'),
                    os.path.join(root_path,  metadata["directory_h5df_files"], 'test')]
    
    
        
    
    train_val_test_loop(model, TensorLog, metadata, loss_func, optimizer, folders, device)


def train_validation_test_parallel(model, TensorLog, metadata, rank):
    loss_func = select_loss(metadata)
    optimizer = optim.Adam(model.parameters(), lr=metadata['LR'])
    root_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
        folders = [ os.path.join(root_path,  metadata["directory_h5df_files"], 'test') ]
    else:
        folders = [ os.path.join(root_path, metadata["directory_h5df_files"], 'train'), 
                    os.path.join(root_path,  metadata["directory_h5df_files"], 'validation'),
                    os.path.join(root_path,  metadata["directory_h5df_files"], 'test')]
    
    train_val_test_loop(model, TensorLog, metadata, loss_func, optimizer, folders, rank)


def set_model_parameters_learning(model, metadata):
    BETAS = (float(metadata['BETAS'].split('#')[0]), float(metadata['BETAS'].split('#')[1]))

    if not "LayoutLM_v2" in metadata.keys() or not metadata['LayoutLM_v2']:
        model_parameters = [{'params': [param for name, param in model.module.bert.named_parameters() if 'embedding' in name],
                            'lr': metadata['LR_BERT']*metadata['decay_factor'],
                            'betas': BETAS}]
        for k in range(model.module.bert.config.n_layers):
            params = [param for name, param in model.module.bert.named_parameters() if '.layer.' in name and int(name.split('.layer.')[1].split('.')[0]) == k]
            model_parameters.append({'params': params,
                                        'lr': metadata['LR_BERT']*(metadata['decay_factor']**k),
                                        'betas': BETAS})
    
    elif "LayoutLM_v2" in metadata.keys() and metadata['LayoutLM_v2']:

        model_parameters = [{'params': [param for name, param in model.module.Layout.named_parameters() if 'embedding' in name],
                            'lr': metadata['LR_EMBEDDING'],
                            'betas': BETAS}]

        model_parameters += [{'params': [param for name, param in model.module.Layout.named_parameters() if 'visual.backbone' in name],
                            'lr': metadata['LR_DETECTRON2'],
                            'betas': BETAS}]

        for k in range(model.module.Layout.config.num_hidden_layers):
            params = [param for name, param in model.module.Layout.named_parameters() if '.layer.' in name and int(name.split('.layer.')[1].split('.')[0]) == k]
            model_parameters.append({'params': params,
                                        'lr': metadata['LR_Layout']*(metadata['decay_factor']**k),
                                        'betas': BETAS})
    
    return model_parameters


def train_validation_test_parallel_AdaptativeLR_and_SlantedTriangular(model, TensorLog, metadata, rank):
    loss_func = select_loss(metadata)
    examples_each_file = 3000

    model_parameters = set_model_parameters_learning(model, metadata)
    
    optimizer = optim.Adam(model_parameters, lr=metadata['LR_BASE'])

    root_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
        folders = [ os.path.join(root_path,  metadata["directory_h5df_files"], 'test') ]
    else:
        folders = [ os.path.join(root_path, metadata["directory_h5df_files"], 'train'), 
                    os.path.join(root_path,  metadata["directory_h5df_files"], 'validation'),
                    os.path.join(root_path,  metadata["directory_h5df_files"], 'test')]
    
    number_train_files = len([f for f in listdir(folders[0]) if isfile(join(folders[0], f))])
    num_steps_per_epoch = int((examples_each_file*number_train_files)/metadata["BATCH"])
    if examples_each_file%metadata["BATCH"] > 0:
        num_steps_per_epoch += 1
    

    if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
        scheduler = None
    else:
        scheduler = SlantedTriangular(optimizer, num_epochs=metadata["EPOCH"], num_steps_per_epoch=num_steps_per_epoch, cut_frac=0.5)

    train_val_test_loop(model, TensorLog, metadata, loss_func, optimizer, folders, rank, scheduler)


def select_dataset(metadata, mode, filename=None, directory=None):
    if "LayoutLM_v2" not in metadata.keys() or not metadata["LayoutLM_v2"]:
        if "VGG16" in metadata.keys() and metadata["VGG16"]:
            transforms = data_transforms_VGG[mode]
        else:
            transforms = data_transforms[mode]


    if "LayoutLM_v2" not in metadata.keys() or not metadata["LayoutLM_v2"]:
        if metadata['num_pages'] == 2:
            dataset = H5Dataset_noRepeat(path=os.path.join(directory, filename),
                            data_transforms=transforms,
                            phase=mode, metadata=metadata)
        else:
            dataset = H5Dataset_ThreePages(path=os.path.join(directory, filename),
                            data_transforms=transforms,
                            phase=mode, metadata=metadata)
    
    else:
        if 'Non_Split_Dataset' in metadata.keys() and metadata['Non_Split_Dataset']:
            dataset = LayouLMV2_OnLine_Dataset(phase=mode, metadata=metadata)
        else:
            if metadata['num_pages'] == 2:
                dataset = LayouLMV2_OffLine_Dataset(path=os.path.join(directory, filename), phase=mode, metadata=metadata)
            elif metadata['num_pages'] == 3:
                dataset = LayouLMV2_Three_OffLine_Dataset(path=os.path.join(directory, filename), phase=mode, metadata=metadata)
    return dataset


def call_model( metadata, model, rank, dataset, img1=None, img2=None, img3=None, ocr=None, 
                tokens1=None, token_type_ids1=None, attention_mask1=None, box1=None, 
                tokens2=None, token_type_ids2=None, attention_mask2=None, box2=None,
                tokens3=None, token_type_ids3=None, attention_mask3=None, box3=None):

    if metadata["BertCalls"] == 2:
        ocr1 = ocr[:, 0, :].to(rank)
        ocr2 = ocr[:, 1, :].to(rank)
    
    elif metadata["BertCalls"] == 3:
        ocr1 = ocr[:, 0, :].to(rank)
        ocr2 = ocr[:, 1, :].to(rank)
        ocr3 = ocr[:, 2, :].to(rank)
    
    if "LayoutLM_v2" in metadata.keys() and metadata['LayoutLM_v2']:
        if metadata['num_pages'] == 2:
            outputs = model(img1, tokens1, token_type_ids1, attention_mask1, box1, 
                            img2, tokens2, token_type_ids2, attention_mask2, box2)
        elif metadata['num_pages'] == 3:
            outputs = model(img1, tokens1, token_type_ids1, attention_mask1, box1, 
                            img2, tokens2, token_type_ids2, attention_mask2, box2,
                            img3, tokens3, token_type_ids3, attention_mask3, box3)
        return outputs


    if metadata['num_pages'] == 2:
        if metadata["BertCalls"] == 0:
            if "VGG16" in metadata.keys() and metadata["VGG16"]:
                outputs = model(img1, img2)
            else:
                outputs = model(img1, img2, ocr)
        if metadata["BertCalls"] == 1:
            attention_mask = get_features(ocr, dataset.get_tokenizer(), rank)
            outputs = model(img1, img2, ocr, attention_mask)
        elif metadata["BertCalls"] == 2:
            attention_mask1 = get_features(ocr1, dataset.get_tokenizer(), rank)
            attention_mask2 = get_features(ocr2, dataset.get_tokenizer(), rank)
            outputs = model(img1, img2, ocr1, ocr2, attention_mask1, attention_mask2)

    
    elif metadata['num_pages'] == 3:
        if metadata["BertCalls"] == 0:
            if "VGG16" in metadata.keys() and metadata["VGG16"]:
                outputs = model(img1, img2, img3)
        if metadata["BertCalls"] == 1:
            attention_mask = get_features(ocr, dataset.get_tokenizer(), rank)
            outputs = model(img1, img2, img3, ocr, attention_mask)
        elif metadata["BertCalls"] == 2:
            attention_mask1 = get_features(ocr1, dataset.get_tokenizer(), rank)
            attention_mask2 = get_features(ocr2, dataset.get_tokenizer(), rank)
            outputs = model(img1, img2, img3, ocr1, ocr2, attention_mask1, attention_mask2)
        elif metadata["BertCalls"] == 3:
            attention_mask1 = get_features(ocr1, dataset.get_tokenizer(), rank)
            attention_mask2 = get_features(ocr2, dataset.get_tokenizer(), rank)
            attention_mask3 = get_features(ocr3, dataset.get_tokenizer(), rank)
            outputs = model(img1, img2, img3, ocr1, ocr2, ocr3, attention_mask1, attention_mask2, attention_mask3)
    
    return outputs


def obtain_ids(metadata, id1, id2, id3):
    id1 = ''.join([chr(c) for elem in id1 for c in elem if c != -1]).split('/gpfs/scratch/bsc31/bsc31282/BigTobacco')[1:]
    id2 = ''.join([chr(c) for elem in id2 for c in elem if c != -1]).split('/gpfs/scratch/bsc31/bsc31282/BigTobacco')[1:]

    if metadata['num_pages'] == 3:
        id3 = ''.join([chr(c) for elem in id3 for c in elem if c != -1]).split('/gpfs/scratch/bsc31/bsc31282/BigTobacco')[1:]
    
    return id1, id2, id3


def extract_data(metadata, batch, rank):
    if "LayoutLM_v2" not in metadata.keys() or not metadata["LayoutLM_v2"]:
        tokens1 = token_type_ids1 = attention_mask1 = box1 = None
        tokens2 = token_type_ids2 = attention_mask2 = box2 = None
        tokens3 = token_type_ids3 = attention_mask3 = box3 = None
        if metadata['num_pages'] == 2:
            img1, img2, ocr, label, id1, id2 = batch['image1'].to(rank), batch['image2'].to(rank), batch['ocr'].to(rank), batch['label'].to(rank), batch['id1'], batch['id2']
            img3 = None
            id3 = None
        else:
            img1, img2, img3, ocr, label, id1, id2, id3 = batch['image1'].to(rank), batch['image2'].to(rank), batch['image3'].to(rank), batch['ocr'].to(rank), batch['label'].to(rank), batch['id1'], batch['id2'], batch['id3']
        
    else:
        if metadata['num_pages'] == 2:
            ocr = id3 = img3 = None
            tokens3 = token_type_ids3 = attention_mask3 = box3 = None
            tokens1, token_type_ids1, attention_mask1, box1, img1 = batch['tokens1'].to(rank), batch['token_type_ids1'].to(rank), batch['attention_mask1'].to(rank), batch['box1'].to(rank), batch['image1'].to(rank)
            tokens2, token_type_ids2, attention_mask2, box2, img2 = batch['tokens2'].to(rank), batch['token_type_ids2'].to(rank), batch['attention_mask2'].to(rank), batch['box2'].to(rank), batch['image2'].to(rank)
            label = batch['label'].to(rank)
            id1, id2 = batch['id1'], batch['id2']
        elif metadata['num_pages'] == 3:
            ocr = None
            tokens1, token_type_ids1, attention_mask1, box1, img1 = batch['tokens1'].to(rank), batch['token_type_ids1'].to(rank), batch['attention_mask1'].to(rank), batch['box1'].to(rank), batch['image1'].to(rank)
            tokens2, token_type_ids2, attention_mask2, box2, img2 = batch['tokens2'].to(rank), batch['token_type_ids2'].to(rank), batch['attention_mask2'].to(rank), batch['box2'].to(rank), batch['image2'].to(rank)
            tokens3, token_type_ids3, attention_mask3, box3, img3 = batch['tokens3'].to(rank), batch['token_type_ids3'].to(rank), batch['attention_mask3'].to(rank), batch['box3'].to(rank), batch['image3'].to(rank)
            label = batch['label'].to(rank)
            id1, id2, id3 = batch['id1'], batch['id2'], batch['id3']


    return img1, img2, img3, ocr, label, id1, id2, id3, tokens1, token_type_ids1, attention_mask1, box1, tokens2, token_type_ids2, attention_mask2, box2, tokens3, token_type_ids3, attention_mask3, box3


def compute_correct_samples(metadata, label, outputs):
    if "VGG16_Loss" in metadata.keys() and metadata['VGG16_Loss']:
        correct = torch.sum(label.argmax(1, keepdim = True) == outputs.argmax(1, keepdim = True))
    else:
        correct = sum((label == (outputs >= 0.5)).cpu().numpy().flatten())

    
    return correct


def display_progress(i, rank, dataloader):
    if i % 100 == 0 and rank == 0:
        p = int((i/len(dataloader))*100000)/1000
        print('Progress', p, '%')


def display_information_endFile(rank, time_start, correct, iteration_length, iden=None):
    if rank == 0 or rank == None:
        time_end =  time.time() - time_start
        time_end = timedelta(seconds=time_end)
        if iden is not None:
            print(iden, 'finished in', time_end)

        accuracy = correct/iteration_length
        print('From {train_len} examples {correct} have been identified correctly'.format(train_len=iteration_length, correct=correct))
        print('The average accuracy for {filename} is {accuracy:.3f}'.format(filename=iden, accuracy=accuracy))  


def register_informtion_endPhase_checkpoint_TensorBoard(metadata, model, TensorLog, rank, epoch, correct, loss, dataset_len, iteration_length, mode, Max_accuracy, y_pred, y_truth):
    if rank == 0 or rank == None:
        loss = loss / dataset_len
        print('Epoch {epoch} {mode} loss: {loss:.3f}'.format(epoch=epoch + 1, mode=mode, loss=loss ))
        print('From {train_len} examples {correct} have been identified correctly'.format(train_len=iteration_length, correct=correct))
        accuracy = correct/iteration_length
        f1_score = f1(y_truth, y_pred)
        kappa_score = kappa(y_truth, y_pred)

        if mode == 'val':
            Max_accuracy = checkPoint(metadata, model, Max_accuracy, accuracy, epoch=epoch)
        else:
            Max_accuracy = 0

        print('The average accuracy is {accuracy:.3f}'.format(accuracy=accuracy))
        print('F1_Score is', f1_score)
        print('Kappa Score is', kappa_score)

        TensorLog.add_scalar('Accuracy/' + mode, accuracy, epoch)
        TensorLog.add_scalar('Loss/' + mode, loss , epoch)

        if "VGG16_Loss" in metadata.keys() and metadata["VGG16_Loss"]:
            accuracy = float(accuracy.cpu().numpy())
        
        if mode == 'val' or mode == 'test':
            TensorLog.add_hparams(  {"Test_Accuracy_" + mode: accuracy, "F1_Score_" + mode: f1_score, "Kappa_Score_" + mode: kappa_score}, 
                                    {"Test_Accuracy_" + mode: accuracy, "F1_Score_" + mode: f1_score, "Kappa_Score_" + mode: kappa_score})

        return Max_accuracy


def train_val_test_loop(model, TensorLog, metadata, loss_func, optimizer, folders, rank=None, scheduler=None):
    Max_accuracy = 0
    for epoch in range(metadata["EPOCH"]):
        if "Non_Split_Dataset" in metadata.keys() and metadata["Non_Split_Dataset"]:
            train_val_test_loop_dataset_United(epoch, model, TensorLog, metadata, loss_func, optimizer, folders, rank, scheduler, Max_accuracy)
        else:
            train_val_test_loop_dataset_split_multipleFiles(epoch, model, TensorLog, metadata, loss_func, optimizer, folders, rank, scheduler, Max_accuracy)


def train_val_test_loop_dataset_split_multipleFiles(epoch, model, TensorLog, metadata, loss_func, optimizer, folders, rank, scheduler, Max_accuracy):
    for m, directory in enumerate(folders):
        if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
            model.eval()
            mode = 'test'
            print('Testing Model')
        elif m == 0:
            model.train()
            mode = 'train'
            if rank == 0:
                print('Training Model')
        elif m == 1:
            model.eval()
            mode = 'val'
            if rank == 0:
                print('Validating Model')
        
        elif m == 2:
            model.eval()
            mode = 'test'
            if rank == 0:
                print('Testing Model')
        
        if model.training:
            assert(mode == 'train')
            i = 0
        else:
            assert((mode == 'val') or (mode == 'test'))
        time_start = time.time()
        random.seed(12)
        all_files = os.listdir(directory)
        random.shuffle(all_files) #sorted(os.listdir(directory), key = lambda fil: int(os.path.splitext(fil)[0].split('_')[-1]))

        sections = int(len(all_files)/metadata["num_GPUs"]) + len(all_files) % metadata["num_GPUs"]
        if mode == 'train' and rank is not None:
            all_files = all_files[sections*rank:sections*(rank+1)]
        
        iteration_length = 0
        loss = 0.0
        correct = 0
        dataset_len = 0
        y_pred = numpy.empty([0])
        y_truth = numpy.empty([0])
        for k, filename in enumerate(all_files):
            loss_file = 0
            dataset = select_dataset(metadata, mode, filename, directory)

            dataloader = DataLoader(dataset=dataset,
                            batch_size=metadata["BATCH"],
                            shuffle=True,
                            num_workers=metadata["workers"],
                            persistent_workers=True)

            dataset_len += len(dataset)
            

            load_start =  time.time()
            for i, batch in enumerate(dataloader):
                load_end =  time.time() - load_start
                time_load = timedelta(seconds=load_end)
                #print('Time Load data', time_load)

                batch_start =  time.time()

                img1, img2, img3, ocr, label, id1, id2, id3, tokens1, token_type_ids1, attention_mask1, box1, tokens2, token_type_ids2, attention_mask2, box2, tokens3, token_type_ids3, attention_mask3, box3 = extract_data(metadata, batch, rank)


                id1, id2, id3 = obtain_ids(metadata, id1, id2, id3)
                

                bh, _, _, _ = img1.shape

                optimizer.zero_grad()


                outputs = call_model(   
                                        metadata, model, rank, dataset, img1=img1, img2=img2, img3=img3, ocr=ocr, 
                                        tokens1=tokens1, token_type_ids1=token_type_ids1,attention_mask1=attention_mask1, box1=box1,
                                        tokens2=tokens2, token_type_ids2=token_type_ids2, attention_mask2=attention_mask2, box2=box2,
                                        tokens3=tokens3, token_type_ids3=token_type_ids3, attention_mask3=attention_mask3, box3=box3
                                    )
                
                


                if "SequenceClassification" in metadata.keys() and metadata['SequenceClassification']:
                    correct += compute_correct_samples(metadata, label, torch.sigmoid(outputs))
                else:
                    correct += compute_correct_samples(metadata, label, outputs)



                if "VGG16_Loss" in metadata.keys() and metadata["VGG16_Loss"]:
                    top_pred = outputs.argmax(1, keepdim = True)
                    top_label = label.argmax(1, keepdim = True)
                    y_pred = numpy.concatenate([y_pred, top_pred.cpu().numpy().flatten()])
                    y_truth = numpy.concatenate([y_truth,top_label.cpu().numpy().flatten()])

                elif "SequenceClassification" in metadata.keys() and metadata['SequenceClassification']:
                    y_pred = numpy.concatenate([y_pred, (torch.sigmoid(outputs) >= 0.5).cpu().numpy().flatten()])
                    y_truth = numpy.concatenate([y_truth,label.cpu().numpy().flatten()])
                else:
                    y_pred = numpy.concatenate([y_pred, (outputs >= 0.5).cpu().numpy().flatten()])
                    y_truth = numpy.concatenate([y_truth,label.cpu().numpy().flatten()])
                    

                loss_batch = loss_func(outputs, label)
                
                '''if torch.sum(torch.isnan(outputs)) >= 1:
                    print('Nan in batch', i)
                    
                    print('=============')
                    print('ID1:', id1)
                    print('ID2:', id2)
                    print('ID3:', id3)
                    print('Label:', label)
                    print('Outputs:', outputs)
                    print('Loss_batch:', loss_batch)
                    print('=============')'''
                  
                loss_batch.backward()
                optimizer.step()

                display_progress(i, rank, dataloader)

                with torch.no_grad():
                    loss += loss_batch.item()
                    loss_file =+ loss_batch.item()
                    iteration_length += bh
                
                if mode == 'train' and scheduler is not None:
                    scheduler.step_batch()
                
                batch_end =  time.time() - batch_start
                time_batch = timedelta(seconds=batch_end)
                #print('Batch Time', time_batch)
                load_start =  time.time()

                
            
            display_information_endFile(rank, time_start, correct, iteration_length, iden=filename) 
            print('Loss_File:', loss_file)
            

                
        if mode == 'val' and scheduler is not None:
            scheduler.step()

        
        Max_accuracy = register_informtion_endPhase_checkpoint_TensorBoard(metadata, model, TensorLog, rank, epoch, correct, loss, dataset_len, iteration_length, mode, Max_accuracy, y_pred, y_truth)
        

def train_val_test_loop_dataset_United(epoch, model, TensorLog, metadata, loss_func, optimizer, folders, rank, scheduler, Max_accuracy):
    phases = ['train', 'val', 'test']
    if 'Only_Test' in metadata.keys() and metadata['Only_Test']:
        phases = ['test']
    for m, mode in enumerate(phases):
        if m == 0:
            model.train()
            if rank == 0:
                print('Training Model')
        elif m == 1:
            model.eval()
            if rank == 0:
                print('Validating Model')
        
        elif m == 2:
            model.eval()
            if rank == 0:
                print('Testing Model')
        
        if model.training:
            assert(mode == 'train')
            i = 0
        else:
            assert((mode == 'val') or (mode == 'test'))
        time_start = time.time()



        iteration_length = 0
        loss = 0.0
        correct = 0
        dataset_len = 0

        dataset = select_dataset(metadata, mode)

        distributed_sampler = DistributedSampler(dataset,
                                    num_replicas=metadata['num_GPUs'],
                                    shuffle=True,
                                    rank=rank)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=metadata["BATCH"],
                                shuffle=False,
                                num_workers=metadata["workers"],
                                sampler=distributed_sampler)
        


        dataset_len += len(dataset)

        dataloader.sampler.set_epoch(epoch)
        load_start =  time.time()
        for i, batch in enumerate(dataloader):
            load_end =  time.time() - load_start
            time_load = timedelta(seconds=load_end)
            #print('Time Load data', time_load)

            batch_start =  time.time()
            img1, img2, img3, ocr, label, id1, id2, id3, tokens1, token_type_ids1, attention_mask1, box1, tokens2, token_type_ids2, attention_mask2, box2, tokens3, token_type_ids3, attention_mask3, box3 = extract_data(metadata, batch, rank)


            id1, id2, id3 = obtain_ids(metadata, id1, id2, id3)

            bh = img1.shape[0]

            optimizer.zero_grad()

            outputs = call_model(   
                                    metadata, model, rank, dataset, img1=img1, img2=img2, img3=img3, ocr=ocr, 
                                    tokens1=tokens1, token_type_ids1=token_type_ids1,attention_mask1=attention_mask1, box1=box1,
                                    tokens2=tokens2, token_type_ids2=token_type_ids2, attention_mask2=attention_mask2, box2=box2,
                                    tokens3=tokens3, token_type_ids3=token_type_ids3, attention_mask3=attention_mask3, box3=box3
                                )

            correct += compute_correct_samples(metadata, label, outputs)
            
            loss_batch = loss_func(outputs, label)

            loss_batch.backward()
            optimizer.step()

            display_progress(i, rank, dataloader)

            with torch.no_grad():
                loss += loss_batch.item()
                iteration_length += bh
            
            if mode == 'train' and scheduler is not None:
                scheduler.step_batch()
            
            batch_end =  time.time() - batch_start
            time_batch = timedelta(seconds=batch_end)
            #print('Batch Time', time_batch)
            load_start =  time.time()
            


        display_information_endFile(rank, time_start, correct, iteration_length, iden=mode) 
    
    if mode == 'val' and scheduler is not None:
        scheduler.step()
    
    Max_accuracy = register_informtion_endPhase_checkpoint_TensorBoard(metadata, model, TensorLog, rank, epoch, correct, loss, dataset_len, iteration_length, mode, Max_accuracy)


def test_tobacco800(model, metadata, TensorLog):
    optimizer = optim.Adam(model.parameters(), lr=metadata['LR_BASE'])
    if "VGG16_Loss" in metadata.keys() and metadata['VGG16_Loss']:
        loss_func = nn.CrossEntropyLoss()

    else:
        loss_func = nn.BCELoss()

    rank = 'cuda:0'
    root_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
    if metadata['fine_tune'] or metadata['full_train']:
        folders_train = [   ('train', os.path.join(root_path,  metadata["directory_tobacco800"] + '_split', 'train')),
                            ('val', os.path.join(root_path,  metadata["directory_tobacco800"] + '_split', 'validation')),
                            ('test', os.path.join(root_path,  metadata["directory_tobacco800"]+ '_split', 'test'))]
        folders_test = [    ('test', os.path.join(root_path,  metadata["directory_tobacco800"]+ '_split', 'test')) ]
    else:
        folders_test = [    ('test', os.path.join(root_path,  metadata["directory_tobacco800"], 'test')) ]
        folders_train = None
    
    model = model.eval()

    if folders_train is not None:
        for i in range(30): #range(metadata['EPOCH']):
            print('==================')
            print('Epoch', i)
            for mode, directory in folders_train:
                if mode == 'train':
                    model.train()
                    print('Training Model')

                elif mode == 'val':
                    model.eval()
                    print('Validating Model')
                
                elif mode == 'test':
                    model.eval()
                    print('Testing Model')
                
                tobacco800_loop(model, metadata, TensorLog, directory, mode, rank, optimizer, loss_func)

            print('==================')
    

    for mode, directory in folders_test:
            model.eval()
            print('Testing Model')
            tobacco800_loop(model, metadata, TensorLog, directory, mode, rank)

        
def tobacco800_loop(model, metadata, TensorLog, directory, mode, rank, optimizer=None, loss_func=None):
    time_start = time.time()
    all_files = sorted(os.listdir(directory), key = lambda fil: int(os.path.splitext(fil)[0].split('_')[-1]))


    

    y_pred = numpy.empty([0])
    y_truth = numpy.empty([0])
    iteration_length = 0
    correct = 0
    for filename in all_files:
        dataset = select_dataset(metadata, mode, filename, directory)



        dataloader = DataLoader(dataset=dataset,
                        batch_size=metadata["BATCH"],
                        shuffle=False,
                        num_workers=metadata["workers"])

    
        for i, batch in enumerate(dataloader):
            img1, img2, img3, ocr, label, id1, id2, id3, tokens1, token_type_ids1, attention_mask1, box1, tokens2, token_type_ids2, attention_mask2, box2, tokens3, token_type_ids3, attention_mask3, box3 = extract_data(metadata, batch, rank)

            id1, id2, id3 = obtain_ids(metadata, id1, id2, id3)

            bh, _, _, _ = img1.shape

            display_progress(i, rank, dataloader)

            
            
            if loss_func is not None:
                optimizer.zero_grad()
                outputs = call_model(   
                                    metadata, model, rank, dataset, img1=img1, img2=img2, img3=img3, ocr=ocr, 
                                    tokens1=tokens1, token_type_ids1=token_type_ids1,attention_mask1=attention_mask1, box1=box1,
                                    tokens2=tokens2, token_type_ids2=token_type_ids2, attention_mask2=attention_mask2, box2=box2,
                                    tokens3=tokens3, token_type_ids3=token_type_ids3, attention_mask3=attention_mask3, box3=box3
                                )
                
                loss_batch = loss_func(outputs, label)
                loss_batch.backward()
                optimizer.step()
            
            else:
                with torch.no_grad():
                    outputs = call_model(   
                                    metadata, model, rank, dataset, img1=img1, img2=img2, img3=img3, ocr=ocr, 
                                    tokens1=tokens1, token_type_ids1=token_type_ids1,attention_mask1=attention_mask1, box1=box1,
                                    tokens2=tokens2, token_type_ids2=token_type_ids2, attention_mask2=attention_mask2, box2=box2,
                                    tokens3=tokens3, token_type_ids3=token_type_ids3, attention_mask3=attention_mask3, box3=box3
                                )
                

            

            if "SequenceClassification" in metadata.keys() and metadata['SequenceClassification']:
                correct += compute_correct_samples(metadata, label, torch.sigmoid(outputs))
            else:
                correct += compute_correct_samples(metadata, label, outputs)
                

            if "VGG16_Loss" in metadata.keys() and metadata["VGG16_Loss"]:
                top_pred = outputs.argmax(1, keepdim = True)
                top_label = label.argmax(1, keepdim = True)
                y_pred = numpy.concatenate([y_pred, top_pred.cpu().numpy().flatten()])
                y_truth = numpy.concatenate([y_truth,top_label.cpu().numpy().flatten()])

            elif "SequenceClassification" in metadata.keys() and metadata['SequenceClassification']:
                y_pred = numpy.concatenate([y_pred, (torch.sigmoid(outputs) >= 0.5).cpu().numpy().flatten()])
                y_truth = numpy.concatenate([y_truth,label.cpu().numpy().flatten()])
            else:
                y_pred = numpy.concatenate([y_pred, (outputs >= 0.5).cpu().numpy().flatten()])
                y_truth = numpy.concatenate([y_truth,label.cpu().numpy().flatten()])

            

            iteration_length += bh
            

        
        display_information_endFile(rank, time_start, correct, iteration_length, iden=filename)            
        

    print('From {train_len} examples {correct} have been identified correctly'.format(train_len=iteration_length, correct=correct))
    accuracy = correct/iteration_length
    f1_score = f1(y_truth, y_pred)
    kappa_score = kappa(y_truth, y_pred)
    print(kappa_score)


    print('The average accuracy is {accuracy:.3f}, f1 score {f1score:.3f} and kappa score {kappascore:.3f}'.format(accuracy=accuracy, f1score=f1_score, kappascore=kappa_score))
    #TensorLog.add_hparams({"Test_Accuracy": accuracy, "F1_Score": f1_score}, {"Test_Accuracy": accuracy, "F1_Score": f1_score})

    if False and mode == 'test':
        labels = ['New_document', 'Same_Document']
        conf_matrix = confusion_matrix(y_truth, y_pred)

        

        ax = sns.heatmap(conf_matrix, annot=True, fmt='g');  #annot=True to annotate cells, ftm='g' to disable scientific notation

        for e, t in enumerate(ax.texts):
            if e < 2:
                des = 0.25
            else:
                des = -0.25

            trans = t.get_transform()
            offs = matplotlib.transforms.ScaledTranslation(0.0, des,
                            matplotlib.transforms.IdentityTransform())
            t.set_transform( offs + trans )
        
        
        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.savefig('../Images/confusion_matrix_' + metadata['name'] + '_.png')
