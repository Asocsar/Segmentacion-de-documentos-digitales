#%%
import h5py
from PIL import Image, ImageSequence
import cv2
import os
import torch


from torch.utils.data import Dataset
import numpy as np
from transformers import DistilBertTokenizer
from torchvision.transforms.transforms import Resize
from torchvision import transforms
import cv2
from os import listdir
from os.path import isfile, join
import pickle
import pandas as pd
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast, LayoutLMv2Processor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class H5Dataset_noRepeat(Dataset):

    def __init__(self, path, data_transforms, phase, metadata):
        self.file_path = path
        self.dataset = None
        self.data = None
        self.target = None
        self.ocr = None
        self.phase = phase
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(os.getcwd(), 'tokenizer_saved'))
        self.metadata = metadata
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_imgs"])
            elif phase == 'val':
                self.dataset_len = len(file["validation_imgs"])
            elif phase == 'test':
                self.dataset_len = len(file["test_imgs"])

        self.data_transforms = data_transforms

    def __len__(self):
        return self.dataset_len - 1
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def treat_OCR(self, ocr, number_ocrs):
        extra_tokens = self.metadata['num_pages'] + 1
        pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        if len(ocr) >= (self.metadata["TokenizerMaxLength"] - extra_tokens)/number_ocrs:
            ocr = ocr[:int((self.metadata["TokenizerMaxLength"] - extra_tokens)/number_ocrs)]
            ocr = torch.Tensor(ocr).type(torch.LongTensor)
            
        elif len(ocr) < self.metadata["TokenizerMaxLength"]/number_ocrs:
            padding = int((self.metadata["TokenizerMaxLength"] - extra_tokens)/number_ocrs) - len(ocr)
            ocr = ocr + ([pad_id] * padding)
            ocr = torch.Tensor(ocr).type(torch.LongTensor)
        
        
        return ocr
    
    def generate_tokens(self, ocrs):
        CLS = torch.Tensor([self.tokenizer.convert_tokens_to_ids('[CLS]')]).type(torch.LongTensor)
        SEP = torch.Tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')]).type(torch.LongTensor)
        all_tokens = []
        for ocr in ocrs:
            tokens = self.tokenizer.tokenize(ocr)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens = self.treat_OCR(tokens, len(ocrs))
            all_tokens.append(tokens)
        
        ocr = torch.cat((CLS, all_tokens[0], SEP), 0)
        for tok in all_tokens[1:]:
            ocr = torch.cat((ocr, tok, SEP), 0)
        
    
        return ocr

    def __getitem__(self, idx):
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img = self.dataset.get('train_imgs')
                self.ocr = self.dataset.get('train_ocrs')
                self.target = self.dataset.get('train_labels')
                self.id = self.dataset.get('train_id')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img = self.dataset.get('validation_imgs')
                self.ocr = self.dataset.get('validation_ocrs')
                self.target = self.dataset.get('validation_labels')
                self.id = self.dataset.get('validation_id')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img = self.dataset.get('test_imgs')
                self.ocr = self.dataset.get('test_ocrs')
                self.target = self.dataset.get('test_labels')
                self.id = self.dataset.get('test_id')
                self.pad_id = max([len(id) for id in self.id])
        
        
        if idx < len(self.target) - 1:

            img1 = self.img[idx,:,:,:]
            img2 = self.img[idx+1,:,:,:]

            img1 = img1.astype('uint8')
            img2 = img2.astype('uint8')

            if 'Otsu' in self.metadata.keys() and self.metadata['Otsu']:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                (_, img1) = cv2.threshold(gray1, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
                
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                (_, img2) = cv2.threshold(gray2, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)




            img1 = Image.fromarray(img1, 'RGB')
            img2 = Image.fromarray(img2, 'RGB')

            ocr1 = str(self.ocr[idx])
            ocr2 = str(self.ocr[idx+1])

            if ocr1 == '':
                ocr = 'empty'
            if ocr2 == '':
                ocr2 = 'empty'



            if self.metadata['BertCalls'] == 1 or self.metadata['BertCalls'] == 0:
                ocrs = [ocr1, ocr2]
                ocr = self.generate_tokens(ocrs)
            
            elif self.metadata['BertCalls'] == 2:

                ocrs1 = [ocr1]
                ocrs2 = [ocr2]
                ocr1 = self.generate_tokens(ocrs1)
                ocr2 = self.generate_tokens(ocrs2)

                ocr = torch.stack((ocr1, ocr2), 0)

            label1 = self.target[idx]
            label2 = self.target[idx+1]
            label = max(label1, label2)
            if "VGG16_Loss" in self.metadata.keys() and self.metadata['VGG16_Loss']:
                #label = int(label)
                label_tensor = torch.empty(2, dtype=torch.float)
                if label == 1:
                    label_tensor[0] = 0
                else:
                    label_tensor[1] = 0

                label_tensor[label] = 1
                label = label_tensor
            else:
                label = torch.tensor(data=label, dtype=torch.float)

            if self.data_transforms is not None:
                try:
                    img1 = self.data_transforms(img1)
                    img2 = self.data_transforms(img2)
                except:
                    print("Cannot transform image: {}")        

            id1 = self.id[idx]
            id1 = [ord(c) for c in id1]
            id1 = id1 + [-1 for _ in range(self.pad_id - len(id1))]
            id1 = torch.tensor(data=id1, dtype=torch.int)

            id2 = self.id[idx+1]
            id2 = [ord(c) for c in id2]
            id2 = id2 + [-1 for _ in range(self.pad_id - len(id2))]
            id2 = torch.tensor(data=id2, dtype=torch.int)
            if "VGG16_Loss" in self.metadata.keys() and self.metadata["VGG16_Loss"]:
                batch = {'image1': img1, 'image2': img2, 'ocr': ocr, 'label': label , 'id1': id1, 'id2': id2}
            else:
                batch = {'image1': img1, 'image2': img2, 'ocr': ocr, 'label': torch.unsqueeze(label, 0) , 'id1': id1, 'id2': id2}
            return batch


class H5Dataset_ThreePages(Dataset):

    def __init__(self, path, data_transforms, phase, metadata):
        self.file_path = path
        self.dataset = None
        self.data = None
        self.target = None
        self.ocr = None
        self.phase = phase
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(os.getcwd(), 'tokenizer_saved'))
        self.metadata = metadata
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_imgs"])
            elif phase == 'val':
                self.dataset_len = len(file["validation_imgs"])
            elif phase == 'test':
                self.dataset_len = len(file["test_imgs"])

        self.data_transforms = data_transforms

    def __len__(self):
        return self.dataset_len - 2
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def treat_OCR(self, ocr, number_ocrs):
        extra_tokens = self.metadata['num_pages'] + 1
        pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        if len(ocr) >= (self.metadata["TokenizerMaxLength"] - extra_tokens)/number_ocrs:
            ocr = ocr[:int((self.metadata["TokenizerMaxLength"] - extra_tokens)/number_ocrs)]
            ocr = torch.Tensor(ocr).type(torch.LongTensor)
            
        elif len(ocr) < self.metadata["TokenizerMaxLength"]/number_ocrs:
            padding = int((self.metadata["TokenizerMaxLength"] - extra_tokens)/number_ocrs) - len(ocr)
            ocr = ocr + ([pad_id] * padding)
            ocr = torch.Tensor(ocr).type(torch.LongTensor)
        
        
        return ocr
    
    def generate_tokens(self, ocrs):
        CLS = torch.Tensor([self.tokenizer.convert_tokens_to_ids('[CLS]')]).type(torch.LongTensor)
        SEP = torch.Tensor([self.tokenizer.convert_tokens_to_ids('[SEP]')]).type(torch.LongTensor)
        all_tokens = []
        for ocr in ocrs:
            tokens = self.tokenizer.tokenize(ocr)
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            tokens = self.treat_OCR(tokens, len(ocrs))
            all_tokens.append(tokens)
        
        ocr = torch.cat((CLS, all_tokens[0], SEP), 0)
        for tok in all_tokens[1:]:
            ocr = torch.cat((ocr, tok, SEP), 0)
        
    
        return ocr

    def __getitem__(self, idx):
        index = idx + 1
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img = self.dataset.get('train_imgs')
                self.ocr = self.dataset.get('train_ocrs')
                self.target = self.dataset.get('train_labels')
                self.id = self.dataset.get('train_id')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img = self.dataset.get('validation_imgs')
                self.ocr = self.dataset.get('validation_ocrs')
                self.target = self.dataset.get('validation_labels')
                self.id = self.dataset.get('validation_id')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img = self.dataset.get('test_imgs')
                self.ocr = self.dataset.get('test_ocrs')
                self.target = self.dataset.get('test_labels')
                self.id = self.dataset.get('test_id')
                self.pad_id = max([len(id) for id in self.id])
        
            #print(self.file_path, sum([label for label in self.target]))
        
        if idx < len(self.target) - 1:

            img1 = self.img[index-1,:,:,:]
            img2 = self.img[index,:,:,:]
            img3 = self.img[index+1,:,:,:]

            if 'Otsu' in self.metadata.keys() and self.metadata['Otsu']:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                (_, img1) = cv2.threshold(gray1, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)

                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                (_, img2) = cv2.threshold(gray2, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

                gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                (_, img3) = cv2.threshold(gray3, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                img3 = cv2.cvtColor(img3,cv2.COLOR_GRAY2RGB)

            img1 = Image.fromarray(img1.astype('uint8'), 'RGB')
            img2 = Image.fromarray(img2.astype('uint8'), 'RGB')
            img3 = Image.fromarray(img3.astype('uint8'), 'RGB')

            ocr1 = str(self.ocr[idx-1])
            ocr2 = str(self.ocr[idx])
            ocr3 = str(self.ocr[idx+1])

            if ocr1 == '':
                ocr = 'empty'
            if ocr2 == '':
                ocr2 = 'empty'
            if ocr3 == '':
                ocr2 = 'empty'


            if self.metadata['BertCalls'] == 1 or self.metadata['BertCalls'] == 0:
                ocrs = [ocr1, ocr2, ocr3]
                ocr = self.generate_tokens(ocrs)
            
            elif self.metadata['BertCalls'] == 2:

                ocrs1 = [ocr1, ocr2]
                ocrs2 = [ocr2, ocr3]
                ocr1 = self.generate_tokens(ocrs1)
                ocr2 = self.generate_tokens(ocrs2)

                ocr = torch.stack((ocr1, ocr2), 0)

            elif self.metadata['BertCalls'] == 3:
                ocrs1 = [ocr1]
                ocrs2 = [ocr2]
                ocrs3 = [ocr3]  
                ocr1 = self.generate_tokens(ocrs1)
                ocr2 = self.generate_tokens(ocrs2)
                ocr3 = self.generate_tokens(ocrs3)

                ocr = torch.stack((ocr1, ocr2, ocr3), 0)
            


            label1 = self.target[idx-1]
            label2 = self.target[idx]
            label3 = self.target[idx+1]

            #print('Label1:', label1, 'Label2:', label2, 'Label3:', label3)

            label = max(max(label1, label2), label3)
            if "VGG16_Loss" in self.metadata.keys() and self.metadata['VGG16_Loss']:
                label_tensor = torch.empty(2, dtype=torch.float)
                if label == 1:
                    label_tensor[0] = 0
                else:
                    label_tensor[1] = 0

                label_tensor[label] = 1
                label = label_tensor
            else:
                label = torch.tensor(data=label, dtype=torch.float)

            if self.data_transforms is not None:
                try:
                    img1 = self.data_transforms(img1)
                    img2 = self.data_transforms(img2)
                    img3 = self.data_transforms(img3)
                except:
                    print("Cannot transform image: {}")        

            id1 = self.id[idx-1]
            id1 = [ord(c) for c in id1]
            id1 = id1 + [-1 for _ in range(self.pad_id - len(id1))]
            id1 = torch.tensor(data=id1, dtype=torch.int)

            id2 = self.id[idx]
            id2 = [ord(c) for c in id2]
            id2 = id2 + [-1 for _ in range(self.pad_id - len(id2))]
            id2 = torch.tensor(data=id2, dtype=torch.int)

            id3 = self.id[idx+1]
            id3 = [ord(c) for c in id3]
            id3 = id3 + [-1 for _ in range(self.pad_id - len(id3))]
            id3 = torch.tensor(data=id3, dtype=torch.int)
            
            if "VGG16_Loss" in self.metadata.keys() and self.metadata['VGG16_Loss']:
                 batch = {'image1': img1, 'image2': img2, 'image3': img3, 'ocr': ocr, 'label': label, 'id1': id1, 'id2': id2, 'id3': id3}
            else:
                batch = {'image1': img1, 'image2': img2, 'image3': img3, 'ocr': ocr, 'label': torch.unsqueeze(label, 0), 'id1': id1, 'id2': id2, 'id3': id3}
            return batch


class H5Dataset_Repeat(Dataset):

    def __init__(self, path, data_transforms, phase, metadata):
        self.file_path = path
        self.dataset = None
        self.data = None
        self.target = None
        self.ocr = None
        self.phase = phase
        self.tokenizer = DistilBertTokenizer.from_pretrained(os.path.join(os.getcwd(), 'tokenizer_saved'))
        self.metadata = metadata
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_img1"])
            elif phase == 'val':
                self.dataset_len = len(file["val_img1"])
            elif phase == 'test':
                self.dataset_len = len(file["test_img1"])

        self.data_transforms = data_transforms

    def __len__(self):
        return self.dataset_len


    def __getitem__(self, idx):
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img1 = self.dataset.get('train_img1')
                self.img2 = self.dataset.get('train_img2')
                self.ocr1 = self.dataset.get('train_ocrs1')
                self.ocr2 = self.dataset.get('train_ocrs2')
                self.target = self.dataset.get('train_labels')
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img1 = self.dataset.get('val_img1')
                self.img2 = self.dataset.get('val_img2')
                self.ocr1 = self.dataset.get('val_ocrs1')
                self.ocr2 = self.dataset.get('val_ocrs2')
                self.target = self.dataset.get('val_labels')
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')
                self.img1 = self.dataset.get('test_img1')
                self.img2 = self.dataset.get('test_img2')
                self.ocr1 = self.dataset.get('test_ocrs1')
                self.ocr2 = self.dataset.get('test_ocrs2')
                self.target = self.dataset.get('test_labels')

        img1 = self.img1[idx,:,:,:]
        img1 = Image.fromarray(img1.astype('uint8'), 'RGB')

        img2 = self.img2[idx,:,:,:]
        img2 = Image.fromarray(img2.astype('uint8'), 'RGB')

        ocr1 = str(self.ocr1[idx])
        if ocr1 == '':
            ocr1 = 'empty'

        ocr2 = str(self.ocr2[idx])
        if ocr2 == '':
            ocr2 = 'empty'
        
        ocr = self.tokenizer.encode(ocr1[:512], ocr2[:512], return_tensors="pt")[:, :512]

        label = self.target[idx]

        if self.data_transforms is not None:
            try:
                img1 = self.data_transforms(img1)
                img2 = self.data_transforms(img2)
            except:
                print("Cannot transform image: {}")


        
        pad_id = self.tokenizer.convert_tokens_to_ids('[PAD]')
        if len(ocr.squeeze()) >= 512:
            ocr = ocr.squeeze().tolist()
            ocr = ocr[:512]
            ocr = torch.Tensor(ocr).type(torch.LongTensor)
            
        elif len(ocr.squeeze()) < 512:
            ocr = ocr.squeeze().tolist()
            padding = 512 - len(ocr)
            ocr = ocr + ([pad_id] * padding)
            ocr = torch.Tensor(ocr).type(torch.LongTensor)
        
        label = torch.tensor(data=label, dtype=torch.float)

        return img1, img2, ocr, label


class LayouLMV2_OffLine_Dataset(Dataset):

    def __init__(self, path, phase, metadata):
        self.file_path = path
        self.dataset = None
        self.target = None
        self.phase = phase
        self.metadata = metadata
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_imgs"])
            elif phase == 'val':
                self.dataset_len = len(file["validation_imgs"])
            elif phase == 'test':
                self.dataset_len = len(file["test_imgs"])


    def __len__(self):
        return self.dataset_len - 1

    def __getitem__(self, idx):
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')

                self.img = self.dataset.get('train_imgs')
                self.token_type_ids = self.dataset.get('train_token_type_ids')
                self.attention_mask = self.dataset.get('train_attention_mask')
                self.ocr = self.dataset.get('train_ocrs')
                self.bbox = self.dataset.get('train_bbox')


                self.id = self.dataset.get('train_id')
                self.target = self.dataset.get('train_labels')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')

                self.img = self.dataset.get('validation_imgs')
                self.token_type_ids = self.dataset.get('validation_token_type_ids')
                self.attention_mask = self.dataset.get('validation_attention_mask')
                self.ocr = self.dataset.get('validation_ocrs')
                self.bbox = self.dataset.get('validation_bbox')

                self.target = self.dataset.get('validation_labels')
                self.id = self.dataset.get('validation_id')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')

                self.img = self.dataset.get('test_imgs')
                self.token_type_ids = self.dataset.get('test_token_type_ids')
                self.attention_mask = self.dataset.get('test_attention_mask')
                self.ocr = self.dataset.get('test_ocrs')
                self.bbox = self.dataset.get('test_bbox')

                self.target = self.dataset.get('test_labels')
                self.id = self.dataset.get('test_id')
                self.pad_id = max([len(id) for id in self.id])
        
        
        if idx < len(self.target) - 1:

            img1 = self.img[idx,:,:,:]
            img2 = self.img[idx+1,:,:,:]


            ocr1 = self.ocr[idx, ...]
            ocr1 = torch.Tensor(ocr1).type(torch.LongTensor)
            ocr2 = self.ocr[idx+1, ...]
            ocr2 = torch.Tensor(ocr2).type(torch.LongTensor)

            token_type_ids1 = self.token_type_ids[idx, ...]
            token_type_ids1 = torch.Tensor(token_type_ids1).type(torch.LongTensor)
            token_type_ids2 = self.token_type_ids[idx+1, ...]
            token_type_ids2 = torch.Tensor(token_type_ids2).type(torch.LongTensor)

            attention_mask1 = self.attention_mask[idx, ...]
            attention_mask1 = torch.Tensor(attention_mask1).type(torch.LongTensor)
            attention_mask2 = self.attention_mask[idx+1, ...]
            attention_mask2 = torch.Tensor(attention_mask2).type(torch.LongTensor)

            bbox1 = self.bbox[idx, ...]
            bbox1 = torch.Tensor(bbox1).type(torch.LongTensor)
            bbox2 = self.bbox[idx+1, ...]
            bbox2 = torch.Tensor(bbox2).type(torch.LongTensor)


            label1 = self.target[idx]
            label2 = self.target[idx+1]
            label = max(label1, label2)


            label = torch.tensor(data=label, dtype=torch.float)



            id1 = self.id[idx]
            id1 = [ord(c) for c in id1]
            id1 = id1 + [-1 for _ in range(self.pad_id - len(id1))]
            id1 = torch.tensor(data=id1, dtype=torch.int)

            id2 = self.id[idx+1]
            id2 = [ord(c) for c in id2]
            id2 = id2 + [-1 for _ in range(self.pad_id - len(id2))]
            id2 = torch.tensor(data=id2, dtype=torch.int)



            batch = {
                        'image1': img1, 'image2': img2, 
                        'tokens1': ocr1, 'tokens2': ocr2, 
                        'token_type_ids1': token_type_ids1, 'token_type_ids2': token_type_ids2, 
                        'attention_mask1': attention_mask1, 'attention_mask2': attention_mask2,
                        'box1': bbox1, 'box2': bbox2, 
                        'label': torch.unsqueeze(label, 0), 
                        'id1': id1, 'id2': id2
                    }
            return batch


class LayouLMV2_Three_OffLine_Dataset(Dataset):

    def __init__(self, path, phase, metadata):
        self.file_path = path
        self.dataset = None
        self.target = None
        self.phase = phase
        self.metadata = metadata
        with h5py.File(self.file_path, 'r') as file:
            if phase == 'train':
                self.dataset_len = len(file["train_imgs"])
            elif phase == 'val':
                self.dataset_len = len(file["validation_imgs"])
            elif phase == 'test':
                self.dataset_len = len(file["test_imgs"])


    def __len__(self):
        return self.dataset_len - 2

    def __getitem__(self, idx):
        idx += 1
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = h5py.File(self.file_path, 'r')

                self.img = self.dataset.get('train_imgs')
                self.token_type_ids = self.dataset.get('train_token_type_ids')
                self.attention_mask = self.dataset.get('train_attention_mask')
                self.ocr = self.dataset.get('train_ocrs')
                self.bbox = self.dataset.get('train_bbox')


                self.id = self.dataset.get('train_id')
                self.target = self.dataset.get('train_labels')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'val':
                self.dataset = h5py.File(self.file_path, 'r')

                self.img = self.dataset.get('validation_imgs')
                self.token_type_ids = self.dataset.get('validation_token_type_ids')
                self.attention_mask = self.dataset.get('validation_attention_mask')
                self.ocr = self.dataset.get('validation_ocrs')
                self.bbox = self.dataset.get('validation_bbox')

                self.target = self.dataset.get('validation_labels')
                self.id = self.dataset.get('validation_id')
                self.pad_id = max([len(id) for id in self.id])
            elif self.phase == 'test':
                self.dataset = h5py.File(self.file_path, 'r')

                self.img = self.dataset.get('test_imgs')
                self.token_type_ids = self.dataset.get('test_token_type_ids')
                self.attention_mask = self.dataset.get('test_attention_mask')
                self.ocr = self.dataset.get('test_ocrs')
                self.bbox = self.dataset.get('test_bbox')

                self.target = self.dataset.get('test_labels')
                self.id = self.dataset.get('test_id')
                self.pad_id = max([len(id) for id in self.id])
        
        
        if idx < len(self.target) - 1:

            img1 = self.img[idx-1,:,:,:]
            img2 = self.img[idx,:,:,:]
            img3 = self.img[idx+1,:,:,:]


            ocr1 = self.ocr[idx-1, ...]
            ocr1 = torch.Tensor(ocr1).type(torch.LongTensor)
            ocr2 = self.ocr[idx, ...]
            ocr2 = torch.Tensor(ocr2).type(torch.LongTensor)
            ocr3 = self.ocr[idx+1, ...]
            ocr3 = torch.Tensor(ocr3).type(torch.LongTensor)

            token_type_ids1 = self.token_type_ids[idx-1, ...]
            token_type_ids1 = torch.Tensor(token_type_ids1).type(torch.LongTensor)
            token_type_ids2 = self.token_type_ids[idx, ...]
            token_type_ids2 = torch.Tensor(token_type_ids2).type(torch.LongTensor)
            token_type_ids3 = self.token_type_ids[idx+1, ...]
            token_type_ids3 = torch.Tensor(token_type_ids3).type(torch.LongTensor)

            attention_mask1 = self.attention_mask[idx-1, ...]
            attention_mask1 = torch.Tensor(attention_mask1).type(torch.LongTensor)
            attention_mask2 = self.attention_mask[idx, ...]
            attention_mask2 = torch.Tensor(attention_mask2).type(torch.LongTensor)
            attention_mask3 = self.attention_mask[idx+1, ...]
            attention_mask3 = torch.Tensor(attention_mask3).type(torch.LongTensor)

            bbox1 = self.bbox[idx-1, ...]
            bbox1 = torch.Tensor(bbox1).type(torch.LongTensor)
            bbox2 = self.bbox[idx, ...]
            bbox2 = torch.Tensor(bbox2).type(torch.LongTensor)
            bbox3 = self.bbox[idx+1, ...]
            bbox3 = torch.Tensor(bbox3).type(torch.LongTensor)


            label1 = self.target[idx-1]
            label2 = self.target[idx]
            label3 = self.target[idx+1]
            label = max(label3, max(label1, label2))


            label = torch.tensor(data=label, dtype=torch.float)



            id1 = self.id[idx-1]
            id1 = [ord(c) for c in id1]
            id1 = id1 + [-1 for _ in range(self.pad_id - len(id1))]
            id1 = torch.tensor(data=id1, dtype=torch.int)

            id2 = self.id[idx]
            id2 = [ord(c) for c in id2]
            id2 = id2 + [-1 for _ in range(self.pad_id - len(id2))]
            id2 = torch.tensor(data=id2, dtype=torch.int)

            id3 = self.id[idx+1]
            id3 = [ord(c) for c in id3]
            id3 = id3 + [-1 for _ in range(self.pad_id - len(id3))]
            id3 = torch.tensor(data=id3, dtype=torch.int)



            batch = {
                        'image1': img1, 'image2': img2, 'image3': img3,
                        'tokens1': ocr1, 'tokens2': ocr2, 'tokens3': ocr3, 
                        'token_type_ids1': token_type_ids1, 'token_type_ids2': token_type_ids2,'token_type_ids3': token_type_ids3, 
                        'attention_mask1': attention_mask1, 'attention_mask2': attention_mask2, 'attention_mask3': attention_mask3,
                        'box1': bbox1, 'box2': bbox2, 'box3': bbox3, 
                        'label': torch.unsqueeze(label, 0), 
                        'id1': id1, 'id2': id2, 'id3': id3
                    }
            return batch


class LayouLMV2_OnLine_Dataset(Dataset):
    def __init__(self, phase, metadata):
        self.phase = phase
        self.metadata = metadata
        self.dataset = None
        
        feature_extractor = LayoutLMv2FeatureExtractor()
        self.tokenizer = LayoutLMv2TokenizerFast.from_pretrained("./tokenizer_LMV2")
        self.processor = LayoutLMv2Processor(feature_extractor, self.tokenizer)


        self.root_path = os.sep.join(os.getcwd().split(os.sep)[:-1])
        dictionary_path = os.path.join(self.root_path, 'CreateDataset', 'TobaccoOcr')
        all_dictionaries = [f for f in listdir(dictionary_path) if isfile(join(dictionary_path, f))]

        self.load_dictionaries(all_dictionaries, dictionary_path)
        self.creation_train_val_test_basedOnCSV()

        if self.phase == 'train':
            self.dataset_len = len(self.train)
        elif self.phase == 'val':
            self.dataset_len = len(self.validation)
        elif self.phase == 'test':
            self.dataset_len = len(self.test)
        
        

    
    def creation_train_val_test_basedOnCSV(self):
        df_train = pd.read_csv(os.path.join(self.root_path, 'CreateDataset', 'train.csv'), header=None)
        df_train.columns = ['img_dir', 'number']
        df_val = pd.read_csv(os.path.join(self.root_path, 'CreateDataset', 'val.csv'), header=None)
        df_val.columns = ['img_dir', 'number']
        df_test = pd.read_csv(os.path.join(self.root_path, 'CreateDataset', 'test.csv'), header=None, sep=' ')
        df_test.columns = ['img_dir', 'number']

        def create_portion_dataset(df):
            base_path = 'scratch/bsc31/bsc31282/BigTobacco'
            samples = []
            columns = ['img_dir']
            keys = self.storage.keys()
            for doc in df['img_dir']:
                if os.path.join(base_path, doc) in keys:
                    samples.append([self.storage[os.path.join(base_path, doc)][0]])

            df = pd.DataFrame(samples, columns=columns)
            return df
        
        def from_documents_to_pages(df):
            all_documents = df['img_dir'].tolist()
            all_pages = []
            all_labels = []
            for doc in all_documents:
                for i, page in enumerate(doc):
                    all_pages.append(page)
                    if i == 0:
                        all_labels.append(1)
                    else:
                        all_labels.append(0)
            
            return all_pages, all_labels
    
        self.train = create_portion_dataset(df_train)
        self.train, self.train_labels = from_documents_to_pages(self.train)
        
        self.validation = create_portion_dataset(df_val)
        self.validation, self.val_labels = from_documents_to_pages(self.validation)

        self.test = create_portion_dataset(df_test)
        self.test, self.test_labels = from_documents_to_pages(self.test)

    
    def load_dictionaries(self, all_dictionaries, dictionary_path):
        self.storage = {}

        for dictionary in all_dictionaries:
            if dictionary[0] != '.':
                with open(os.path.join(dictionary_path, dictionary), 'rb') as fp:
                    print('Loading', dictionary + '....')
                    self.storage = {**self.storage, **pickle.load(fp)}
                    print(dictionary, 'loaded')


    def __len__(self):
        return self.dataset_len - 1


    def __getitem__(self, idx):
        if self.dataset is None:
            if self.phase == 'train':
                self.dataset = self.train
                self.labels = self.train_labels
            elif self.phase == 'val':
                self.dataset = self.validation
                self.labels = self.val_labels
            elif self.phase == 'test':
                self.dataset = self.test
                self.labels = self.test_labels
            
            self.pad_id = max([len(n) for n in self.dataset])

        img1 = self.dataset[idx]
        img1 = Image.open(img1).convert("RGB")


        img2 = self.dataset[idx + 1]
        img2 = Image.open(img2).convert("RGB")

        label = max(self.labels[idx], self.labels[idx + 1])


        encoding1 = self.processor(img1, return_tensors='pt', truncation=True, padding="max_length", max_length=512)
        tokens1, token_type_ids1, attention_mask1, box1, image1 = encoding1['input_ids'],  encoding1['token_type_ids'], encoding1['attention_mask'], encoding1['bbox'], encoding1['image']

        encoding2 = self.processor(img2, return_tensors='pt', truncation=True, padding="max_length", max_length=512)
        tokens2, token_type_ids2, attention_mask2, box2, image2 = encoding2['input_ids'],  encoding2['token_type_ids'], encoding2['attention_mask'], encoding2['bbox'], encoding2['image'] 

        
        
        label = torch.tensor(data=label, dtype=torch.float)

        id1 = self.dataset[idx]
        id1 = [ord(c) for c in id1]
        id1 = id1 + [-1 for _ in range(self.pad_id - len(id1))]
        id1 = torch.tensor(data=id1, dtype=torch.int)

        id2 = self.dataset[idx + 1]
        id2 = [ord(c) for c in id2]
        id2 = id2 + [-1 for _ in range(self.pad_id - len(id2))]
        id2 = torch.tensor(data=id2, dtype=torch.int)

        


        batch = {   
                    'tokens1': tokens1[0, :], 'token_type_ids1': token_type_ids1[0, :], 'attention_mask1': attention_mask1[0, :], 'box1': box1[0, :, :] , 'image1': image1[0, :, :, :], 'id1' : id1,
                    'tokens2': tokens2[0, :], 'token_type_ids2': token_type_ids2[0, :], 'attention_mask2': attention_mask2[0, :], 'box2': box2[0, :, :] , 'image2': image2[0, :, :, :], 'id2' : id2,
                    'label': torch.unsqueeze(label, 0)
                }

        return batch




data_transforms_VGG = {
'train': transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((214, 214)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((214, 214)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((214, 214)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}




data_transforms = {
'train': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'val': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
'test': transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}

#data_transforms_VGG = data_transforms

