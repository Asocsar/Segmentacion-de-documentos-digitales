#%%
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import h5py
import sys
from tqdm import trange
import cv2
from PIL import Image
from pathlib import Path
import pickle
import json
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm
import argparse
from statistics import median
from multiprocessing import Pool
from multiprocessing import freeze_support

os.environ["TOKENIZERS_PARALLELISM"] = "false"

path_data = 'TobaccoOcr'
path_data_800 = 'TobaccoOcr800'

train_folder = './train'
validation_folder = './validation'
test_folder = './test'
path_dictionaries = os.path.join(os.getcwd(), 'TobaccoOcr')
path_dictionaries_800 = os.path.join(os.getcwd(), 'TobaccoOcr800')
size = 15000 #Maximum number of examples for each file
padding_layout = 512

def create_json_information_documents_pages(train, validation, test):
    l1 = len( [f for f in list(train['img_dir'].values) if len(f) == 1] )
    l2 = len( [f for f in list(validation['img_dir'].values) if len(f) == 1] )
    l3 = len( [f for f in list(test['img_dir'].values) if len(f) == 1] )

    n_l1 = len( [f for f in list(train['img_dir'].values) if len(f) > 1] )
    n_l2 = len( [f for f in list(validation['img_dir'].values) if len(f) > 1] )
    n_l3 = len( [f for f in list(test['img_dir'].values) if len(f) > 1] )

    print('Train {} single page documents {} multipage documents which is {} %'.format(l1,n_l1,l1/(l1+n_l1)*100))
    print('Test {} single page documents {} multipage documents which is {} %'.format(l3,n_l3,l3/(l3+n_l3)*100))
    print('Validation {} single page documents {} multipage documents which is {} %'.format(l2,n_l2,l2/(l2+n_l2)*100))

    print('Total {} % is unipage'.format((l1+l2+l3)/(l1+l2+l3+n_l1+n_l2+n_l3)*100))

    t = l1 + l2 + l3 + n_l1 + n_l2 + n_l3

    print('Train is {} % of total'.format( (l1+n_l1)/t*100 ) )
    print('Val is {} % of total'.format( (l2+n_l2)/t*100 ) )
    print('Test is {} % of total'.format( (l3+n_l3)/t*100 ) )


    l1 =  [f for f in list(train['img_dir'].values)] 
    l2 =  [f for f in list(validation['img_dir'].values)] 
    l3 =  [f for f in list(test['img_dir'].values)]


    long_train = {}
    for e in l1:
        longitude = len(e)
        if longitude in long_train.keys():
            long_train[longitude] += 1
        else:
            long_train[longitude] = 1


    long_val = {}
    for e in l2:
        longitude = len(e)
        if longitude in long_val.keys():
            long_val[longitude] += 1
        else:
            long_val[longitude] = 1


    long_test = {}
    for e in l2:
        longitude = len(e)
        if longitude in long_test.keys():
            long_test[longitude] += 1
        else:
            long_test[longitude] = 1

    long_train = {k: v for k, v in sorted(long_train.items(), key=lambda item: item[0])}
    long_val = {k: v for k, v in sorted(long_val.items(), key=lambda item: item[0])}
    long_test = {k: v for k, v in sorted(long_test.items(), key=lambda item: item[0])}

    all_keys = list(set(list(long_train.keys()) + list(long_val.keys()) + list(long_test.keys())))
    all_keys = sorted(all_keys)

    all_long = {}
    for k in all_keys:
        if k in long_train:
            tr = long_train[k]
        else:
            tr = 0
        
        if k in long_val:
            vl = long_val[k]
        else:
            vl = 0
        
        if k in long_test:
            ts = long_test[k]
        else:
            ts = 0
        all_long[k] = tr + vl + ts


    BigTobaccoInfo = {'Train': long_train, 'Validation': long_val, 'Test': long_test,
                        'AllBigTobacco': all_long}

    with open('BigTobaccoInfo.json', 'w+') as f:
        json.dump(BigTobaccoInfo, f)


def create_image_number_of_documents_for_different_sizes(name, keys, values, filtering=False):
    plt.figure(figsize=(15, 10), dpi=500)
    plt.xticks(size = 10)
    plt.xticks(rotation='vertical')
    plt.yscale('log')
    plt.bar(keys, values)
    if filtering:
        plt.savefig('TobaccoFiltered_' + name + '.png')
    else:
        plt.savefig('Tobacco_No_Filtered_' + name + '.png')


def create_image_distribution_new_document(name, labels, data):
    plt.figure(figsize=(15, 10), dpi=500)
    plt.bar(labels, data)
    plt.savefig('TobaccoFiltered_' + name + '_Proportion' + '.png')


def obtain_information_documents_by_longitude(train):
    l1 =  [f for f in list(train['img_dir'].values)]

    long_train = {}
    for e in l1:
        longitude = len(e)
        if longitude in long_train.keys():
            long_train[longitude] += 1
        else:
            long_train[longitude] = 1

    long_train = {k: v for k, v in sorted(long_train.items(), key=lambda item: item[0])}

    return long_train


def plot_ratios(df, name):
    i = 0
    K_0 = 0
    K_1 = 0
    for _, elem in df.iterrows():
        images_paths = elem[0]
        for i in range(len(images_paths)):

            if i == 0:
                K_1 += 1
                
            else :
                K_0 += 1

    create_image_distribution_new_document(name, ['0', '1'], [(K_0)/(K_0+K_1), (K_1)/(K_0+K_1)])
    

def apply_threshold(df, name, mode, plot_relation=False):
    dictionary = obtain_information_documents_by_longitude(df)
    print(df.columns)
    final_df = None
    print('MODE {}'.format(mode))

    number_documents = sorted(list(dictionary.values()))[::-1]
    number_pages = sum([int(k*v) for k,v in dictionary.items()])
    if mode == 'Head':
        threshold = int(max(sorted(list(dictionary.keys())))*0.9)
        long_train_aux = {k: v for k, v in dictionary.items() if k >= threshold}
    elif mode == 'Tail':
        threshold = int(max(sorted(list(dictionary.keys())))*0.012) #0.0055
        long_train_aux = {k: v for k, v in dictionary.items() if k <= threshold}
        #for i in range(2,max(list(long_train_aux.keys()))+1,1):
        #    long_train_aux[i] = int(long_train_aux[i]/3)
    
    
    all_new_documents_sum = sum(long_train_aux.values())
    all_new_pages_sum = sum([int(k*v) for k,v in long_train_aux.items()])

    print('-------------')
    print('Threshold', threshold)
    print('-------------')
    print('Number of ocurrences for each longitude\n', long_train_aux)
    print('-------------')
    print('Total number of documents remaining {} which is {} % of total'.format(all_new_documents_sum, all_new_documents_sum/sum(number_documents)))
    print('Total number of pages remaining {} which is {} % of total'.format(all_new_pages_sum, all_new_pages_sum/number_pages))
    print('Number of documents with 1 page {} vs with more than 1 page {}'.format(list(long_train_aux.values())[0], sum(list(long_train_aux.values())[1:])))
    print('-------------', end="\n\n\n")

    

    create_image_distribution_new_document('Single_Page_vs_Multiple_Page_' + mode, ['Single page', 'Multiple Page'], [list(long_train_aux.values())[0]/sum(list(long_train_aux.values())), sum(list(long_train_aux.values())[1:]) / sum(list(long_train_aux.values())) ])

    longitudes =  df.iloc[:,0].apply(lambda x: len(x)) 

    for k,v in long_train_aux.items():
        df_aux = df[longitudes == k]
        rows = np.random.choice(df_aux.index.values, v)
        sampled_df = df_aux.loc[rows]

        if final_df is None:
            final_df = sampled_df
        else:
            final_df = pd.concat([final_df, sampled_df])



    if plot_relation:
        print('Generating Relation Images...')
        keys = list(long_train_aux.keys())
        values = list(long_train_aux.values())
        create_image_number_of_documents_for_different_sizes('Plot result of ' + name + '_' + mode, keys, values, filtering=True)
        plot_ratios(final_df, name + '_' + mode)
    

    return final_df


def creation_train_val_test_basedOnCSV(st):
    df_train = pd.read_csv('train.csv', header=None)
    df_train.columns = ['img_dir', 'number']
    df_val = pd.read_csv('val.csv', header=None)
    df_val.columns = ['img_dir', 'number']
    df_test = pd.read_csv('test.csv', header=None, sep=' ')
    df_test.columns = ['img_dir', 'number']

    def create_portion_dataset(df, st):
        base_path = 'scratch/bsc31/bsc31282/BigTobacco'
        samples = []
        columns = ['img_dir', 'ocr']
        keys = st.keys()
        for doc in df['img_dir']:
            if os.path.join(base_path, doc) in keys:
                samples.append(st[os.path.join(base_path, doc)])

        df = pd.DataFrame(samples, columns=columns)
        return df
    
    train = create_portion_dataset(df_train, st)
    
    validation = create_portion_dataset(df_val, st)

    test = create_portion_dataset(df_test, st)

    return train, validation, test


def load_dictionaries(tobacco800):
    if tobacco800:
        all_dictionaries = [f for f in listdir(path_dictionaries_800) if isfile(join(path_dictionaries_800, f))]
        path_selected = path_data_800 
    else:
        all_dictionaries = [f for f in listdir(path_dictionaries) if isfile(join(path_dictionaries, f))]
        path_selected = path_data
    
    if tobacco800:
        storage = []
    else:
        storage = {}



    for dictionary in all_dictionaries:
        if dictionary[0] != '.':
            with open(os.path.join(os.getcwd(), path_selected, dictionary), 'rb') as fp:
                print('Loading', dictionary + '....')
                if tobacco800:
                    storage += pickle.load(fp)
                else:
                    storage = {**storage, **pickle.load(fp)}
                print(dictionary, 'loaded')
    
    return storage


def create_relations(df):
    dataset = []
    i = 0
    K_0 = 0
    K_1 = 0
    for index, elem in df.iterrows():
        images_paths = elem[0]
        ocrs_content = elem[1]
        for i in range(len(images_paths)):

            if i == 0:
                dataset.append([images_paths[i], ocrs_content[i], 1])
                K_1 += 1
                
            else :
                dataset.append([images_paths[i], ocrs_content[i], 0])
                K_0 += 1
    
    return dataset


def creation_train_val_test_Random(st, train=0.8, val=0.1, test=0.1):
    columns = ['img_dir', 'ocr']
    df = pd.DataFrame(st, columns=columns)
    df = df.sample(frac=1).reset_index(drop=True)



    test = train + val

    train, validation, test = np.split(df.sample(frac=1).reset_index(drop=True), [int(train*len(df)), int(test*len(df))])

    return train, validation, test


def creation_test_800(st):
    columns = ['img_dir', 'ocr']
    df = pd.DataFrame(st, columns=columns)
    print(df)
    test = df.sample(frac=1).reset_index(drop=True)


    return test


def create_h5df_files(df, section, img_size, tobacco800=False, filtered=False, tobacco800_split=False):
    columns = ['img_dirs', 'ocrs', 'labels']
    df = pd.DataFrame(df, columns=columns)
    base_path = os.sep.join(os.getcwd().split(os.sep)[:-1])


    
    list_of_dfs = [df.loc[i:i+size-1,:] for i in range(0, len(df),size)]
    
    #ADD FIRST ELEMENT OF THE NEXT FILE AS THE LAST ELEMENT OF THE ACTUAL FILE
    for i in range(len(list_of_dfs)-1):
        list_of_dfs[i] = pd.concat([list_of_dfs[i], pd.DataFrame(list_of_dfs[i+1].iloc[0, :].to_dict(), columns=list_of_dfs[i].columns, index=[len(list_of_dfs[i])])], ignore_index=True, axis=0)

    for i, sub_df in enumerate(list_of_dfs):
        name_file = str(section) + '_' + str(i) + '.hdf5'
        if tobacco800:
            if filtered:
                Path(os.path.join(base_path, 'Tobacco800_filtered', section)).mkdir(parents=True, exist_ok=True)
                hdf5_path = os.path.join(base_path, 'Tobacco800_filtered', section, name_file)
            elif tobacco800_split:
                Path(os.path.join(base_path, 'Tobacco800_split', section)).mkdir(parents=True, exist_ok=True)
                hdf5_path = os.path.join(base_path, 'Tobacco800_split', section, name_file)
            else:
                Path(os.path.join(base_path, 'Tobacco800', section)).mkdir(parents=True, exist_ok=True)
                hdf5_path = os.path.join(base_path, 'Tobacco800', section, name_file)
        else:
            if filtered:
                Path(os.path.join(base_path, 'BigTobacco_filtered', section)).mkdir(parents=True, exist_ok=True)
                hdf5_path = os.path.join(base_path, 'BigTobacco_filtered', section, name_file)
            else:
                Path(os.path.join(base_path, 'BigTobacco', section)).mkdir(parents=True, exist_ok=True)
                hdf5_path = os.path.join(base_path, 'BigTobacco', section, name_file)

        hdf5_file = h5py.File(hdf5_path, mode='w')
        dt = h5py.special_dtype(vlen=str)

        image_datatype = h5py.h5t.STD_U8BE
        sub_df_shape = (len(sub_df),  img_size, img_size, 3)

        hdf5_file.create_dataset(section + "_id", (len(sub_df),), dtype=dt)
        
        hdf5_file.create_dataset(section + "_imgs", sub_df_shape, image_datatype)
        hdf5_file.create_dataset(section + "_ocrs", (len(sub_df),), dtype=dt)
        hdf5_file[section + "_ocrs"][...] = sub_df['ocrs'].values
        hdf5_file.create_dataset(section + "_labels", (len(sub_df),), np.int8)
        hdf5_file[section + "_labels"][...] = sub_df['labels'].values



        def load_image(df, hdf5_file, tag, img_size):
            image_paths1 = df["img_dirs"].values

            pbar = trange(len(df), desc='Processing ' + tag, leave=True)
            
            
            for i in pbar:
                #image_paths1[i] = os.path.join(os.sep.join(os.getcwd().split(os.sep)[:6]), os.sep.join(image_paths1[i].split(os.sep)[5:]))
                im1 = cv2.imread(image_paths1[i])
                im1 = cv2.resize(im1, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
                hdf5_file[tag + "_imgs"][i,...] = im1[None]
                hdf5_file[tag + "_id"][i,...] = image_paths1[i]


        load_image(sub_df, hdf5_file, section, img_size)

        hdf5_file.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtering", type=bool,
                    help="Indicate if we are filtering", default=False)
    parser.add_argument("--visualize_data", type=bool,
                    help="Indicate if we want to create plots", default=False)
    parser.add_argument("--create_json_information", type=bool,
                    help="Indicate if we want to create json with information", default=False)
    parser.add_argument("--mode", type=str,
                    help="Indicate mode of filtering which can be Head or Tail", choices=['Head', 'Tail'])
    parser.add_argument("--tobacco800", type=bool,
                    help="Indicate that we are creating Tobacco800 h5df files", default=False)
    parser.add_argument("--splitTobacco800", type=bool,
                    help="Indicate that we are dividing tobaccco800 into Train, validation and Test", default=False)
    parser.add_argument("--trainT800", type=float,
                    help="Indicate proportion of training for Tobacco800")
    parser.add_argument("--valT800", type=float,
                    help="Indicate proportion of validation for Tobacco800")
    parser.add_argument("--testT800", type=float,
                    help="Indicate proportion of test for Tobacco800")
    
    args = parser.parse_args()

    mode = args.mode
    if mode is not None:
        print('Mode', mode)
    storage = load_dictionaries(args.tobacco800)
    
    if args.tobacco800 and args.splitTobacco800:
        if args.valT800 is None or args.valT800 is None or args.testT800 is None:
            train, validation, test = creation_train_val_test_Random(storage)
        else:
            train, validation, test = creation_train_val_test_Random(storage, 
                                                                    args.trainT800, 
                                                                    args.valT800, 
                                                                    args.testT800)
    
    elif args.tobacco800:
        test = creation_test_800(storage)
    
    else:
        train, validation, test = creation_train_val_test_basedOnCSV(storage)
    if not args.filtering and args.visualize_data:
        dictionary = obtain_information_documents_by_longitude(train)
        create_image_number_of_documents_for_different_sizes('train', 
                                                                list(dictionary.keys()), 
                                                                list(dictionary.values()))

    if args.create_json_information:
        create_json_information_documents_pages(train, validation, test)
    if args.filtering:
        train = apply_threshold(train, name='train', mode=mode, plot_relation=args.visualize_data)

    if args.tobacco800 and not args.splitTobacco800:
        all_df = [('test', test)]
    else:
        all_df = [('train', train), ('validation', validation), ('test', test)]
    img_size = 512
    for section, df in all_df:
        df = create_relations(df)
        create_h5df_files(df, section, img_size, tobacco800=args.tobacco800, filtered=args.filtering, tobacco800_split=args.splitTobacco800)







