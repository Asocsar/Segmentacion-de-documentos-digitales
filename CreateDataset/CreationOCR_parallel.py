#%%
from PIL import Image, ImageFilter
from pandas._config.config import set_option
import pytesseract
import os
from os import walk
from tqdm import tqdm
import subprocess
import sys
import numpy as np
from pathlib import Path
import pickle
from datetime import timedelta
import time
from tqdm.contrib.concurrent import process_map, thread_map
from functools import partial
from multiprocessing import Pool
from multiprocessing import freeze_support
import argparse
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2TokenizerFast, LayoutLMv2Processor

os.environ["TOKENIZERS_PARALLELISM"] = "true"

path_ocrs = 'TobaccoOcr'
path_ocrs_800 = 'TobaccoOcr800'
path_Layout = 'TobaccoLayout'
path_Layout800 = 'TobaccoLayout800'

feature_extractor = LayoutLMv2FeatureExtractor()
tokenizer = LayoutLMv2TokenizerFast.from_pretrained("../Code/tokenizer_LMV2")
processor = LayoutLMv2Processor(feature_extractor, tokenizer)

### GENERATE OCR FROM IMAGE ###
def create_OCR(documents, config, LayoutLM):

    result_text = []
    result_name = []

    result_tokens = []
    result_token_type_ids = []
    result_attention_mask = []
    result_box = []
    result_image = []

    for file in documents:
        if not LayoutLM:
            text = pytesseract.image_to_string(file, config=config)

            result_text.append(text)
            result_name.append(file)
        
        else:
            im1 = Image.open(file).convert("RGB")
            encoding = processor(im1, return_tensors='pt', truncation=True, padding="max_length", max_length=512)

            for k,v in encoding.items():
                    encoding[k] = v[0, ...].numpy()
            
            tokens, token_type_ids, attention_mask, box, image = encoding['input_ids'],  encoding['token_type_ids'], encoding['attention_mask'], encoding['bbox'], encoding['image']
    
            result_name.append(file)
            result_tokens.append(tokens)
            result_token_type_ids.append(token_type_ids)
            result_attention_mask.append(attention_mask)
            result_box.append(box)
            result_image.append(image)


    if not LayoutLM:
        return (result_name, result_text)
    else:
        return (result_name, result_tokens, result_token_type_ids, result_attention_mask, result_box, result_image)


def run_multiprocessing(func, documents, n_processors):
    ocrs = []
    img_dirs = []
    max = len(documents)
    with Pool(processes=n_processors) as pool:
        with tqdm(total=max) as pbar:
            for i, (r_name, r_text) in enumerate(pool.imap_unordered(func, documents)):
                img_dirs.append(r_name)
                ocrs.append(r_text)
                pbar.update()
    return img_dirs, ocrs


def run_multiprocessing_Layout(func, documents, n_processors):
    img_dirs = []
    tokens = []
    token_type_ids = []
    attention_masks = []
    boxs = []
    images = []
    max = len(documents)
    with Pool(processes=n_processors) as pool:
        with tqdm(total=max) as pbar:
            for i, (result_name, result_tokens, result_token_type_ids, result_attention_mask, result_box, result_image) in enumerate(pool.imap_unordered(func, documents)):
                img_dirs.append(result_name)
                tokens.append(result_tokens)
                token_type_ids.append(result_token_type_ids)
                attention_masks.append(result_attention_mask)
                boxs.append(result_box)
                #images.append(result_image)
                pbar.update()
    return img_dirs, tokens, token_type_ids, attention_masks, boxs, images


def main(documents, id, BigTobacco, tobacco800, LayoutLM, LayoutLM_Tobacco800):
    process = 32
    base_path = os.getcwd()
    start_execution = time.time()
    print('Procesos:', process)
    config = ('tesseract image.jpg output -l eng --oem 1 --psm 3')
    create_OCR_partial = partial(create_OCR, config=config, LayoutLM=(LayoutLM or LayoutLM_Tobacco800))


    def fixed(l1):
        path = os.sep.join(l1[0].split(os.sep)[2:])
        if len(l1) > 1:
            name, ext = os.path.splitext(path)
            name = name[:-2]
            ext = '.tif'
            return name + ext

        return path[:-3] + 'tif'


    if not (LayoutLM or LayoutLM_Tobacco800):
        img, ocrs = run_multiprocessing(create_OCR_partial, documents, process)
        df = [[e1, e2] for (e1,e2) in zip(img, ocrs)]
        storage = {fixed(l1): [l1,l2] for l1,l2 in df}
    else:
        img_dirs, tokens, token_type_ids, attention_masks, boxs, _ = run_multiprocessing_Layout(create_OCR_partial, documents, process)
        df = [[e1,e2,e3,e4,e5] for (e1,e2,e3,e4,e5) in zip(img_dirs, tokens, token_type_ids, attention_masks, boxs)]
        storage = {fixed(l1): [l1,l2,l3,l4,l5] for l1,l2,l3,l4,l5 in df}
    
    

    if BigTobacco:
        Path(os.path.join(base_path, path_ocrs)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(base_path, path_ocrs, 'dictionary' + id + '.json'), 'wb+') as fp:
            pickle.dump(storage, fp)

    elif tobacco800:
        Path(os.path.join(base_path, path_ocrs_800)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(base_path, path_ocrs_800, 'dictionary' + id + '.json'), 'wb+') as fp:
            pickle.dump(df, fp)
    
    elif LayoutLM:
        Path(os.path.join(base_path, path_Layout)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(base_path, path_Layout, 'dictionary' + id + '.json'), 'wb+') as fp:
            pickle.dump(storage, fp)
    
    elif LayoutLM_Tobacco800:
        Path(os.path.join(base_path, path_Layout800)).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(base_path, path_Layout800, 'dictionary' + id + '.json'), 'wb+') as fp:
            pickle.dump(storage, fp)
    
    end_execution = time.time()
    elapsed_time = end_execution - start_execution
    elapsed_time = timedelta(seconds=elapsed_time)
    print("Elapsed Time", elapsed_time)



#%%
if __name__ == "__main__":
    omp_threads = '2'
    print('OMP_THREAD_LIMIT', omp_threads)
    os.environ['OMP_THREAD_LIMIT'] = omp_threads
    start_execution = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--iden", type=str,
                    help="Select list to load")
    parser.add_argument("--BigTobacco", type=bool,
                    help="Indicate that we are creating BigTobacco Data", default=False)
    parser.add_argument("--tobacco800", type=bool,
                    help="Indicate that we are creating Tobacco800 OCR", default=False)
    parser.add_argument("--LayoutLM", type=bool,
                    help="Indicate that we are creating LayoutLM Data", default=False)
    parser.add_argument("--LayoutLM_Tobacco800", type=bool,
                    help="Indicate that we are creating LayoutLM Data", default=False)
    
    args = parser.parse_args()

    if args.tobacco800:
        with open(os.path.join(os.getcwd(), 'sub_lists_tobacco800', "lista_" + args.iden + ".txt"), "rb") as fp:
            documents = pickle.load(fp)
    elif args.BigTobacco:
        with open(os.path.join(os.getcwd(), 'sub_lists', "lista_" + args.iden + ".txt"), "rb") as fp:
            documents = pickle.load(fp)
    elif args.LayoutLM:
        with open(os.path.join(os.getcwd(), 'sub_lists', "lista_" + args.iden + ".txt"), "rb") as fp:
            documents = pickle.load(fp)
    elif args.LayoutLM_Tobacco800:
        with open(os.path.join(os.getcwd(), 'sub_lists_tobacco800', "lista_" + args.iden + ".txt"), "rb") as fp:
            documents = pickle.load(fp)

    freeze_support() 
    print(len(documents))
    main(documents, args.iden, args.BigTobacco, args.tobacco800, args.LayoutLM, args.LayoutLM_Tobacco800)
    end_execution = time.time()
    elapsed_time = end_execution - start_execution
    elapsed_time = timedelta(seconds=elapsed_time)
    print("Elapsed Time", elapsed_time)