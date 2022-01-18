import os
from os import walk
import sys
import pickle
from os import listdir
from os.path import isfile, join
from pathlib import Path

mypath = '/gpfs/scratch/bsc31/bsc31168/Document_Segmentation/Tobacco800' #'/gpfs/scratch/bsc31/bsc31282/BigTobacco' #Path to search for all images

### OBTAIN ALL NAMES OF FILENAMES ORDERED AND GROUPED BY DOCUMENT ###
def documents_search(path, tobacco800):
    images_dirs = []

    if tobacco800:
        with open(os.path.join(mypath, 'files.txt'), 'r') as f:
            data = f.read()
        
        images_dirs = [os.path.join(mypath, 'images', f) for f in data.split('\n') if len(f) > 0]
        groups = {}
        for document in images_dirs:
            groups.setdefault(os.path.splitext(document)[0][:-2],[]).append(document)
        images_dirs = list(groups.values())
        return images_dirs

    for (dirpath, dirnames, filenames) in walk(path):
        document_images = []
        filenames = [f for f in filenames if os.path.splitext(f)[1] == '.png' and f[0] != '.']
        if len(filenames) > 1:
            filenames = sorted(filenames, key = lambda fil: int(os.path.splitext(fil)[0].split('_')[-1]))
        for file in filenames:
            if os.path.splitext(file)[1] == '.png':
                document_images.append(os.path.join(dirpath,file))
        if (len(document_images) > 0):
            images_dirs.append(document_images)
    
    return images_dirs


#%%
if __name__ == "__main__":
    tobacco800 = int(sys.argv[2])
    all_documents = documents_search(mypath, tobacco800)
    print('Number of documents', len(all_documents))
    number_sublist = int(sys.argv[1])
    longitude_each_sublist = int(len(all_documents)/number_sublist)
    sub_list = [all_documents[x:x+longitude_each_sublist] for x in range(0, len(all_documents), longitude_each_sublist)]
    if len(sub_list) == 5:
        sub_list[3] = sub_list[3] + sub_list[4]
    Path(os.path.join(os.getcwd(), 'sub_lists_tobacco800')).mkdir(parents=True, exist_ok=True)
    for i, sub in enumerate(sub_list[:4]):
        print('Sublist', i, 'contains', len(sub), 'elements')
        with open(os.path.join(os.getcwd(), 'sub_lists_tobacco800', "lista_" + str(i) + ".txt"), "wb+") as fp:
            pickle.dump(sub, fp)