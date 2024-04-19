# Dataset: https://github.com/smallboy-code/Breast-cancer-dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import walk
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.patches as patches
import pickle
import torch.nn.functional as F
import torch

curr_idx = 0
data_path = 'full_data/BreastDM/seg3D/'
output_path = 'data/BreastDM/'
os.makedirs(output_path, exist_ok=True)
os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

for dirpath, dirnames, filenames in tqdm(walk(data_path)):
    if dirnames == [] and filenames != [] and 'labels' in dirpath:
        images_path = dirpath.replace('labels', 'images')
        if not os.path.exists(images_path):
            print(f"{images_path} doesn't exist.")
            continue
        seqs, labels = [], []
        for i, filename in enumerate(filenames):
            label = np.load(os.path.join(dirpath, filename)).transpose(-1, 0, 1)
            imgs = np.load(os.path.join(images_path, filename)).transpose(-1, 0, 1)
            to_delete = []
            for i2, img in enumerate(imgs):
                if np.sum(img) == 0:
                    to_delete.append(i2)
            imgs = np.delete(imgs, to_delete, axis=0)
            label = np.delete(label, to_delete, axis=0)
            labels.append(label)
            seqs.append(imgs)
        seqs, labels = np.array(seqs), np.array(labels)
        seqs, labels = seqs.transpose(1, 0, 2, 3), labels.transpose(1, 0, 2, 3)
        np.save(output_path + f'images/dm_{curr_idx}.npy', seqs)
        np.save(output_path + f'masks/dm_{curr_idx}.npy', labels)
        curr_idx += 1