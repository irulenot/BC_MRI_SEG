import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import walk

curr_idx = 0
data_path = 'data/BreastDM/seg3D/'
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
        np.save(f'data/BreastDM/images/dm_{curr_idx}.npy', seqs)
        np.save(f'data/BreastDM/masks/dm_{curr_idx}.npy', labels)
        curr_idx += 1