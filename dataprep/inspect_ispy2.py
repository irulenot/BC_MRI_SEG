import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.patches as patches
import pickle
import torch.nn.functional as F
import torch
from sklearn.preprocessing import MinMaxScaler


datasets = ['ISPY2']
data_path = 'data/'

for i3, dataset in enumerate(datasets):
    path = os.path.join(data_path, dataset)
    
    images_path = os.path.join(path, 'images')
    images_output_path = os.path.join(path, 'images_std3')
    images_files = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])

    masks_path = os.path.join(path, 'masks')
    masks_output_path = os.path.join(path, 'masks_std3')
    masks_files = sorted([f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))])
    
    # counter = 0
    # for image_file, mask_file in tqdm(zip(images_files, masks_files)):
    #     seq_path = os.path.join(images_path, image_file)
    #     seq = np.load(seq_path)

    #     seq_masks_path = os.path.join(masks_path, mask_file)
    #     seq_masks = np.load(seq_masks_path)

    #     if seq.shape[1] == 2:
    #         seq = seq[:, 0].reshape(seq.shape[0], 1, seq.shape[2], seq.shape[3])
    #         seq_masks = seq_masks[:, 0].reshape(seq_masks.shape[0], 1, seq_masks.shape[2], seq_masks.shape[3])

    #     idxs = [0, 1, 2, -3, -2, -1]
    #     sums_per_image = np.sum(seq_masks, axis=(2, 3))
    #     index_of_max_sum = np.argmax(sums_per_image)
    #     additional_idxs = [index_of_max_sum-1, index_of_max_sum, index_of_max_sum+1]

    #     fig, axs = plt.subplots(2, 3)
    #     for i in range(6):
    #         idx = idxs[i]
    #         image = seq[idx]
    #         x = i // 3
    #         y = i % 3
    #         axs[x][y].imshow(image.squeeze(), cmap='gray')
    #     plt.savefig(f'outputs/{dataset}_{counter}_0.png')
    #     plt.tight_layout()
    #     plt.close()

    #     fig, axs = plt.subplots(1, 6, figsize=(25, 5))
    #     for i in range(3):
    #         idx = additional_idxs[i]
    #         image = seq[idx]
    #         mask = seq_masks[idx]
    #         axs[i*2].imshow(image.squeeze(), cmap='gray')
    #         axs[i*2+1].imshow(mask.squeeze(), cmap='gray')
    #     plt.savefig(f'outputs/{dataset}_{counter}_1.png')
    #     plt.tight_layout()
    #     plt.close()

    #     counter += 1
    #     if counter == 20:
    #         break

    counter = 0
    for image_file, mask_file in tqdm(zip(images_files, masks_files)):
        seq_path = os.path.join(images_path, image_file)
        seq = np.load(seq_path)

        seq_masks_path = os.path.join(masks_path, mask_file)
        seq_masks = np.load(seq_masks_path)

        if seq.shape[1] == 1:
            continue

        idxs = [0, 1, 2, -3, -2, -1]
        sums_per_image = np.sum(seq_masks, axis=(2, 3))
        index_of_max_sum = np.argmax(sums_per_image, axis=0)
        additional_idxs = [[index_of_max_sum[0]-1, index_of_max_sum[0], index_of_max_sum[0]+1],
                           [index_of_max_sum[1]-1, index_of_max_sum[1], index_of_max_sum[1]+1]]

        fig, axs = plt.subplots(2, 6, figsize=(75, 25))
        for i2 in range(2):
            for i in range(6):
                idx = idxs[i]
                image = seq[idx][i2]
                axs[i2][i].imshow(image.squeeze(), cmap='gray')
        plt.savefig(f'outputs/{dataset}_{counter}_0.png')
        print(f'outputs/{dataset}_{counter}_0.png')
        plt.tight_layout()
        plt.close()

        fig, axs = plt.subplots(2, 6, figsize=(75, 25))
        for i2 in range(2):
            for i in range(3):
                idx = additional_idxs[i2][i]
                image = seq[idx][i2]
                mask = seq_masks[idx][i2]
                axs[i2][i*2].imshow(image.squeeze(), cmap='gray')
                axs[i2][i*2+1].imshow(mask.squeeze(), cmap='gray')
        plt.savefig(f'outputs/{dataset}_{counter}_1.png')
        print(f'outputs/{dataset}_{counter}_1.png')
        plt.tight_layout()
        plt.close()

        counter += 1
        if counter == 20:
            break