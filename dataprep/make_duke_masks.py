import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.patches as patches
import pickle
import torch.nn.functional as F
import torch

datasets = ['DUKE']
data_path = 'data/'

for i3, dataset in enumerate(datasets):
    path = os.path.join(data_path, dataset)
    
    images_path = os.path.join(path, 'images')
    images_files = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])

    boxes_path = os.path.join(path, 'boxes')
    masks_output_path = os.path.join(path, 'masks')
    boxes_files = sorted([f for f in os.listdir(boxes_path) if os.path.isfile(os.path.join(boxes_path, f))])
    
    seqs, masks, image_output_paths = [], [], []
    masks_files = boxes_files
    for image_file, mask_file in tqdm(zip(images_files, masks_files)):
        seq_path = os.path.join(images_path, image_file)
        seq = np.load(seq_path)
        original_shape = seq.shape
        seqs.append(seq)
        box_path = os.path.join(boxes_path, mask_file)
        with open(box_path, 'r') as file:
            box = json.load(file)
        start_row, end_row = box['Start Row'], box['End Row']
        start_column, end_column = box['Start Column'], box['End Column']
        start_slice, end_slice = box['Start Slice'], box['End Slice']
        seq_masks = np.zeros(original_shape)
        seq_masks[start_slice:end_slice+1, :, start_row:end_row+1, start_column:end_column+1] = 1
        masks.append(seq_masks.astype(np.float32))

        mask_output_path = os.path.join(masks_output_path, image_file)
        if not os.path.exists(masks_output_path):
            os.makedirs(masks_output_path)
        np.save(mask_output_path, seq_masks)

    # for i, seq in enumerate(seqs):
    #     if i > 30:
    #         break
    #     i2 = len(seq) // 2
    #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #     axs[0].imshow(seq[i2].squeeze(), cmap='gray')
    #     axs[0].set_title(f'Image')
    #     axs[1].imshow(masks[i][i2].squeeze(), cmap='gray')
    #     axs[1].set_title('Grayscale Mask')
    #     plt.savefig(f'outputs/{dataset}_{i}_mask.png')
    #     plt.close()