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

standard_shape = (256, 256, 128)
scaler = MinMaxScaler()
for i3, dataset in enumerate(datasets):
    path = os.path.join(data_path, dataset)
    
    images_path = os.path.join(path, 'images')
    images_output_path = os.path.join(path, 'images_std3')
    images_files = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])

    masks_path = os.path.join(path, 'masks')
    masks_output_path = os.path.join(path, 'masks_std3')
    masks_files = sorted([f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))])

    seqs = []
    for image_file in tqdm(images_files):  # Iterate over every other image file for DUKE
        seq_path = os.path.join(images_path, image_file)
        seq = np.load(seq_path)
        seqs.append(seq.flatten())
    seqs = np.concatenate(seqs)
    print(dataset)
    print('mean', np.mean(seqs))
    print('std', np.std(seqs))
    top_percent_indices = int(0.001 * len(seqs))
    sorted_values = np.sort(seqs)
    top_0_1_percent_max = sorted_values[-top_percent_indices:][0]
    top_0_1_percent_min = sorted_values[:top_percent_indices][0]
    print("top_0_1_percent_max", top_0_1_percent_max)
    print("top_0_1_percent_min", top_0_1_percent_min)
    del seqs, sorted_values
    
    # seqs, masks, image_output_paths = [], [], []
    for image_file, mask_file in tqdm(zip(images_files, masks_files)):
        seq_path = os.path.join(images_path, image_file)
        seq = np.load(seq_path)
        original_shape = seq.shape
        # Don't normalize until after
        if dataset == 'ISPY2' and seq.shape[1] == 2:  # ISPY2 has only some multiple angle labels
            seq = seq[:, 0].reshape(seq.shape[0], 1, seq.shape[2], seq.shape[3])
        # seq = np.clip(seq, top_0_1_percent_min, top_0_1_percent_max)
        seq = torch.tensor(seq.transpose(1, 2, 3, 0).astype(np.float32))
        seq = seq.reshape(1, seq.shape[0], seq.shape[1], seq.shape[2], seq.shape[3])
        seq = F.interpolate(seq, size=standard_shape, mode='trilinear', align_corners=False)
        seq = seq.squeeze(0).numpy().transpose(-1, 0, 1, 2)

        seq_masks_path = os.path.join(masks_path, mask_file)
        seq_masks = np.load(seq_masks_path)
        if dataset == 'ISPY2' and seq_masks.shape[1] == 2:
            seq_masks = seq_masks[:, 0].reshape(seq_masks.shape[0], 1, seq_masks.shape[2], seq_masks.shape[3])
        seq_masks = torch.tensor(seq_masks.transpose(1, 2, 3, 0).astype(np.float32))
        seq_masks = seq_masks.reshape(1, seq_masks.shape[0], seq_masks.shape[1], seq_masks.shape[2], seq_masks.shape[3])
        seq_masks = F.interpolate(seq_masks, size=standard_shape, mode='trilinear', align_corners=False)
        seq_masks = seq_masks.squeeze(0).numpy().transpose(-1, 0, 1, 2)

        image_output_path = os.path.join(images_output_path, image_file)
        # image_output_paths.append(image_output_path)
        mask_output_path = os.path.join(masks_output_path, image_file)
        if not os.path.exists(images_output_path):
            os.makedirs(images_output_path)
            os.makedirs(masks_output_path)
        np.save(mask_output_path, seq_masks)
        np.save(image_output_path, seq)


    # seqs = np.array(seqs)
    # masks = np.array(masks)

    # # Z-Score normalization
    # datastet_channels = seqs.shape[2]
    # channel_means = np.mean(seqs, axis=(0, 1, 3, 4))  
    # channel_stds = np.std(seqs, axis= (0, 1, 3, 4))
    # seqs = (seqs - channel_means.reshape(1, 1, datastet_channels, 1, 1)) / channel_stds.reshape(1, 1, datastet_channels, 1, 1)
    # for i, save_path in enumerate(image_output_paths):
    #     np.save(save_path, seqs[i])

    # for i, seq in enumerate(seqs):
    #     if i > 30:
    #         break
    #     i2 = len(seq) // 2
    #     fig, axs = plt.subplots(1, datastet_channels+len(masks[i][i2]), figsize=(10, 5))
    #     for i4 in range(datastet_channels):
    #         axs[i4].imshow(seq[i2][i4], cmap='gray')
    #         axs[i4].set_title(f'Image {i4}')
    #     if len(masks[i][i2]) > 1:
    #         for i4 in range(datastet_channels, datastet_channels+len(masks[i][i2])):
    #             axs[i4].imshow(masks[i][i2][i4-datastet_channels], cmap='gray')
    #             axs[i4].set_title(f'Mask {i4-datastet_channels}')
    #     else:
    #         axs[datastet_channels].imshow(masks[i][i2][0], cmap='gray')
    #         axs[datastet_channels].set_title('Grayscale Mask')
    #     plt.savefig(f'outputs/{dataset}_{i}.png')
    #     plt.close()