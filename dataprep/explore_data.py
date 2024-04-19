import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.patches as patches
import pickle

datasets = ['BreastDM', 'ISPY1', 'ISPY2', 'RIDER', 'DUKE']
data_path = 'data/'

for dataset in tqdm(datasets):
    path = os.path.join(data_path, dataset)
    images_path = os.path.join(path, 'images')
    masks_path = os.path.join(path, 'masks')
    images_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    masks_files = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]
    sequence_shapes, mask_count = [], 0
    for image_file, mask_file in zip(images_files, masks_files):
        seq_path = os.path.join(images_path, image_file)
        mask_path = os.path.join(masks_path, mask_file)
        seq = np.load(seq_path)
        mask = np.load(mask_path)
        mask_sums = np.sum(mask, axis=(2, 3))
        mask_count += np.sum(mask_sums > 0)
        sequence_shapes.append(seq.shape)
    total_images, sequence_count = 0, 0
    lengths, channels, sizes = [], [], []
    for sequence_shape in sequence_shapes:
        lengths.append(sequence_shape[0])
        channels.append(sequence_shape[1])
        sequence_count += sequence_shape[1]
        total_images += sequence_shape[0] * sequence_shape[1]
        sizes.append(sequence_shape[2])
    print(dataset)
    print(f'total_images: {total_images}, mask_count: {mask_count}')
    print(f'images/masks: {mask_count/total_images:.3f}')
    print(f'total patients: {len(sequence_shapes)}')
    print(f'total sequences: {sequence_count}')
    mean_value, std_value, max, min = np.mean(lengths), np.std(lengths), np.max(lengths), np.min(lengths)
    print(f'lengths: mean: {mean_value:.2f}, std: {std_value:.2f}, max: {max:.2f}, min: {min:.2f}')
    mean_value, std_value, max, min = np.mean(channels), np.std(channels), np.max(channels), np.min(channels)
    print(f'channels: mean: {mean_value:.2f}, std: {std_value:.2f}, max: {max:.2f}, min: {min:.2f}')
    mean_value, std_value, max, min = np.mean(sizes), np.std(sizes), np.max(sizes), np.min(sizes)
    print(f'sizes: mean: {mean_value:.2f}, std: {std_value:.2f}, max: {max:.2f}, min: {min:.2f}')