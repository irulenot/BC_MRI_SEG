import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

datasets = ['RIDER', 'ISPY1', 'DUKE']
data_path = 'data/'

for i, dataset in enumerate(datasets):
    print(dataset)
    path = os.path.join(data_path, dataset)
    
    images_path = os.path.join(path, 'images')
    images_output_path = os.path.join(path, 'images_std')
    images_files = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])

    seq_path = os.path.join(images_path, images_files[0])
    seq = np.load(seq_path)
    channels = seq.shape[1]

    channel_metrics = {}
    for c in tqdm(range(channels)):
        seqs = []
        for image_file in tqdm(images_files):
            seq_path = os.path.join(images_path, image_file)
            if dataset == 'DUKE':
                seq = np.load(seq_path)[:, c].astype(np.float16)
            else:
                seq = np.load(seq_path)[:, c].astype(np.float32)
            seqs.append(seq.flatten())
        seqs = np.concatenate(seqs)
        top_percent_indices = int(0.001 * len(seqs))
        sorted_values = np.sort(seqs)
        top_0_1_percent_max = sorted_values[-top_percent_indices-1:][0]
        top_0_1_percent_min = sorted_values[:top_percent_indices][-1]
        print("top_0_1_percent_max", top_0_1_percent_max)
        print("top_0_1_percent_min", top_0_1_percent_min)
        seqs = np.clip(seq, top_0_1_percent_min, top_0_1_percent_max)
        if dataset == 'DUKE':
            stds, means = [], []
            for image_file in images_files:
                seq_path = os.path.join(images_path, image_file)
                seq = np.load(seq_path)[:, c].astype(np.float32)
                means.append(np.mean(seq))
                stds.append(np.std(seq))
            mean = np.mean(means)
            std = np.std(stds)
        else:
            mean = np.mean(seqs)
            std = np.std(seqs)
        print('channel', c)
        print('mean', mean)
        print('std', std)
        channel_metrics[c] = {'clip_max': top_0_1_percent_max,
                              'clip_min': top_0_1_percent_min,
                              'mean': mean,
                              'std': std}
    del seqs
    
    for i2, image_file in tqdm(enumerate(images_files)):
        seq_path = os.path.join(images_path, image_file)
        seq = np.load(seq_path).astype(np.float32)
        for c in range(channels):
            clip_max, clip_min = channel_metrics[c]['clip_max'], channel_metrics[c]['clip_min']
            mean, std = channel_metrics[c]['mean'], channel_metrics[c]['std']
            seq[:, c] = np.clip(seq[:, c], clip_min, clip_max)
            seq[:, c] = (seq[:, c] - mean) / std

        if not os.path.exists(images_output_path):
            os.makedirs(images_output_path)
        image_output_path = os.path.join(images_output_path, image_file)
        np.save(image_output_path, seq)