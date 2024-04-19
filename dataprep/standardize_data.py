import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


datasets = ['RIDER', 'BreastDM' 'ISPY1', 'ISPY2', 'DUKE']
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
        if dataset == 'ISPY2' and seq.shape[1] == 2:  # ISPY2 has only some multiple angle labels
            seq = seq[:, 0]
            seq = seq.reshape(seq.shape[0], 1, seq.shape[1], seq.shape[2])
        for c in range(channels):
            clip_max, clip_min = channel_metrics[c]['clip_max'], channel_metrics[c]['clip_min']
            mean, std = channel_metrics[c]['mean'], channel_metrics[c]['std']
            seq[:, c] = np.clip(seq[:, c], clip_min, clip_max)
            seq[:, c] = (seq[:, c] - mean) / std

        # Image 2 features 1,2,3 from RIDER were misaligned
        if dataset == 'RIDER' and i2 == 1:
            channels_to_flip = [1, 2, 3]
            for channel in channels_to_flip:
                seq[:, channel, :, :] = seq[:, channel, ::-1, :]

        if not os.path.exists(images_output_path):
            os.makedirs(images_output_path)
        image_output_path = os.path.join(images_output_path, image_file)
        np.save(image_output_path, seq)

    # counter = 0
    # masks_path = os.path.join(path, 'masks')
    # for i2, image_file in tqdm(enumerate(images_files)):
    #     seq_path = os.path.join(images_output_path, image_file)
    #     mask_path = os.path.join(masks_path, image_file)
    #     seq = np.load(seq_path)
    #     masks = np.load(mask_path)
    #     image_count = 3
    #     mid_idx = len(seq) // 2 - image_count // 2
    #     fig, axs = plt.subplots(channels, image_count*2, figsize=(10, 5))
    #     for i3 in range(image_count):
    #         idx = mid_idx + i3
    #         img = seq[idx]
    #         mask = masks[idx]
    #         for c in range(channels):
    #             img_c = img[c]
    #             if mask.shape[0] == 1:
    #                 mask_c = mask[0]
    #             else:
    #                 mask_c = mask[c]
    #             if channels > 1:
    #                 axs[c][i3*2].imshow(img_c, cmap='gray')
    #                 axs[c][i3*2+1].imshow(mask_c, cmap='gray')
    #             else:
    #                 axs[i3*2].imshow(img_c, cmap='gray')
    #                 axs[i3*2+1].imshow(mask_c, cmap='gray')
    #     plt.savefig(f'outputs/{dataset}_{i2}.png')
    #     plt.close()
    #     counter += 1
    #     if counter > 15:
    #         break