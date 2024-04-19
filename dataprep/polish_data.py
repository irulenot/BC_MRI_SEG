import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.patches as patches
import pickle

'''
1. Add dim 1 for RIDER and TCGA
2. Save TCGA metadata file
'''

datasets = ['ISPY1', 'ISPY2', 'RIDER', 'TCGA', 'DUKE']
datasets = ['TCGA']
data_path = 'data/'

for dataset in tqdm(datasets):
    path = os.path.join(data_path, dataset)
    images_path = os.path.join(path, 'images')
    masks_path = os.path.join(path, 'masks')
    images_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    masks_files = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]
    counter = 5
    for image_file, mask_file in zip(images_files, masks_files):
        seq_path = os.path.join(images_path, image_file)
        seq_masks_path = os.path.join(masks_path, mask_file)
        seq = np.load(seq_path)
        seq_masks = np.load(seq_masks_path)

        for i in range(seq.shape[1]):
            for i2 in range(seq.shape[0]):
                print(f'{seq_path}_{i2}')

                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(seq[i2, 0], cmap='gray')
                axs[0].set_title('Grayscale Image')
                axs[1].imshow(seq_masks[i2, 0], cmap='gray')
                axs[1].set_title('Grayscale Mask')
                plt.savefig(f"outputs/{image_file[:-4]}_{i2}.png")
                plt.close()
    counter -= 1
    if counter < 0:
        break

# 1. Dim for RIDER and TCGA
# for dataset in tqdm(datasets):
#     path = os.path.join(data_path, dataset)
#     images_path = os.path.join(path, 'images')
#     masks_path = os.path.join(path, 'masks')
#     images_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
#     masks_files = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]
#     for image_file, mask_file in zip(images_files, masks_files):
#         seq_path = os.path.join(images_path, image_file)
#         seq_masks_path = os.path.join(masks_path, mask_file)
#         seq = np.load(seq_path)
#         seq_masks = np.load(seq_masks_path)

#         for i in range(seq.shape[1]):
#             for i2 in range(seq.shape[0]):
#                 print(f'{seq_path}_c:{i}')
#                 seq_channel = seq[:, i][i2]
#                 seq_mask_channel = seq_masks[:, i][i2]

#                 fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#                 axs[0].imshow(seq_channel, cmap='gray')
#                 axs[0].set_title('Grayscale Image')
#                 axs[1].imshow(seq_mask_channel, cmap='gray')
#                 axs[1].set_title('Grayscale Mask')
#                 plt.savefig(f"outputs/{image_file[:-4]}_c:{i}_{i2}.png")
#                 plt.close()
#         break

# 2. Visual Inspection of DUKE
# for dataset in tqdm(datasets):
#     path = os.path.join(data_path, dataset)
#     images_path = os.path.join(path, 'images')
#     masks_path = os.path.join(path, 'boxes')
#     images_files = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])
#     masks_files = sorted([f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))])
#     counter = 0
#     for image_file, mask_file in zip(images_files, masks_files):
#         seq_path = os.path.join(images_path, image_file)
#         seq_masks_path = os.path.join(masks_path, mask_file)
#         seq = np.load(seq_path)
#         with open(seq_masks_path, 'r') as file:
#             box = json.load(file)
#         print(seq.shape[1])
#         seq_channel = seq[:, 0]
#         if dataset == 'DUKE':
#             median_index = (box['Start Slice'] + box['End Slice']) // 2
#         image = seq_channel[median_index]

#         fig, ax = plt.subplots()
#         ax.imshow(image, cmap='gray')
#         rect = patches.Rectangle(
#             (box['Start Column'], box['Start Row']),
#             box['End Column'] - box['Start Column'],
#             box['End Row'] - box['Start Row'],
#             linewidth=1,
#             edgecolor='r',
#             facecolor='none'
#         )
#         ax.add_patch(rect)
#         plt.savefig(f'outputs/DUKE_{counter}b.png')
#         print(f'outputs/DUKE_{counter}b.png')
#         plt.close()
#         if counter == 30:
#             break
#         counter += 1

# # 1. Dim for RIDER and TCGA
# for dataset in tqdm(['RIDER', 'TCGA']):
#     path = os.path.join(data_path, dataset)
#     images_path = os.path.join(path, 'images')
#     masks_path = os.path.join(path, 'masks')
#     images_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
#     masks_files = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]
#     for image_file, mask_file in zip(images_files, masks_files):
#         seq_path = os.path.join(images_path, image_file)
#         seq_masks_path = os.path.join(masks_path, mask_file)
#         # seq = np.load(seq_path)
#         seq_masks = np.load(seq_masks_path)
#         seq_masks = seq_masks.reshape(seq_masks.shape[0], 1, seq_masks.shape[1], seq_masks.shape[2])
#         np.save(seq_masks_path, seq_masks)

        # Visual inspection of ['ISPY1', 'ISPY2', 'RIDER', 'TCGA'] before dim fix
        # for i in range(seq.shape[1]):
        #     print(f'{seq_path}_c:{i}')
        #     seq_channel = seq[:, i]
        #     if dataset == 'ISPY1' or dataset == 'RIDER':
        #         i = 0
        #     if dataset == 'RIDER' or dataset == 'TCGA':
        #         seq_masks_channel = seq_masks
        #     else:
        #         seq_masks_channel = seq_masks[:, i]
        #     sums = np.sum(seq_masks_channel, axis=(1, 2))
        #     max_sum_index = np.argmax(sums)
        #     image = seq_channel[max_sum_index]
        #     mask = seq_masks_channel[max_sum_index]

        #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #     axs[0].imshow(image, cmap='gray')
        #     axs[0].set_title('Grayscale Image')
        #     axs[1].imshow(mask, cmap='gray')
        #     axs[1].set_title('Grayscale Mask')
        #     plt.show()
        #     plt.close()

# TCGA meta file
# save_path = os.path.join('E:/breast_cancer_mri/TCGA', f'tcga_meta_data.pkl')
# data_dict = {'data_description': 'Some scans have multiple channels. They are either different views or repeat scans.',
#              'original_dataset': 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19039112'}
# with open(save_path, 'wb') as pickle_file:
#     pickle.dump(data_dict, pickle_file)