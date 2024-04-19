import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import cv2


def main():
    data = np.load('D:/data/DUKE/duke_data0.npy')
    with open('D:/data/DUKE/duke_meta_data0.pkl', 'rb') as pickle_file:
        meta_data = pickle.load(pickle_file)

    # Visual Inspection
    # keys = list(meta_data['meta_data'].keys())
    # index = 0
    # key = keys[index]
    # mask_indices = meta_data['meta_data'][key]['mask_indices']
    # sr, er, sc, ec = meta_data['meta_data'][key]['box'].values()
    # index += 1
    # key = keys[index]
    # next_start_index = meta_data['meta_data'][key]['start_index']
    # for i, image in enumerate(data):
    #     if i == next_start_index:
    #         mask_indices = meta_data['meta_data'][key]['mask_indices']
    #         sr, er, sc, ec = meta_data['meta_data'][key]['box'].values()
    #         index += 1
    #         key = keys[index]
    #         next_start_index = meta_data['meta_data'][key]['start_index']
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(image, cmap='gray')
    #     plt.title('Image')
    #     if i in mask_indices:
    #         cv2.rectangle(image, (sc, sr), (ec, er), np.max(image), 2)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(image, cmap='gray')
    #     plt.title('Image + Box')
    #     plt.tight_layout()
    #     plt.show()

    images = data

    print('DUKE MRI DATASET STATISTICS')
    print(f'Dataset Shape: {data.shape}')
    print(f"Patient Count: {len(meta_data['meta_data'])}")
    print(f"Average Images per Patient: {len(images)/len(meta_data['meta_data']):.2f}")
    print('============================\n')

    print('Image Statistics')
    print('----------------')
    mean1, std1, range1, mode1, median1 = np.mean(images), np.std(images), np.max(images) - np.min(images), \
                                     stats.mode(images.flatten()), np.median(images)
    print(f'Count: {len(images)}')
    total_pixels1 = images.shape[0] * images.shape[1] * images.shape[2]
    average_image_size = (total_pixels1-mode1[1])/len(images)
    print(f'Average Image Size: {average_image_size:.2f} pixels')
    print(f'Non-Zero Pixel Percentage: {(total_pixels1-mode1[1])/total_pixels1*100:.2f}%')
    print(f'Mean: {mean1:.2f}, Std: {std1:.2f}, Range: {range1:.2f}, Mode: {mode1}')
    print()

    print('Dataset Statistics')
    print('------------------')
    print(f'Image dimensions: {images.shape[1]} x {images.shape[2]} (*Images were padded)')
    box_count = 0
    for key in meta_data['meta_data']:
        box_count += len(meta_data['meta_data'][key]['mask_indices'])
    print(f'Percentage of Images with Masks: {box_count/len(images)*100:.2f}%')

def verify():
    key_list = []
    for i in range(9):
        with open(f'D:/data/DUKE/duke_meta_data{i}.pkl', 'rb') as pickle_file:
            meta_data = pickle.load(pickle_file)
            key_list += list(meta_data['meta_data'].keys())
    print(len(key_list))
    # print(sorted(key_list))

if __name__ == "__main__":
    # main()
    verify()