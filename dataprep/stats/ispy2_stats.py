import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import cv2


def main():
    data = np.load('D:/data/ISPY2/ispy2_data.npy')
    with open('D:/data/ISPY2/ispy2_meta_data.pkl', 'rb') as pickle_file:
        meta_data = pickle.load(pickle_file)

    # Visual Inspection
    for key1 in meta_data['meta_data']:
        for key2 in meta_data['meta_data'][key1]:
            start_index, end_index = meta_data['meta_data'][key1][key2]['start_index'], \
                                     meta_data['meta_data'][key1][key2]['end_index']
            for i in range(start_index, end_index):
                annotation = data[i, 0, :, :]
                image = data[i, 1, :, :]
                plt.subplot(1, 2, 1)
                plt.imshow(image, cmap='gray')
                plt.title('image')
                plt.subplot(1, 2, 2)
                plt.imshow(annotation, cmap='gray')
                plt.title('annotation')
                plt.tight_layout()
                plt.show()

    images = data[:, 1, :, :]
    masks = data[:, 0, :, :]
    annotated_indices = []
    for key1 in meta_data['meta_data']:
        for key2 in meta_data['meta_data'][key1]:
            annotated_indices.extend(meta_data['meta_data'][key1][key2]['mask_indices'])
    annotated_masks = masks[annotated_indices]

    print('ISPY2 MRI DATASET STATISTICS')
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

    print('Annotated Mask Statistics')
    print('--------------------------')
    mean2, std2, range2, mode2, median2 = np.mean(annotated_masks), np.std(annotated_masks), \
                                          np.max(annotated_masks) - np.min(annotated_masks), \
                                          stats.mode(annotated_masks.flatten()), np.median(annotated_masks)
    print(f'Count: {len(annotated_masks)}')
    total_pixels2 = annotated_masks.shape[0] * annotated_masks.shape[1] * annotated_masks.shape[2]
    average_mask_size = (total_pixels2 - mode2[1]) / len(annotated_masks)
    print(f'Average Mask Size: {average_mask_size:.2f} pixels')
    print(f'Non-Zero Pixel Percentage: {(total_pixels2 - mode2[1]) / total_pixels2 * 100:.2f}%')
    print(f'Mean: {mean2:.2f}, Std: {std2:.2f}, Range: {range2}, Mode: {mode2}')
    print()

    print('Dataset Statistics')
    print('------------------')
    print(f'Image dimensions: {images.shape[1]} x {images.shape[2]}')
    print(f'Percentage of Images with Masks: {len(annotated_masks) / len(images) * 100:.2f}%')
    print(
        f'Image to Mask Ratio: {average_image_size / average_mask_size:.2f} (~{round(average_image_size / average_mask_size)}x more Image Pixels than Mask Pixels)')
    print(
        f'Total Pixels to Mask Ratio: {total_pixels1 / average_mask_size:.2f} (~{round(total_pixels1 / average_mask_size)}x more Pixels than Annotated Pixels)')

if __name__ == "__main__":
    main()