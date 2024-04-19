import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import cv2
import math


def create_centered_rectangle(point1, point2, width, image):
    x1, y1 = point1
    x2, y2 = point2
    # Calculate the angle between the two points
    angle = math.atan2(y2 - y1, x2 - x1)
    # Calculate half of the width
    half_width = width / 2
    # Calculate the coordinates of the four vertices of the rectangle
    x1_new = x1 + half_width * math.cos(angle + math.pi / 2)
    y1_new = y1 + half_width * math.sin(angle + math.pi / 2)
    x2_new = x2 + half_width * math.cos(angle + math.pi / 2)
    y2_new = y2 + half_width * math.sin(angle + math.pi / 2)
    x3_new = x2 + half_width * math.cos(angle - math.pi / 2)
    y3_new = y2 + half_width * math.sin(angle - math.pi / 2)
    x4_new = x1 + half_width * math.cos(angle - math.pi / 2)
    y4_new = y1 + half_width * math.sin(angle - math.pi / 2)
    # Draw the rectangle on the image
    cv2.line(image, (int(x1_new), int(y1_new)), (int(x2_new), int(y2_new)), np.max(image), 2)
    cv2.line(image, (int(x2_new), int(y2_new)), (int(x3_new), int(y3_new)), np.max(image), 2)
    cv2.line(image, (int(x3_new), int(y3_new)), (int(x4_new), int(y4_new)), np.max(image), 2)
    cv2.line(image, (int(x4_new), int(y4_new)), (int(x1_new), int(y1_new)), np.max(image), 2)
    return image

def main():
    data = np.load('D:/data/TCGA/tcga_data.npy')
    with open('D:/data/TCGA/tcga_meta_data.pkl', 'rb') as pickle_file:
        meta_data = pickle.load(pickle_file)

    # Visual Inspection
    # meta_data = meta_data['meta_data']
    # for key in meta_data:
    #     for key2 in meta_data[key]:
    #         for key3 in meta_data[key][key2]:
    #             annotated_index = meta_data[key][key2][key3]['data_index']
    #             annotated_points = meta_data[key][key2][key3]['POINTS']
    #             annotated_measurements = meta_data[key][key2][key3]['original_MEASUREMENTS']
    #             annotated_image = data[annotated_index]
    #             pt1x, pt1y, pt2x, pt2y = annotated_points[0][0], annotated_points[0][1], annotated_points[1][0], annotated_points[1][1]
    #             cv2.line(annotated_image, [pt1x, pt1y], [pt2x, pt2y], np.max(annotated_image), 1)
    #             plt.subplot(1, 2, 1)
    #             plt.imshow(annotated_image, cmap='gray')
    #             plt.title('annotated_image')
    #             annotated_image2 = create_centered_rectangle([pt1x, pt1y], [pt2x, pt2y], annotated_measurements, annotated_image)
    #             plt.subplot(1, 2, 2)
    #             plt.imshow(annotated_image2, cmap='gray')
    #             plt.title('annotated_image2 (my interpretation)')
    #             plt.show()

    images = data

    print('TCGA MRI DATASET STATISTICS')
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
    print('Percentage of Images with Masks: 100%')
    # box_count = 0
    # for key in meta_data['meta_data']:
    #     box_count += len(meta_data['meta_data'][key]['mask_indices'])
    # print(f'Percentage of Images with Masks: {box_count/len(images)*100:.2f}%')

if __name__ == "__main__":
    main()