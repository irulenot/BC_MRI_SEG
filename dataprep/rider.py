# Dataset: https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI

from os import walk
import os
import numpy as np
import pydicom
from tqdm import tqdm
import pickle

def main():
    # Aggregating file tree and images
    data_path = 'full_data\RIDER\RIDER Breast MRI'
    file_dict = {}
    master_dir = None
    for dirpath, dirnames, filenames in tqdm(walk(data_path)):
        if len(file_dict) == 0:
            master_dir = dirpath.split('\\')[-1]
            file_dict[master_dir] = {}
            for dirname in dirnames:
                file_dict[master_dir][dirname] = {}
            continue

        if len(filenames) == 0:
            continue

        sheriff_dir = dirpath.split('\\')[-3]
        deputy_dir = dirpath.split('\\')[-2]
        current_dir = dirpath.split('\\')[-1]

        patient, image_type = None, None
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            dcm = pydicom.dcmread(path)
            if patient == None:
                patient = dcm['PatientID'].value
                SeriesDescription = dcm['SeriesDescription'].value
                if deputy_dir not in file_dict[master_dir][sheriff_dir]:
                    file_dict[master_dir][sheriff_dir][deputy_dir] = {}
                file_dict[master_dir][sheriff_dir][deputy_dir][current_dir] = {'patient': patient,
                                                                               'SeriesDescription': SeriesDescription,
                                                                               'images': []}
            current_dict = file_dict[master_dir][sheriff_dir][deputy_dir][current_dir]
            current_dict['images'].append(dcm.pixel_array)

    # Putting images into a dictionary
    for key0 in tqdm(file_dict):
        patients = {}
        for id, key1 in enumerate(file_dict[key0]):
            patient_id = 'patient_' + str(id)
            patients[patient_id] = {'images': [0, 0, 0, 0],
                                    'masks': {}}
            for key2 in file_dict[key0][key1]:
                for key3 in file_dict[key0][key1][key2]:
                    data_type = file_dict[key0][key1][key2][key3]['SeriesDescription']
                    images = file_dict[key0][key1][key2][key3]['images']
                    images = np.array(images)
                    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])
                    if data_type == 'reference VOI mask':
                        masks = images
                        patients[patient_id]['masks'] = masks
                    else:
                        if data_type == 'early anatomical reference':
                            patients[patient_id]['images'][0] = images
                        elif data_type == 'ADC = short int value x 0.2E-06 (units: 10E-03 mm^2/s)':
                            patients[patient_id]['images'][1] = images
                        elif data_type == 'B0':
                            patients[patient_id]['images'][2] = images
                        elif data_type == 'B800':
                            patients[patient_id]['images'][3] = images

    output_path = 'data/RIDER/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    for patient_id in patients:
        patient_data = patients[patient_id]
        masks_save_path = os.path.join(output_path, f'masks/rider_{patient_id}.npy')
        images_save_path = os.path.join(output_path, f'images/rider_{patient_id}.npy')
        masks = patient_data['masks']
        images = np.array(patient_data['images'])
        images = np.squeeze(images)
        images = np.transpose(images, (1, 0, 2, 3))

        # Image 2 features 1,2,3 from RIDER were misaligned
        if patient_id == 'patient_1':
            channels_to_flip = [1, 2, 3]
            for channel in channels_to_flip:
                images[:, channel, :, :] = images[:, channel, ::-1, :]

        np.save(masks_save_path, masks)
        np.save(images_save_path, images)
    data_dict = {'images_description': "Four modalities include: "
                                       "0: 'early anatomical reference',"
                                       "1: 'ADC = short int value x 0.2E-06 (units: 10E-03 mm^2/s)',"
                                       "2: 'B0', "
                                       "3: 'B800'",
                 'original_dataset_link': 'https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI'}
    meta_save_path = os.path.join(output_path, 'rider_meta_data.pkl')
    with open(meta_save_path, 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)
        print()

if __name__ == "__main__":
    main()