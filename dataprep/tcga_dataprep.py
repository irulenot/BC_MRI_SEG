import os
import pydicom
import numpy as np
from tqdm import tqdm
from dataprep.util import find_image_dim
import pickle
import ast
import json
from os import walk
import copy
import pandas as pd


def main():
    data_path = 'E:/breast_cancer_mri/TCGA/TCGA-BRCA/'
    annotations_path = 'E:/breast_cancer_mri/TCGA/tcga breast radiologist reads.xls'
    annotations = pd.read_excel(annotations_path)

    series_paths_path = 'E:/breast_cancer_mri/TCGA/series_paths.json'
    if not os.path.exists(series_paths_path):
        print('series_paths')
        series_paths = {}
        for dirpath, dirnames, filenames in tqdm(walk(data_path)):
            if dirnames == [] and filenames != []:
                for i, file_name in enumerate(filenames):
                    image_path = os.path.join(dirpath, file_name)
                    dcm = pydicom.dcmread(image_path)
                    PatientID = dcm['PatientID'].value
                    SeriesDescription = dcm['SeriesDescription'].value
                    # SOPInstanceUID = dcm['SOPInstanceUID'].value
                    if PatientID not in series_paths:
                        series_paths[PatientID] = {'series_ids': [],
                                                   'SeriesDescriptions': [],
                                                   'series_path': []}
                    series_paths[PatientID]['series_path'].append(dirpath)
                    series_id = image_path.split('\\')[-2]
                    series_paths[PatientID]['series_ids'].append(series_id)
                    series_paths[PatientID]['SeriesDescriptions'].append(SeriesDescription)
                    break
        with open(series_paths_path, 'w') as json_file:
            json.dump(series_paths, json_file)
    else:
        with open(series_paths_path, 'r') as json_file:
            series_paths = json.load(json_file)

    print('patient_series')
    patient_series = {}
    for i in tqdm(range(len(annotations))):
        annotation = annotations.iloc[i]
        PATIENT_ID = annotation['PATIENT_ID']
        if PATIENT_ID in series_paths:
            SERIES_UID = annotation['SERIES_UID']
            for i2, series_id in enumerate(series_paths[PATIENT_ID]['series_ids']):
                if SERIES_UID == series_id:
                    SeriesDescription = series_paths[PATIENT_ID]['SeriesDescriptions'][i2]
                    series_path = series_paths[PATIENT_ID]['series_path'][i2]
                    if PATIENT_ID not in patient_series:
                        patient_series[PATIENT_ID] = {}
                    patient_series[PATIENT_ID][SeriesDescription] = series_path
        else:
            print(f'{PATIENT_ID} not in series_paths')

    patient_dict = {}
    output_path = 'E:/breast_cancer_mri/TCGA/'
    curr_id = 0
    for PATIENT_ID in tqdm(patient_series):
        for i2, SeriesDescription in enumerate(patient_series[PATIENT_ID]):
            series_path = patient_series[PATIENT_ID][SeriesDescription]
            image_names = [f for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
            images = []
            for image_name in image_names:
                image_path = os.path.join(series_path, image_name)
                dcm = pydicom.dcmread(image_path)
                img = dcm.pixel_array
                images.append(img)
            images_save_path = os.path.join(output_path, f'images0/tcga_{curr_id}_{i2}.npy')
            seq = np.array(images)
            seq = seq.reshape(seq.shape[0], 1, seq.shape[1], seq.shape[2])
            np.save(images_save_path, seq)
            if PATIENT_ID not in patient_dict:
                patient_dict[PATIENT_ID] = {'idx': curr_id,
                                            'series_descriptions': [],
                                            'series_paths': []}
            patient_dict[PATIENT_ID]['series_descriptions'].append(SeriesDescription)
            patient_dict[PATIENT_ID]['series_paths'].append(series_path)
        curr_id += 1

    # data_dict = {'data_description': 'meta_data keys are: PATIENT_ID, SeriesDescription, image_index'
    #                                  'data_index denotes the index of the image in the data.'
    #                                  'data shape (238, 512, 512) consists of images with 2 dimensions (height x width) of size 512 x 512'
    #                                  'images have been padded to be size 512 x 512, refer to \'original_shape\' if you want to revert the size',
    #              'original_dataset': 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=19039112'}
    save_path = os.path.join(output_path, f'patient_dict.pkl')
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(patient_dict, pickle_file)

if __name__ == "__main__":
    main()