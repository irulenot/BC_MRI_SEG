# Dataset: https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/
# Code: https://sites.duke.edu/mazurowski/2022/07/13/breast-mri-cancer-detect-tutorial-part1/

import numpy as np
import os
import pydicom
import pandas as pd
import json
import pickle
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import matplotlib.patches as patches
import pickle
import torch.nn.functional as F
import torch


def main():
    data_path = 'full_data/DUKE/'
    boxes_path = 'full_data/DUKE/Annotation_Boxes.xlsx'
    mapping_path = 'full_data/DUKE/Breast-Cancer-MRI-filepath_filename-mapping.xlsx'

    output_path = 'data/DUKE/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'boxes'), exist_ok=True)

    boxes_df = pd.read_excel(boxes_path)
    mapping_df = pd.read_excel(mapping_path)
    mapping_df = mapping_df[mapping_df['original_path_and_filename'].str.contains('pre')]
    mapping_df = mapping_df['classic_path'].reset_index(drop=True)

    patient_ids = []
    for idx, row in tqdm(boxes_df.iterrows()):
        patient_ids.append(int(row['Patient ID'].split('_')[-1]))
    boxes_df['Patient ID'] = patient_ids

    patients = {}
    for idx, row in tqdm(enumerate(mapping_df)):
        # Indices start at 1
        patient_id = int((row.split('/')[1]).split('_')[-1])
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(row)

    for key in tqdm(patients):
        images = []
        for path in patients[key]:
            path = os.path.join(data_path, path)
            try:
                dcm = pydicom.dcmread(path)
            except FileNotFoundError:
                dcm_fname_split = path.split('/')
                dcm_fname_end = dcm_fname_split[-1]
                assert dcm_fname_end.split('-')[1][0] == '0'
                dcm_fname_end_split = dcm_fname_end.split('-')
                dcm_fname_end = '-'.join([dcm_fname_end_split[0], dcm_fname_end_split[1][1:]])
                dcm_fname_split[-1] = dcm_fname_end
                dcm_fname = '/'.join(dcm_fname_split)
                dcm = pydicom.dcmread(dcm_fname)
            image = np.array(dcm.pixel_array)
            images.append(image)

        images = np.array(images)
        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])
        images_save_path = os.path.join(output_path, f'images/duke_{key}.npy')
        np.save(images_save_path, images)

        box_save_path = os.path.join(output_path, f'boxes/duke_{key}.json')
        box = boxes_df[boxes_df['Patient ID'] == key]
        box = box.to_dict(orient='records')[0]
        with open(box_save_path, 'w') as json_file:
            json.dump(box, json_file)

    data_dict = {'data_description': 'images: fat-saturated gradient echo T1-weighted pre-contrast sequences,'
                                     'boxes: annotation box for tumors',
                 'original_dataset': 'https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/'}
    save_path = os.path.join(output_path, f'duke_meta_data.pkl')
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)

    
    images_path = os.path.join(output_path, 'images')
    images_files = sorted([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])

    boxes_path = os.path.join(output_path, 'boxes')
    masks_output_path = os.path.join(output_path, 'masks')
    boxes_files = sorted([f for f in os.listdir(boxes_path) if os.path.isfile(os.path.join(boxes_path, f))])
    
    seqs, masks, image_output_paths = [], [], []
    masks_files = boxes_files
    for image_file, mask_file in tqdm(zip(images_files, masks_files)):
        seq_path = os.path.join(images_path, image_file)
        seq = np.load(seq_path)
        original_shape = seq.shape
        seqs.append(seq)
        box_path = os.path.join(boxes_path, mask_file)
        with open(box_path, 'r') as file:
            box = json.load(file)
        start_row, end_row = box['Start Row'], box['End Row']
        start_column, end_column = box['Start Column'], box['End Column']
        start_slice, end_slice = box['Start Slice'], box['End Slice']
        seq_masks = np.zeros(original_shape)
        seq_masks[start_slice:end_slice+1, :, start_row:end_row+1, start_column:end_column+1] = 1
        masks.append(seq_masks.astype(np.float32))

        mask_output_path = os.path.join(masks_output_path, image_file)
        if not os.path.exists(masks_output_path):
            os.makedirs(masks_output_path)
        np.save(mask_output_path, seq_masks)

if __name__ == "__main__":
    main()