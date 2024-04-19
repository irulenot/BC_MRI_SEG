import os
from os import walk
import pydicom
import numpy as np
from tqdm import tqdm
import json
import pickle


def main():
    data_path = 'E:/breast_cancer_mri/ISPY2/ACRIN-6698/'
    file_ids_path = 'dataprep/file_dict.json'

    if not os.path.exists(file_ids_path):
        print('Collecting annotations.')
        file_dict, reference_images = {}, {}
        for dirpath, dirnames, filenames in tqdm(walk(data_path)):
            if dirnames == [] and filenames != []:
                annotation_path = os.path.join(dirpath, filenames[0])
                if os.path.getsize(annotation_path) == 0:
                    continue
                dcm = pydicom.dcmread(annotation_path)
                ClinicalTrialTimePointID = dcm['ClinicalTrialTimePointID'].value
                if ClinicalTrialTimePointID != 'T0':
                    continue
                SeriesDescription = dcm['SeriesDescription'].value
                SEG_index = SeriesDescription.find('SEG')
                if SEG_index != -1:
                    ClinicalTrialSubjectID = int(dcm['ClinicalTrialSubjectID'].value)
                    if ClinicalTrialSubjectID not in file_dict:
                        file_dict[ClinicalTrialSubjectID] = {}
                    SEG_key = len(file_dict[ClinicalTrialSubjectID])

                    for i, file_name in enumerate(filenames):
                        annotation_path = os.path.join(dirpath, file_name)
                        dcm = pydicom.dcmread(annotation_path)

                        annotated_image_indices, UIDs = [], []
                        i = 0
                        while i != -1:
                            try:
                                UIDs.append(dcm['SourceImageSequence'][i]['ReferencedSOPInstanceUID'].value)
                                annotated_image_indices.append(-1)
                                i += 1
                            except Exception as e:
                                i = -1
                        file_dict[ClinicalTrialSubjectID][SEG_key] \
                            = {'annotation_path': annotation_path,
                               'annotated_image_indices': annotated_image_indices,
                               'UIDs': UIDs}

        print('\nLinking images.')
        for dirpath, dirnames, filenames in tqdm(walk(data_path)):
            if dirnames == [] and filenames != []:
                image_path = os.path.join(dirpath, filenames[0])
                if os.path.getsize(image_path) == 0:
                    continue
                dcm = pydicom.dcmread(image_path)
                ClinicalTrialTimePointID = dcm['ClinicalTrialTimePointID'].value
                if ClinicalTrialTimePointID != 'T0':
                    continue
                SeriesDescription = dcm['SeriesDescription'].value
                TRACE_index = SeriesDescription.find('TRACE')
                if TRACE_index != -1:
                    ClinicalTrialSubjectID = int(dcm['ClinicalTrialSubjectID'].value)
                    if ClinicalTrialSubjectID not in file_dict:  # TEMP
                        continue
                    updated_filenames = [dirpath + "/" + filename for filename in filenames]
                    for seg_key in file_dict[ClinicalTrialSubjectID]:
                        UIDs = file_dict[ClinicalTrialSubjectID][seg_key]['UIDs'].copy()
                        linked = False
                        for image_index, filename in enumerate(updated_filenames):
                            dcm = pydicom.dcmread(filename)
                            SOPInstanceUID = dcm['SOPInstanceUID'].value
                            if SOPInstanceUID in UIDs:
                                UID_index = UIDs.index(SOPInstanceUID)
                                file_dict[ClinicalTrialSubjectID][seg_key]['annotated_image_indices'][UID_index] = image_index
                                if linked == False:
                                    file_dict[ClinicalTrialSubjectID][seg_key]['image_paths'] = updated_filenames
                                    linked = True
        with open(file_ids_path, 'w', encoding='utf-8') as json_file:
            json.dump(file_dict, json_file)
    else:
        with open(file_ids_path, 'r', encoding='utf-8') as json_file:
            file_dict = json.load(json_file)

    print('\nCreating Dataset.')
    output_path = 'E:/breast_cancer_mri/ISPY2/'
    curr_id = 0
    for patient_id in tqdm(file_dict):
        patient_dict = file_dict[patient_id]
        images, masks = [], []
        for SEG_key in patient_dict:
            seg_dict = patient_dict[SEG_key]
            annotated_image_indices = seg_dict['annotated_image_indices']
            if -1 in annotated_image_indices:
                print(patient_id, SEG_key, 'images are missing links.')
                break
            image_paths = seg_dict['image_paths']
            annotation_path = seg_dict['annotation_path']
            annotations = pydicom.dcmread(annotation_path).pixel_array
            if len(annotations.shape) == 2:
                annotations = annotations.reshape(1,annotations.shape[0], annotations.shape[1])
            if annotations.shape[1] != annotations.shape[2]:
                print(patient_id, SEG_key, 'images have variable sizes.')
                break
            images.append([])
            masks.append([])
            for i, image_path in enumerate(image_paths):
                image = pydicom.dcmread(image_path).pixel_array
                if i in annotated_image_indices:
                    mask = annotations[annotated_image_indices.index(i)]
                else:
                    mask = np.zeros(image.shape)
                images[int(SEG_key)].append(image)
                masks[int(SEG_key)].append(mask)
        if len(masks) > 1:
            if len(masks[0]) != len(masks[1]):
                print(patient_id, SEG_key, 'channels have variable sizes.')
                continue
        elif len(masks) == 0:
            print(patient_id, SEG_key, 'is empty.')
            continue
        masks_save_path = os.path.join(output_path, f'masks/ispy2_{curr_id}.npy')
        images_save_path = os.path.join(output_path, f'images/ispy2_{curr_id}.npy')
        masks = np.array(masks)
        images = np.array(images)
        images = images.transpose(1, 0, 2, 3)
        masks = masks.transpose(1, 0, 2, 3)
        np.save(masks_save_path, masks)
        np.save(images_save_path, images)
        curr_id += 1

    data_dict = {'images: ACRIN-6698: DWI TRACE: from S5: bVals=0,100,600,800.'
                 'original_dataset': 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447'}
    save_path = os.path.join(output_path, f'ispy2_meta_data.pkl')
    with open(save_path, 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)

if __name__ == "__main__":
    main()