# Dataset: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541#101942541215b684587f64c8cab1ffc45cd63f339

import os
import gzip
from tqdm import tqdm
import numpy as np
import nibabel as nib
import pickle

def main():
    data_dir = 'full_data/ISPY1/ISPY1-Tumor-SEG-Radiomics/NIfTI-Files/'
    output_path = 'data/ISPY1/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    patient_id = 0
    for i in tqdm(range(0, 240)):
        file_path1 = f'{data_dir}masks_stv_manual/ISPY1_1{i:03d}.nii.gz'
        if not os.path.exists(file_path1):
            continue

        image_paths = []
        for i2 in range(3):
            gzip_file = f'{data_dir}images_bias-corrected_resampled_zscored_nifti/ISPY1_1{i:03d}/ISPY1_1{i:03d}_DCE_000{i2}_N3_zscored.nii.gz'
            save_path = f'{data_dir}images_bias-corrected_resampled_zscored_nifti/ISPY1_1{i:03d}/ISPY1_1{i:03d}_DCE_000{i2}_N3_zscored.nii'
            image_paths.append(save_path)
            if os.path.isfile(gzip_file):
                with gzip.open(gzip_file, 'rb') as gz_file, open(save_path, 'wb') as output_file:
                    output_file.write(gz_file.read())

        # Load the NIfTI image
        masks = nib.load(file_path1).get_fdata()
        images = []
        for i2, image_path in enumerate(image_paths):
            images.append(nib.load(image_path).get_fdata())
        images = np.array(images)

        masks = np.transpose(masks, (-1, 0, 1)).reshape(masks.shape[-1], 1, masks.shape[0], masks.shape[1])
        images = np.transpose(images, (-1, 0, 1, 2))
        if masks.shape[2] != images.shape[2]:
            continue

        masks_save_path = os.path.join(output_path, f'masks/ispy1_{patient_id}.npy')
        images_save_path = os.path.join(output_path, f'images/ispy1_{patient_id}.npy')
        np.save(masks_save_path, masks)
        np.save(images_save_path, images)
        data_dict = {'images_description': "We used the T1 weighted DCE-MR images, pre contrast scan and first two post contrast scans. "
                                           "masks: manual segmentations of the structural tumor volume,"
                                           "images: DICOM images converted to NIfTI and then bias-corrected + resampled to [1,1,1] and then z-scored across the 3 timepoints,",
                     'original_dataset_link': 'https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541#101942541215b684587f64c8cab1ffc45cd63f339'}
        meta_save_path = os.path.join(output_path, 'ispy1_meta_data.pkl')
        with open(meta_save_path, 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)
            print()
        patient_id += 1

if __name__ == "__main__":
    main()