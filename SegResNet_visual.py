import os
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, MeanIoU, ConfusionMatrixMetric
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from tqdm import tqdm
from monai.networks.nets import SegResNet
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np

standard_shape = (256, 256, 128)
class DatasetMRI(Dataset):
    def __init__(self, data_paths, label_paths, transforms=None, train=True):
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.train = train

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        dataset = self.data_paths[idx].split('/')[-1].split('_')[0]
        image = np.load(self.data_paths[idx]).transpose(1, 2, 3, 0)
        mask = np.load(self.label_paths[idx]).transpose(1, 2, 3, 0)
        if dataset == 'dm':
            mask[mask > 0] = 1

        image = torch.tensor(image.astype(np.float32)).unsqueeze(0)
        image = F.interpolate(torch.tensor(image), size=(standard_shape), mode='trilinear', align_corners=False).squeeze(0)
        if self.train:
            mask = torch.tensor(mask.astype(np.float32)).unsqueeze(0)
            mask = F.interpolate(torch.tensor(mask), size=(standard_shape), mode='trilinear', align_corners=False).squeeze(0)
        sample = {'image': image, 'label': mask}
        if self.transforms:
            sample = self.transforms(sample)
        if self.train:
            label = sample['label'].as_tensor()
        else:
            label = torch.tensor(sample['label'].astype(np.float32))
            image = sample['image']

        if dataset == 'ispy2':
            if label.shape[0] > 1:
                image, label = image[0].unsqueeze(0), label[0].unsqueeze(0)
            image = image.repeat(3, 1, 1, 1)
            label = label.repeat(3, 1, 1, 1)
        if dataset == 'ispy1':
            label = label.repeat(3, 1, 1, 1)

        return image, label, dataset

def main():
    save_dir = 'weights/'
    set_determinism(seed=0)

    test_transform = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    data_dirs = ['data/ISPY1/', 'data/BreastDM/', 'data/DUKE/', 'data/RIDER/']
    all_image_paths, all_mask_paths = [], []
    individual_image_paths, individual_mask_paths = [], []
    for data_dir in data_dirs:
        images_path = os.path.join(data_dir, 'images')
        if data_dir == 'data/BreastDM/':
            images_st_path = os.path.join(data_dir, 'images')
        else:
            images_st_path = os.path.join(data_dir, 'images_std')
        masks_path = os.path.join(data_dir, 'masks')
        image_paths = os.listdir(images_path)
        image_st_paths = [os.path.join(images_st_path, image_path) for image_path in image_paths]
        mask_paths = [os.path.join(masks_path, image_path) for image_path in image_paths]
        image_paths = [os.path.join(images_path, image_path) for image_path in image_paths]

        all_image_paths.extend(image_st_paths)
        all_mask_paths.extend(mask_paths)
        individual_image_paths.append(image_st_paths)
        individual_mask_paths.append(mask_paths)

    train_ds0 = DatasetMRI(individual_image_paths[0], individual_mask_paths[0], test_transform, train=False)
    train_ds1 = DatasetMRI(individual_image_paths[1], individual_mask_paths[1], test_transform, train=False)
    train_ds2 = DatasetMRI(individual_image_paths[2], individual_mask_paths[2], test_transform, train=False)
    train_ds3 = DatasetMRI(individual_image_paths[3], individual_mask_paths[3], test_transform, train=False)    
    train_loader0 = DataLoader(train_ds0, batch_size=1, shuffle=False)
    train_loader1 = DataLoader(train_ds1, batch_size=1, shuffle=False)
    train_loader2 = DataLoader(train_ds2, batch_size=1, shuffle=False) 
    train_loader3 = DataLoader(train_ds3, batch_size=1, shuffle=False) 
    individual_test_loaders = [train_loader0, train_loader1, train_loader2, train_loader3]
    individual_names = ['ISPY1', 'BreastDM', 'DUKE', 'RIDER']

    device = torch.device("cuda:3")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=3,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    checkpoint_path = 'weights/SegResNet.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    torch.backends.cudnn.benchmark = True
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # TEST
    model.eval()
    with torch.no_grad():
        for test_loader in individual_test_loaders:
            for i, (image, label, dataset) in enumerate(test_loader):
                if dataset[0] == 'duke':
                    image = image.repeat(1, 3, 1, 1, 1)
                elif dataset[0] == 'rider':
                    image = image[0, :3].unsqueeze(0)
                if i > 5 and dataset[0] != 'duke':
                    break
                elif i > 20 and dataset[0] == 'duke':
                    break
                output = model(image.to(device))
                output = F.interpolate(output, size=(list(label.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                output = post_trans(output)

                output = F.interpolate(output, size=(list(image.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                label = F.interpolate(label, size=(list(image.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                
                overlap_ratios = torch.sum(output * label, dim=(0, 1, 2, 3)) / (torch.sum(output, dim=(0, 1, 2, 3)) + torch.sum(label, dim=(0, 1, 2, 3)) - torch.sum(output * label, dim=(0, 1, 2, 3)))
                # Find the index with the maximum overlap ratio
                valid_indices = torch.nonzero(overlap_ratios < 1, as_tuple=False).squeeze()
                max_overlap_index = valid_indices[torch.argmax(overlap_ratios[valid_indices])]

                fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                ax[0].imshow(image[0, 0, :, :, max_overlap_index], cmap='gray')
                ax[0].set_title('Image', fontsize=26)
                ax[1].imshow(output[0, 0, :, :, max_overlap_index], cmap='gray')
                ax[1].set_title('Model Output', fontsize=26)
                ax[2].imshow(label[0, 0, :, :, max_overlap_index], cmap='gray')
                ax[2].set_title('Expert Annotation', fontsize=26)
                dataset_name = dataset[0].upper()
                if dataset[0].upper() == 'DM':
                    dataset_name = 'BreastDM'
                plt.suptitle(f'SegResNet output on {dataset_name}', fontsize=32)
                plt.savefig(f'outputs/{dataset}_{i}.png')
                plt.close()
                print(f'{dataset}_{i}.png')

if __name__ == "__main__":
    main()