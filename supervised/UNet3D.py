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
import wandb
wandb.init(project="ichi2024", name="UNet3D")


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

        return image, label

def main():
    save_dir = 'weights/'
    set_determinism(seed=0)
        
    train_transform = Compose(
        [
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            RandSpatialCropd(keys=["image", "label"], roi_size=[256, 256, 128], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    test_transform = Compose(
        [
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    data_dirs = ['data/ISPY1/', 'data/BreastDM/']
    all_train_image_paths, all_test_image_paths, all_train_mask_paths, all_test_mask_paths = [],[],[],[]
    individual_test_image_paths, individual_test_mask_paths = [], []
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

        train_indices, test_indices = train_test_split(range(len(image_paths)), test_size=0.2)
        train_image_paths = [image_st_paths[i] for i in train_indices]
        test_image_paths = [image_st_paths[i] for i in test_indices]
        train_mask_paths = [mask_paths[i] for i in train_indices]
        test_mask_paths = [mask_paths[i] for i in test_indices]

        all_train_image_paths.extend(train_image_paths)
        all_test_image_paths.extend(test_image_paths)
        all_train_mask_paths.extend(train_mask_paths)
        all_test_mask_paths.extend(test_mask_paths)
        individual_test_image_paths.append(test_image_paths)
        individual_test_mask_paths.append(test_mask_paths)

    train_ds = DatasetMRI(all_train_image_paths, all_train_mask_paths, train_transform, train=True)
    test_ds = DatasetMRI(all_test_image_paths, all_test_mask_paths, test_transform, train=False)
    test_ds0 = DatasetMRI(individual_test_image_paths[0], individual_test_mask_paths[0], test_transform, train=False)
    test_ds1 = DatasetMRI(individual_test_image_paths[1], individual_test_mask_paths[1], test_transform, train=False)
    # test_ds2 = DatasetMRI(individual_test_image_paths[2], individual_test_mask_paths[2], test_transform, train=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    test_loader0 = DataLoader(test_ds0, batch_size=1, shuffle=False)
    test_loader1 = DataLoader(test_ds1, batch_size=1, shuffle=False) 
    # test_loader2 = DataLoader(test_ds2, batch_size=1, shuffle=False) 
    individual_test_loaders = [test_loader0, test_loader1]
    individual_names = ['ISPY1', 'BreastDM']

    max_epochs = 300
    device = torch.device("cuda:2")
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    def count_learnable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('learnable_parameters', count_learnable_parameters(model))
    def count_all_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print('all_parameters', count_all_parameters(model))

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    iou_metric_batch = MeanIoU(include_background=True, reduction="mean_batch")
    tpf_metric = ConfusionMatrixMetric(metric_name="sensitivity", reduction="mean")
    tpf_metric_batch = ConfusionMatrixMetric(metric_name="sensitivity", reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # TRAIN
    best_metric, best_metric2, best_metric3 = -1, -1, -1
    best_metric_epoch = -1
    total_start = time.time()
    for epoch in tqdm(range(max_epochs)):
        model.train()
        epoch_loss, step = 0, 0
        for image, label in train_loader:
            step += 1
            optimizer.zero_grad()
            output = model(image.to(device))  # (B, C, H, W, D)
            loss = loss_function(output, label.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        lr_scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # TEST
        model.eval()
        with torch.no_grad():
            for image, label in test_loader:
                output = model(image.to(device))
                output = F.interpolate(output, size=(list(label.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                output = post_trans(output)
                dice_metric(y_pred=output, y=label)
                iou_metric(y_pred=output, y=label)
                tpf_metric(y_pred=output, y=label)
            metric = dice_metric.aggregate().item()
            avg_dice = metric
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                metric2 = iou_metric.aggregate().item()
                metric3 = tpf_metric.aggregate()[0].item()
                best_metric2 = metric2
                best_metric3 = metric3
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "UNet3D.pth"),
                )
                print(f"saved new best metric model at epoch: {best_metric_epoch}")
                print(
                    f"best mean dice: {best_metric:.4f}"
                    f"\nbest mean iou: {best_metric2:.4f}"
                    f"\nbest mean tpf: {best_metric3:.4f}"
                )
                
                for i, individual_test_loader in enumerate(individual_test_loaders):
                    dice_metric.reset()
                    dice_metric_batch.reset()
                    iou_metric.reset()
                    iou_metric_batch.reset()
                    tpf_metric.reset()
                    tpf_metric_batch.reset()

                    dataset = individual_names[i]
                    for image, label in individual_test_loader:
                        output = model(image.to(device))
                        output = F.interpolate(output, size=(list(label.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                        output = post_trans(output)
                        dice_metric(y_pred=output, y=label)
                        iou_metric(y_pred=output, y=label)
                        tpf_metric(y_pred=output, y=label)

                    metric = dice_metric.aggregate().item()
                    metric2 = iou_metric.aggregate().item()
                    metric3 = tpf_metric.aggregate()[0].item()
                    print(
                        f"{dataset}"
                        f"\nmean dice: {metric:.4f}"
                        f"\nmean iou: {metric2:.4f}"
                        f"\nmean tpf: {metric3:.4f}"
                    )
                
            dice_metric.reset()
            dice_metric_batch.reset()
            iou_metric.reset()
            iou_metric_batch.reset()
            tpf_metric.reset()
            tpf_metric_batch.reset()
            wandb.log({"epoch": epoch, "average_train_loss": epoch_loss,
                "val_dice": avg_dice})

    # FINAL RESULTS
    total_time = time.time() - total_start
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    print(
        f"best mean dice: {best_metric:.4f}"
        f"\nbest mean iou: {best_metric2:.4f}"
        f"\nbest mean tpf: {best_metric3:.4f}"
        f"\n at epoch: {best_metric_epoch}"
    )

    print('learnable_parameters', count_learnable_parameters(model))
    print('all_parameters', count_all_parameters(model))
    wandb.log({"total_time": total_time,
               'all_parameters': count_all_parameters(model),
               'learnable_parameters': count_learnable_parameters(model)})

if __name__ == "__main__":
    main()