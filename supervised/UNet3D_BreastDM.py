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
wandb.init(project="ichi2024", name="UNet3D_BreastDM")


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
        image = np.load(self.data_paths[idx]).transpose(1, 2, 3, 0)
        mask = np.load(self.label_paths[idx]).transpose(1, 2, 3, 0)
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
        return image, label


def main():
    data_dir = 'data/BreastDM/'
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

    images_path = os.path.join(data_dir, 'images')
    masks_path = os.path.join(data_dir, 'masks')
    image_paths = os.listdir(images_path)
    mask_paths = sorted([os.path.join(masks_path, image_path) for image_path in image_paths])
    image_paths = sorted([os.path.join(images_path, image_path) for image_path in image_paths])

    train_indices, test_indices = train_test_split(range(len(image_paths)), test_size=0.2)
    train_image_paths = [image_paths[i] for i in train_indices]
    test_image_paths = [image_paths[i] for i in test_indices]
    train_mask_paths = [mask_paths[i] for i in train_indices]
    test_mask_paths = [mask_paths[i] for i in test_indices]

    train_ds = DatasetMRI(train_image_paths, train_mask_paths, train_transform, train=True)
    test_ds = DatasetMRI(test_image_paths, test_mask_paths, test_transform, train=False)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    max_epochs = 300
    device = torch.device("cuda:0")
    model = UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

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
    post_label = Compose([AsDiscrete(threshold=0.5)])

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
            output = model(image.to(device))  # (C, H, W, B)
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
                    os.path.join(save_dir, "UNet3D_BreastDM.pth"),
                )
                print("saved new best metric model")
                print(
                    f"best mean dice: {best_metric:.4f}"
                    f"\nbest mean iou: {best_metric2:.4f}"
                    f"\nbest mean tpf: {best_metric3:.4f}"
                    f"\n at epoch: {best_metric_epoch}"
                )
                
            dice_metric.reset()
            dice_metric_batch.reset()
            iou_metric.reset()
            iou_metric_batch.reset()
            tpf_metric.reset()
            tpf_metric_batch.reset()
            wandb.log({"epoch": epoch, "average_train_loss": epoch_loss,
                "val_dice": metric})

    # FINAL RESULTS
    total_time = time.time() - total_start
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    print(
        f"best mean dice: {best_metric:.4f}"
        f"\nbest mean iou: {best_metric2:.4f}"
        f"\nbest mean tpf: {best_metric3:.4f}"
        f"\n at epoch: {best_metric_epoch}"
    )

    def count_learnable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('learnable_parameters', count_learnable_parameters(model))
    def count_all_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print('all_parameters', count_all_parameters(model))

    wandb.log({"total_time": total_time,
               'all_parameters': count_all_parameters(model),
               'learnable_parameters': count_learnable_parameters(model)})

if __name__ == "__main__":
    main()