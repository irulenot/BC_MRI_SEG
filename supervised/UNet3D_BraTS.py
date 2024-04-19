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
wandb.init(project="ichi2024", name="UNet3D_BraTS")


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
    root_dir = 'data/MONAI/'
    save_dir = 'weights/'
    print(root_dir)

    set_determinism(seed=0)

    class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
        """
        Convert labels to multi channels based on brats classes:
        label 1 is the peritumoral edema
        label 2 is the GD-enhancing tumor
        label 3 is the necrotic and non-enhancing tumor core
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).

        """

        def __call__(self, data):
            d = dict(data)
            for key in self.keys:
                result = []
                # merge label 2 and label 3 to construct TC
                result.append(torch.logical_or(d[key] == 2, d[key] == 3))
                # merge labels 1, 2 and 3 to construct WT
                result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
                # label 2 is ET
                result.append(d[key] == 2)
                d[key] = torch.stack(result, axis=0).float()
            return d
        
    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=[256, 256, 128], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )

    # here we don't cache any data in case out of memory issue
    train_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=train_transform,
        section="training",
        download=True,
        cache_rate=0.0,
        num_workers=4,
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_ds = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        transform=val_transform,
        section="validation",
        download=False,
        cache_rate=0.0,
        num_workers=4,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
    
    max_epochs = 300
    device = torch.device("cuda:1")
    model = UNet(
        spatial_dims=3,
        in_channels=4,
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
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    # TRAIN
    best_metric, best_metric2, best_metric3 = -1, -1, -1
    best_metric_epoch = -1
    total_start = time.time()
    for epoch in tqdm(range(max_epochs)):
        model.train()
        epoch_loss, step = 0, 0
        for batch_data in train_loader:
            step += 1
            optimizer.zero_grad()
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            inputs = F.interpolate(inputs, size=(standard_shape), mode='trilinear', align_corners=False)
            labels = F.interpolate(labels, size=(standard_shape), mode='trilinear', align_corners=False)
            outputs = model(inputs)  # Inputs: (4, 224, 224, 144), 4 channels, 144 slices
            loss = loss_function(outputs, labels)
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
            for val_data in val_loader:
                inputs, labels = (
                    val_data["image"],
                    val_data["label"],
                )
                inputs = F.interpolate(inputs, size=(standard_shape), mode='trilinear', align_corners=False)
                output = model(inputs.to(device))
                output = F.interpolate(output, size=(list(labels.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                output = post_trans(output)
                labels = post_label(labels)
                dice_metric(y_pred=output, y=labels)
                dice_metric_batch(y_pred=output, y=labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()
            print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}")

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, "UNet3D_BraTS.pth"),
                )
                print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
                
            dice_metric.reset()
            dice_metric_batch.reset()
            wandb.log({"epoch": epoch, "average_train_loss": epoch_loss,
                "val_dice": metric})

    # FINAL RESULTS
    total_time = time.time() - total_start
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    print(
        f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
        f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
        f"\nbest mean dice: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
    )

    print('learnable_parameters', count_learnable_parameters(model))
    print('all_parameters', count_all_parameters(model))
    wandb.log({"total_time": total_time,
               'all_parameters': count_all_parameters(model),
               'learnable_parameters': count_learnable_parameters(model)})

if __name__ == "__main__":
    main()