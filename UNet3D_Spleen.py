from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
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
import numpy as np
import wandb
wandb.init(project="ichi2024", name="UNet_Spleen")


standard_shape = (256, 256, 128)
def main():
    save_dir = 'weights/'
    root_dir = 'data/MONAI/'
    weight_dir = 'weights/'
    print(root_dir)
    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
    md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

    compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
    data_dir = os.path.join(root_dir, "Task09_Spleen")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)

    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    set_determinism(seed=0)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ]
    )

    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    device = torch.device("cuda:2")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    max_epochs = 600
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    iou_metric = MeanIoU(include_background=True, reduction="mean")
    tpf_metric = ConfusionMatrixMetric(metric_name="sensitivity", reduction="mean")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    val_interval = 2
    best_metric, best_metric2, best_metric3 = -1, -1, -1
    best_metric_epoch = -1
    epoch_loss_values = []
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
            input, label = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            input = F.interpolate(input, size=(standard_shape), mode='trilinear', align_corners=False)
            label = F.interpolate(label, size=(standard_shape), mode='trilinear', align_corners=False)
            output = model(input)
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
            for val_data in val_loader:
                input, label = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                input = F.interpolate(input, size=(standard_shape), mode='trilinear', align_corners=False)
                output = model(input)
                output = F.interpolate(output, size=(list(label.shape)[2:]), mode='trilinear', align_corners=False).cpu()
                output = post_trans(output).cpu()
                label = post_label(label).cpu()
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
                    os.path.join(save_dir, "UNet_ISPY1_spleen.pth"),
                )
                print("saved new best metric model")
                print(
                    f"best mean dice: {best_metric:.4f}"
                    f"\nbest mean iou: {best_metric2:.4f}"
                    f"\nbest mean tpf: {best_metric3:.4f}"
                    f"\n at epoch: {best_metric_epoch}"
                )
                
            dice_metric.reset()
            iou_metric.reset()
            tpf_metric.reset()
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