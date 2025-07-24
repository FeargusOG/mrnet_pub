import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from monai.transforms import (
    Compose, RandAffined, RandFlipd, ToTensord, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandZoomd
)
from monai.utils import set_determinism
from monai.data import Dataset
from sklearn import metrics
from tqdm import tqdm
import random

from torch.utils.tensorboard import SummaryWriter

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)
set_determinism(seed=42)

###########################
#       Utilities         #
###########################

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_mrnet_data(root_dir, plane, train=True):
    subset = "train" if train else "valid"
    folder = os.path.join(root_dir, subset, plane)

    label_names = ['abnormal', 'acl', 'meniscus']
    records = {}

    for name in label_names:
        df = pd.read_csv(os.path.join(root_dir, f"{subset}-{name}.csv"),
                         header=None, names=["id", name])
        df["id"] = df["id"].astype(str).str.zfill(4)
        records[name] = df.set_index("id")

    ids = records['abnormal'].index.tolist()

    data = []
    for id_ in ids:
        path = os.path.join(folder, f"{id_}.npy")
        label = np.array([
            records["abnormal"].loc[id_]["abnormal"],
            records["acl"].loc[id_]["acl"],
            records["meniscus"].loc[id_]["meniscus"]
        ], dtype=np.float32)
        data.append({"image": path, "label": label})

    return data

def load_numpy(x):
    array = np.load(x["image"])  # (slices, 256, 256)
    array = np.expand_dims(array, axis=1)  # (slices, 1, 256, 256)
    array = np.repeat(array, 3, axis=1)     # (slices, 3, 256, 256)
    x["image"] = array
    return x

###########################
#    Attention Modules    #
###########################

class SliceAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: (slices, 512)
        scores = self.attn(x)               # (slices, 1)
        weights = torch.softmax(scores, dim=0)  # (slices, 1)
        weighted = x * weights              # (slices, 512)
        return weighted.sum(dim=0, keepdim=True)  # (1, 512)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # x: (512, H, W)
        attn = self.conv(x)                            # (512, H, W)
        B, H, W = attn.shape                           # (512, H, W)

        attn_flat = attn.view(B, -1)                   # (512, H*W)
        attn_soft = self.softmax(attn_flat)            # (512, H*W)

        max_vals = attn_soft.max(dim=1, keepdim=True)[0]  # (512, 1)
        attn_norm = attn_soft / (max_vals + 1e-6)         # (512, H*W), avoid div-by-zero

        attn_final = attn_norm.view(B, H, W)           # (512, H, W)
        return x * attn_final                          # (512, H, W)

###########################
#         Model           #
###########################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to layer4 (no avgpool or fc)
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.slice_attention = SliceAttention(input_dim=512)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, 3)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # (slices, 3, 256, 256)
        features = []
        for slice_img in x:  # each slice_img: (3, 256, 256)
            feat = self.backbone(slice_img.unsqueeze(0)).squeeze(0)  # (512, H, W)
            feat = self.spatial_attention(feat)  # (512, H, W)
            pooled = self.pool(feat).view(-1)   # (512,)
            features.append(pooled)

        features = torch.stack(features)        # (slices, 512)
        attended = self.slice_attention(features)  # (1, 512)
        dropped = self.dropout(attended)
        return self.classifier(dropped)         # (1, 3)

###########################
#       Training Loop     #
###########################

def main():
    #  Params # 
    root = "/mnt/8TB/fogorman/mrnet_pub/data"
    plane = "sagittal"
    run_title = "baseline"
    save_dir = f"runs/{run_title}_{plane}"
    os.makedirs(save_dir, exist_ok=True)
    
    lr = 1e-5
    epochs = 50
    batch_size = 1
    num_workers = 8
    TRAIN_N = None

    #  Setup # 
    train_transforms = Compose([
        RandAffined(
            keys="image",
            prob=1.0,
            rotate_range=(0, 0, np.pi / 8),
            translate_range=(0.1, 0.1, 0.0),
            scale_range=(0.1, 0.1, 0.0),
            padding_mode="border"
        ),
        RandFlipd(keys="image", spatial_axis=0, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.1),
        RandScaleIntensityd(keys="image", prob=0.5, factors=0.2),
        RandShiftIntensityd(keys="image", prob=0.5, offsets=0.1),
        RandZoomd(keys="image", prob=0.3, min_zoom=0.9, max_zoom=1.1, mode="bilinear", padding_mode="edge", keep_size=True),
        ToTensord(keys=["image", "label"])
    ])

    val_transforms = Compose([
        ToTensord(keys=["image", "label"])
    ])

    train_data = load_mrnet_data(root, plane, train=True)
    if TRAIN_N is not None:
        train_data = train_data[:TRAIN_N]
    val_data = load_mrnet_data(root, plane, train=False)

    train_ds = Dataset(data=train_data, transform=Compose([load_numpy, train_transforms]))
    val_ds = Dataset(data=val_data, transform=Compose([load_numpy, val_transforms]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=torch.Generator().manual_seed(42))

    device = get_best_device()
    print(f"\n******* DEVICE - {device} *******\n")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=4, 
                                                           factor=0.3, 
                                                           threshold=1e-4)

    # Class weights
    train_df = pd.DataFrame(train_data)
    labels = np.stack(train_df["label"].values)
    label_sums = labels.sum(axis=0)             # count of positives per class
    label_counts = len(labels) - label_sums     # count of negatives per class
    pos_weight = torch.tensor(label_counts / label_sums, dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = 0
    early_stop = 0
    trigger = 10

    writer = SummaryWriter(log_dir=save_dir)

    for epoch in range(epochs):
        model.train()
        y_trues, y_preds, losses = [], [], []

        #  TRAIN # 
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x = batch["image"].to(device).float()  # (1, s, 3, 256, 256)
            y = batch["label"].to(device).float()  # (1, 3)

            optimizer.zero_grad()
            pred = model(x).squeeze(0)             # (3,)
            y = y.view(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            prob = torch.sigmoid(pred).detach().cpu().numpy().flatten()
            y_preds.append(prob)
            y_trues.append(y.cpu().numpy().flatten())
            losses.append(loss.item())

        y_preds = np.vstack(y_preds)
        y_trues = np.vstack(y_trues)
        train_auc = [metrics.roc_auc_score(y_trues[:, i], y_preds[:, i]) for i in range(3)]
        train_loss = np.mean(losses)

        #  VALIDATION # 
        model.eval()
        y_trues, y_preds, val_losses = [], [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                x = batch["image"].to(device).float()
                y = batch["label"].to(device).float()

                pred = model(x).squeeze(0)
                y = y.view(-1)
                loss = criterion(pred, y)

                prob = torch.sigmoid(pred).cpu().numpy().flatten()
                y_preds.append(prob)
                y_trues.append(y.cpu().numpy().flatten())
                val_losses.append(loss.item())

        y_preds = np.vstack(y_preds)
        y_trues = np.vstack(y_trues)
        val_auc = [metrics.roc_auc_score(y_trues[:, i], y_preds[:, i]) for i in range(3)]
        val_loss = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Plane: {plane}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train AUCs - Abnormal: {train_auc[0]:.4f}, ACL: {train_auc[1]:.4f}, Meniscus: {train_auc[2]:.4f}, Mean {np.mean(train_auc)}")
        print(f"Val   AUCs - Abnormal: {val_auc[0]:.4f}, ACL: {val_auc[1]:.4f}, Meniscus: {val_auc[2]:.4f}, Mean {np.mean(val_auc)}")
        print("-" * 60)

        # Report metrics to tensorboard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        writer.add_scalars("AUC/Abnormal", {
            "Train": train_auc[0],
            "Val": val_auc[0],
        }, epoch)
        
        writer.add_scalars("AUC/ACL", {
            "Train": train_auc[1],
            "Val": val_auc[1],
        }, epoch)

        writer.add_scalars("AUC/Meniscus", {
            "Train": train_auc[2],
            "Val": val_auc[2],
        }, epoch)

        mean_val_auc = np.mean(val_auc)
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            early_stop = 0
            print(f"Current Best Val AUC: {best_val_auc}")
            torch.save(model.state_dict(), save_dir+"/best_model.pt")
        else:
            early_stop += 1

        scheduler.step(val_loss)

        if early_stop == trigger:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    writer.close()

if __name__ == "__main__":
    main()
