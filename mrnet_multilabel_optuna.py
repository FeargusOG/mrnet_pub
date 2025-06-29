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
import optuna

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
        df = pd.read_csv(os.path.join(root_dir, f"{subset}-{name}.csv"), header=None, names=["id", name])
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
    array = np.load(x["image"])
    array = np.expand_dims(array, axis=1)
    array = np.repeat(array, 3, axis=1)
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
    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=0)
        weighted = x * weights
        return weighted.sum(dim=0, keepdim=True)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 512, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        attn = self.conv(x)
        B, H, W = attn.shape
        attn_flat = attn.view(B, -1)
        attn_soft = self.softmax(attn_flat)
        max_vals = attn_soft.max(dim=1, keepdim=True)[0]
        attn_norm = attn_soft / (max_vals + 1e-6)
        attn_final = attn_norm.view(B, H, W)
        return x * attn_final

###########################
#         Model           #
###########################
class Net(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.slice_attention = SliceAttention(512)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, 3)
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = []
        for slice_img in x:
            feat = self.backbone(slice_img.unsqueeze(0)).squeeze(0)
            feat = self.spatial_attention(feat)
            pooled = self.pool(feat).view(-1)
            features.append(pooled)
        features = torch.stack(features)
        attended = self.slice_attention(features)
        dropped = self.dropout(attended)
        return self.classifier(dropped)

###########################
#       Training Loop     #
#        with Optuna      #
###########################
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.7)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

    root = "/mnt/8TB/adoloc/MRNet/MRNet-v1.0"
    plane = "sagittal"
    batch_size = 1
    epochs = 5
    num_workers = 8

    train_data = load_mrnet_data(root, plane, train=True)
    val_data = load_mrnet_data(root, plane, train=False)

    train_transforms = Compose([
        RandAffined(keys="image", prob=1.0, rotate_range=(0, 0, np.pi / 8),
                    translate_range=(0.1, 0.1, 0.0), scale_range=(0.1, 0.1, 0.0), padding_mode="border"),
        RandFlipd(keys="image", spatial_axis=0, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.1),
        RandScaleIntensityd(keys="image", prob=0.5, factors=0.2),
        RandShiftIntensityd(keys="image", prob=0.5, offsets=0.1),
        RandZoomd(keys="image", prob=0.3, min_zoom=0.9, max_zoom=1.1, mode="bilinear", padding_mode="edge", keep_size=True),
        ToTensord(keys=["image", "label"])
    ])
    val_transforms = Compose([ToTensord(keys=["image", "label"])])

    train_ds = Dataset(data=train_data, transform=Compose([load_numpy, train_transforms]))
    val_ds = Dataset(data=val_data, transform=Compose([load_numpy, val_transforms]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = get_best_device()
    model = Net(dropout=dropout).to(device)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_df = pd.DataFrame(train_data)
    labels = np.stack(train_df["label"].values)
    label_sums = labels.sum(axis=0)
    label_counts = len(labels) - label_sums
    pos_weight = torch.tensor(label_counts / label_sums, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        y_trues, y_preds, losses = [], [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x = batch["image"].to(device).float()
            y = batch["label"].to(device).float()
            optimizer.zero_grad()
            pred = model(x).squeeze(0)
            y = y.view(-1)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            prob = torch.sigmoid(pred).detach().cpu().numpy().flatten()
            y_preds.append(prob)
            y_trues.append(y.cpu().numpy().flatten())
            losses.append(loss.item())
        train_auc = [metrics.roc_auc_score(np.vstack(y_trues)[:, i], np.vstack(y_preds)[:, i]) for i in range(3)]

        # Validation
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

        val_auc = [metrics.roc_auc_score(np.vstack(y_trues)[:, i], np.vstack(y_preds)[:, i]) for i in range(3)]
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Plane: {plane}")
        print(f"Train Loss: {np.mean(losses):.4f} | Val Loss: {np.mean(val_losses):.4f}")
        print(f"Train AUCs - Abnormal: {train_auc[0]:.4f}, ACL: {train_auc[1]:.4f}, Meniscus: {train_auc[2]:.4f}")
        print(f"Val   AUCs - Abnormal: {val_auc[0]:.4f}, ACL: {val_auc[1]:.4f}, Meniscus: {val_auc[2]:.4f}")
        print("-" * 60)

    return np.mean(val_auc)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    print(study.best_trial)
