import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from monai.transforms import Compose, RandAffined, RandFlipd, ToTensord
from monai.data import Dataset, CacheDataset
from sklearn import metrics
from tqdm import tqdm

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
#      Attention Layer    #
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

###########################
#         Model           #
###########################

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        self.attention = SliceAttention(input_dim=512)
        self.classifier = nn.Linear(512, 3)

    def forward(self, x):
        # x: (1, slices, 3, 256, 256)
        x = torch.squeeze(x, dim=0)  # (slices, 3, 256, 256)
        features = self.feature_extractor(x)  # (slices, 512)
        pooled = self.attention(features)     # (1, 512)
        return self.classifier(pooled)        # (1, 3)

###########################
#       Training Loop     #
###########################

def main():
    # --- Params ---
    root = "/mnt/8TB/adoloc/MRNet/MRNet-v1.0"
    plane = "sagittal"
    lr = 1e-5
    epochs = 50
    batch_size = 1
    TRAIN_N = None

    # --- Setup ---
    train_transforms = Compose([
        RandAffined(keys="image", prob=1.0, rotate_range=(0, 0, np.pi/8), translate_range=(0.1, 0.1, 0.0)),
        RandFlipd(keys="image", spatial_axis=0, prob=0.5),
        ToTensord(keys=["image", "label"])
    ])

    val_transforms = Compose([
        ToTensord(keys=["image", "label"])
    ])

    train_data = load_mrnet_data(root, plane, train=True)
    if TRAIN_N is not None:
        train_data = train_data[:TRAIN_N]
    val_data = load_mrnet_data(root, plane, train=False)

    train_ds = CacheDataset(data=train_data, transform=Compose([load_numpy, train_transforms]), cache_rate=1.0)
    val_ds = CacheDataset(data=val_data, transform=Compose([load_numpy, val_transforms]), cache_rate=1.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = get_best_device()
    print(f"\n******* DEVICE - {device} *******\n")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.3, threshold=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0
    early_stop = 0
    trigger = 10

    for epoch in range(epochs):
        model.train()
        y_trues, y_preds, losses = [], [], []

        # === TRAIN ===
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

        # === VALIDATION ===
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
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train AUCs - Abnormal: {train_auc[0]:.4f}, ACL: {train_auc[1]:.4f}, Meniscus: {train_auc[2]:.4f}")
        print(f"Val   AUCs - Abnormal: {val_auc[0]:.4f}, ACL: {val_auc[1]:.4f}, Meniscus: {val_auc[2]:.4f}")
        print("-" * 60)

        mean_val_auc = np.mean(val_auc)
        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            early_stop = 0
            torch.save(model.state_dict(), "best_model_attention.pt")
        else:
            early_stop += 1

        scheduler.step(val_loss)

        if early_stop == trigger:
            print(f"Early stopping after {epoch+1} epochs")
            break

if __name__ == "__main__":
    main()
