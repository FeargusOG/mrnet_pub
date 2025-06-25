import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from monai.transforms import Compose, RandAffined, RandFlipd, ToTensord
from monai.data import Dataset
from sklearn import metrics
from tqdm import tqdm

######################
#       Utils        #
######################
def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_mrnet_data(root_dir, task, plane, train=True):
    subset = "train" if train else "valid"
    folder = os.path.join(root_dir, subset, plane)
    csv_file = os.path.join(root_dir, f"{subset}-{task}.csv")

    df = pd.read_csv(csv_file, header=None, names=["id", "label"])
    df["id"] = df["id"].astype(str).str.zfill(4)

    data = []
    for _, row in df.iterrows():
        path = os.path.join(folder, f"{row['id']}.npy")
        label = np.float32([row["label"]])
        data.append({"image": path, "label": label})
    return data

######################
#       Model        #
######################
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.classifier = nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # (slices, 3, 256, 256)
        x = self.pretrained_model(x)
        x = torch.max(x, dim=0, keepdim=True)[0]  # (1, 1000)
        x = F.relu(x)
        return self.classifier(x)  # (1, 1)

######################
#       Params       #
######################
root = "./data/"
task = "meniscus"
plane = "coronal"
lr = 1e-5
epochs = 50
batch_size = 1
num_workers = 2
TRAIN_N = None  # dev subsample, set to None to use all

######################
#       Setup        #
######################
device = get_best_device()
# GPU = 1
# torch.cuda.set_device(GPU)
# device = torch.device(f"cuda:{GPU}")
print(f"\n******* DEVICE - {device} *******\n")

train_transforms = Compose([
    RandAffined(keys="image", prob=1.0, rotate_range=(0, 0, np.pi/8), translate_range=(0.1, 0.1, 0.0)),
    RandFlipd(keys="image", spatial_axis=0, prob=0.5),
    ToTensord(keys=["image", "label"])
])

val_transforms = Compose([
    ToTensord(keys=["image", "label"])
])

train_data = load_mrnet_data(root, task, plane, train=True)
labels = [int(x["label"][0]) for x in train_data]
pos = sum(labels)
neg = len(labels) - pos
pos_weight = torch.tensor([neg / pos]).to(device) # Global calss weight.

if TRAIN_N is not None:
    train_data = train_data[:TRAIN_N]

val_data = load_mrnet_data(root, task, plane, train=False)

def load_numpy(x):
    array = np.load(x["image"])  # shape (slices, 256, 256)
    array = np.expand_dims(array, axis=1)  # shape (slices, 1, 256, 256)
    array = np.repeat(array, 3, axis=1)     # shape (slices, 3, 256, 256)
    x["image"] = array
    return x

train_ds = Dataset(data=train_data, transform=Compose([load_numpy, train_transforms]))
val_ds = Dataset(data=val_data, transform=Compose([load_numpy, val_transforms]))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.3, threshold=1e-4)

######################
#   Train and Eval   #
######################
best_val_auc = 0
early_stop = 0
trigger = 10

for epoch in range(epochs):
    model.train()
    y_trues, y_preds, losses = [], [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        x = batch["image"].to(device).float()
        y = batch["label"].to(device).float()

        optimizer.zero_grad()
        pred = model(x).squeeze(0)
        y = y.view(-1)
        loss = F.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        prob = torch.sigmoid(pred).detach().cpu().numpy()[0]
        y_preds.append(prob)
        y_trues.append(y.item())
        losses.append(loss.item())

    train_auc = metrics.roc_auc_score(y_trues, y_preds) if len(set(y_trues)) > 1 else 0.5

    model.eval()
    y_trues, y_preds, val_losses = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            x = batch["image"].to(device).float()
            y = batch["label"].to(device).float()
            pred = model(x).squeeze(0)
            y = y.view(-1)
            loss = F.binary_cross_entropy_with_logits(pred, y)
            prob = torch.sigmoid(pred).cpu().numpy()[0]

            y_preds.append(prob)
            y_trues.append(y.item())
            val_losses.append(loss.item())

    val_auc = metrics.roc_auc_score(y_trues, y_preds) if len(set(y_trues)) > 1 else 0.5
    val_loss = np.mean(val_losses)

    print(f"{task}[{plane}] - epoch: {epoch+1} | train loss: {np.mean(losses):.4f} | train auc: {train_auc:.4f} | val loss: {val_loss:.4f} | val auc: {val_auc:.4f}")
    print("-" * 30)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        early_stop = 0
    else:
        early_stop += 1

    scheduler.step(val_loss)

    if early_stop == trigger:
        print(f"Early stopping after {epoch + 1} epochs")
        break
