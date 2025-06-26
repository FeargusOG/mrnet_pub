import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all(42)
set_determinism(seed=42)

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
        self.pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.soft = nn.Softmax(2)
        self.classifer = nn.Linear(1000, 1)

    def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        if torch.cuda.is_available():
            a = a.cuda()
            order_index = order_index.cuda()
        return torch.index_select(a, dim, order_index)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.pretrained_model.conv1(x)
        x = self.pretrained_model.bn1(x)
        x = self.pretrained_model.maxpool(x)
        x = self.pretrained_model.layer1(x)
        x = self.pretrained_model.layer2(x)
        x = self.pretrained_model.layer3(x)
        x = self.pretrained_model.layer4(x)
        attention = self.conv(x)
        attention =  self.soft(attention.view(*attention.size()[:2], -1)).view_as(attention)
        maximum = torch.max(attention.flatten(2), 2).values
        maximum = Net.tile(maximum, 1, attention.shape[2]*attention.shape[3])
        attention_norm = attention.flatten(2).flatten(1) / maximum
        attention_norm= torch.reshape(attention_norm, (attention.shape[0],attention.shape[1],attention.shape[2],attention.shape[3]))
        o = x*attention_norm
        out= self.pretrained_model.avgpool(o)
        out = self.pretrained_model.fc(out.squeeze())
        output = torch.max(out, 0, keepdim=True)[0]
        output = self.classifer(output)

        return output

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

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, generator=torch.Generator().manual_seed(42))

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.3, threshold=1e-4)

######################
#   Train and Eval   #
######################
best_val_auc = 0
early_stop = 0
trigger = 6

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
