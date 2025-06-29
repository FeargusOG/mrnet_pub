import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.visualize import GradCAMpp
from monai.transforms import Compose, ToTensord
from monai.data import Dataset, DataLoader
from mrnet_attention_multilabel import Net
import torch.nn.functional as F

class NetForCAM(Net):
    def forward(self, x):
        feat = self.backbone(x)[0]  # (512, H, W)
        feat = self.spatial_attention(feat)
        pooled = self.pool(feat).view(1, -1)
        dropped = self.dropout(pooled)
        return self.classifier(dropped)
    
def load_mrnet_data(root_dir, plane):
    folder = os.path.join(root_dir, "valid", plane)
    label_csv = os.path.join(root_dir, "valid-acl.csv")
    df = np.loadtxt(label_csv, delimiter=",", dtype=str)
    samples = []
    for pid, label in df:
        path = os.path.join(folder, f"{pid.zfill(4)}.npy")
        label_vec = np.zeros(3, dtype=np.float32)
        label_vec[target_index] = float(label)
        samples.append({"image": path, "label": label_vec, "id": pid})
    return samples

def load_target_sample(root_dir, plane, target_id):
    target_id = str(target_id)
    folder = os.path.join(root_dir, "valid", plane)
    path = os.path.join(folder, f"{target_id.zfill(4)}.npy")
    label_vec = np.zeros(3, dtype=np.float32)
    label_vec[target_index] = 1.0
    return [{"image": path, "label": label_vec, "id": target_id}]

def load_numpy(x):
    arr = np.load(x["image"])  # (slices, 256, 256)
    arr = np.stack([arr] * 3, axis=1)  # (slices, 3, 256, 256)
    x["image"] = arr
    return x

# config
root_dir = "data"
plane = "sagittal"
model_path = "runs/baseline_sagittal/best_model.pt"
target_index = 1  # ACL class
target_id = 1207 # Set to None to auto find the best sample
force_slice_idx = 16  # Set to None to auto select the most activated slice.

# slect the device
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Usng device: {device}")

if target_id == None:
    val_data = load_mrnet_data(root_dir, plane)
    val_ds = Dataset(data=val_data, transform=Compose([load_numpy, ToTensord(keys=["image", "label"])]))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # load up the pretrained model
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # find the sample with the highest prob score
    best_conf, best_batch = -1, None
    for batch in val_loader:
        x = batch["image"].to(device).float()
        x.requires_grad = True
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)[0, target_index].item()
        if batch["label"][0][target_index] == 1 and prob > best_conf:
            best_conf = prob
            best_batch = batch

    print(f"Best ACL prob: {best_conf:.4f}, Sample ID: {best_batch['id'][0]}")
    target_id = best_batch['id'][0]

# Data loader 
data = load_target_sample(root_dir, plane, target_id)
ds = Dataset(data=data, transform=Compose([load_numpy, ToTensord(keys=["image", "label"])]))
loader = DataLoader(ds, batch_size=1, shuffle=False)

# Load model 
model_for_cam = NetForCAM().to(device)
model_for_cam.load_state_dict(torch.load(model_path, map_location=device))
model_for_cam.eval()

cam = GradCAMpp(nn_module=model_for_cam, target_layers="backbone.7.1.conv2")

#  Get batch and compute per-slice activations 
batch = next(iter(loader))
x = batch["image"].to(device).float()

slice_scores = []
cam_maps = []

for i in range(x.shape[1]):
    slice_input = x[0, i].unsqueeze(0).to(device)
    cam_map = cam(x=slice_input, class_idx=target_index)[0][0]
    cam_map_resized = F.interpolate(
        cam_map.unsqueeze(0).unsqueeze(0), size=(256, 256),
        mode="bilinear", align_corners=False
    )[0, 0]
    cam_maps.append(cam_map_resized)
    slice_scores.append(cam_map_resized.mean().item())

#  Print ranked slices by average CAM activation 
sorted_indices = np.argsort(slice_scores)[::-1]  # descending
print("Slice indices ranked by mean CAM activation:")
for rank, idx in enumerate(sorted_indices):
    print(f"{rank+1:2d}: Slice {idx} - Mean activation: {slice_scores[idx]:.4f}")


#  Select best slice 
best_slice_idx = int(np.argmax(slice_scores)) if force_slice_idx is None else force_slice_idx
print(f"Using slice index: {best_slice_idx}")

best_slice_input = x[0, best_slice_idx].unsqueeze(0)
logits = model_for_cam(best_slice_input)
prob = torch.sigmoid(logits)[0, target_index].item()
print(f"ID: {target_id} - ACL Probability: {prob:.4f}")

#  Normalise CAM 
best_cam = cam_maps[best_slice_idx]
cam_map_norm = (best_cam - best_cam.min()) / (best_cam.max() - best_cam.min())

#  Plot 
best_img = x[0, best_slice_idx].permute(1, 2, 0).cpu().numpy()
best_img_norm = best_img / best_img.max()

fig, axs = plt.subplots(1, 2, figsize=(14, 8))

axs[0].imshow(best_img_norm, cmap='gray')
axs[0].set_title(f"Sample {target_id}, Slice {best_slice_idx}")
axs[0].axis("off")

axs[1].imshow(best_img_norm, cmap='gray')
axs[1].imshow(cam_map_norm.cpu().numpy(), cmap="jet_r", alpha=0.5)
axs[1].set_title("Grad-CAM++ Overlay (ACL)")
axs[1].axis("off")

fig.suptitle("Grad-CAM++ Visualisation for ACL Tear Detection, Sagittal Plane, Tear Present", fontsize=16)

plt.tight_layout()
plt.show()
