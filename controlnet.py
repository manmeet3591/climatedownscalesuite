import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn, optim
from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionControlNetPipeline
import numpy as np
from copy import deepcopy

# Define dataset class
class ncDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index]).unsqueeze(0)
        y = torch.from_numpy(self.targets[index]).unsqueeze(0)
        return x, y

    def __len__(self):
        return len(self.data)

# Patchify and Unpatchify helper functions
def patchify(img, patch_size):
    img_shape = img.shape
    patches = np.array([
        img[i:i + patch_size, j:j + patch_size]
        for i in range(0, img_shape[0], patch_size)
        for j in range(0, img_shape[1], patch_size)
    ])
    return patches

def unpatchify(patches, img_shape):
    patch_size = patches.shape[1]
    img = np.zeros(img_shape, dtype=patches.dtype)
    patch_idx = 0
    for i in range(0, img_shape[0], patch_size):
        for j in range(0, img_shape[1], patch_size):
            img[i:i + patch_size, j:j + patch_size] = patches[patch_idx]
            patch_idx += 1
    return img

# Dataset preparation
patch_size = 32
x_train = ds_viirs.values.astype(np.float32)
y_train = ds_dmsp.values[0, :, :].astype(np.float32)

x_train_max = x_train.max()
y_train_max = y_train.max()
x_train /= x_train_max
y_train /= y_train_max

x_train_patches = patchify(x_train[:576, :576], patch_size)
y_train_patches = patchify(y_train[:576, :576], patch_size)

x_val_patches = x_train_patches[200:300]
y_val_patches = y_train_patches[200:300]

x_test_patches = x_train_patches[300:]
y_test_patches = y_train_patches[300:]

x_train_patches = x_train_patches[:200]
y_train_patches = y_train_patches[:200]

train_dataset = ncDataset(x_train_patches, y_train_patches)
val_dataset = ncDataset(x_val_patches, y_val_patches)
test_dataset = ncDataset(x_test_patches, y_test_patches)

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)

# Initialize ControlNet and related components
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4")
device = "cuda" if torch.cuda.is_available() else "cpu"
controlnet.to(device)

optimizer = optim.Adam(controlnet.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
def train_controlnet(model, train_dataloader, val_dataloader, optimizer, criterion, device, num_epochs=50):
    model.train()
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        train_loss = 0.0
        for lr, hr in train_dataloader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()

            # ControlNet forward pass
            output = model(lr, hr)

            loss = criterion(output, hr)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for lr, hr in val_dataloader:
                lr, hr = lr.to(device), hr.to(device)
                output = model(lr, hr)
                val_loss += criterion(output, hr).item()
        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)

    return best_model

best_controlnet = train_controlnet(controlnet, train_dataloader, val_dataloader, optimizer, criterion, device)

# Save the best model
model_save_path = "best_controlnet.pth"
torch.save(best_controlnet.state_dict(), model_save_path)

# Inference
controlnet.eval()
x_train_patches_tensor = torch.from_numpy(x_train_patches[:, np.newaxis, :, :]).to(device)
with torch.no_grad():
    predicted_patches = best_controlnet(x_train_patches_tensor)
predicted_patches_np = predicted_patches.cpu().numpy() * y_train_max
predicted_patches_np[predicted_patches_np < 0] = 0.0

# Reconstruct the image
reconstructed_image = unpatchify(predicted_patches_np[:, 0, :, :], (576, 576))

# Save results as Xarray Dataset
lats = ds_dmsp.y.values[:576]
lons = ds_dmsp.x.values[:576]
ds_result = xr.Dataset({
    'dmsp': (['lat', 'lon'], ds_dmsp.values[0, :576, :576]),
    'viirs_controlnet': (['lat', 'lon'], reconstructed_image),
    'viirs': (['lat', 'lon'], ds_viirs.values[:576, :576]),
}, coords={'lat': lats, 'lon': lons})

print(ds_result)
