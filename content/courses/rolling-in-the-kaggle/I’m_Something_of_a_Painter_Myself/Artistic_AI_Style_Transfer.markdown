---
date: '2024-01-02'
title: "🎨 Artistic AI: Style Transfer 🌌"
author: JDW
type: book
weight: 10
output:
  rmarkdown::html_document()
editor_options:
  markdown:
    wrap: 255
---






<center>

 **Original Notebook** : <https://www.kaggle.com/code/anthonytherrien/artistic-ai-style-transfer>

</center>



```python
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import torchvision
import shutil
import torch
import os
```



```python
class MonetDataset(Dataset):
  def __init__(self, folder_path, transform = None):
    self.folder_path = folder_path
    self.transform   = transform
    self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]


  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image_path = self.images[idx]
    image      = Image.open(image_path).convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image

class PhotoDataset(Dataset):
  def __init__(self, folder_path, transform = None):
    self.folder_path = folder_path
    self.transform   = transform
    self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    image_path = self.images[idx]
    image      = Image.open(image_path).convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image
```



```python
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
IMAGE_DIM     = 128
num_epochs    = 30
batch_size    = 16
learning_rate = 1e-4
weight_decay  = 1e-3

# Define transformations
transform = transforms.Compose([
  transforms.Resize((256, 256)),

  # Data augmentation
  transforms.RandomHorizontalFlip(p = 0.5), # Randomly flip the image Horizontallly
  transforms.RandomRotation(degrees = 15),  # Random rotation of the image
  transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1), # Randomly change brightness, contrast, and saturation

  transforms.ToTensor(),
  transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

# Initialzie datasets
monet_dataset = MonetDataset("./data/monet_jpg", transform)
photo_dataset = PhotoDataset("./data/monet_jpg", transform)

# Initialize data loaders
monet_loader = DataLoader(monet_dataset, batch_size = batch_size, shuffle = True)
photo_loader = DataLoader(photo_dataset, batch_size = batch_size, shuffle = True)
```




```python
class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
      nn.BatchNorm2d(channels),
      nn.ReLU(inplace = True),
      nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
      nn.BatchNorm2d(channels)
    )

  def forward(self, x):
    return x + self.block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Correct the input size of the linear layer to match the flattened output size
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(262144, 1024),  # Updated to match the output size of conv_layers
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        conv_out = self.conv_layers(img)
        validity = self.linear_layers(conv_out)
        return validity


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Downsample
        self.down = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        # Upsample
        self.up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        x = self.up(x)
        return x
```



```python
# Instantiate models
D = Discriminator().to(device)
G = Generator().to(device)

# Optimizers(No needs for BCELoss)
d_optimizer = optim.AdamW(D.parameters(), lr = learning_rate, weight_decay = weight_decay)
g_optimizer = optim.AdamW(G.parameters(), lr = learning_rate, weight_decay = weight_decay)

criterion_GAN           = nn.MSELoss()
criterion_cycle         = nn.L1Loss()
criterion_identity      = nn.L1Loss()
criterion_discriminator = nn.BCELoss()
```



```python
# Lamda factors for different components of the loss
lambda_cycle    = 10.0
lambda_identity = 0.5 * lambda_cycle


def g_loss_function(real_monet, generated_photo, reconstructed_monet, real_photo, D):
    valid = torch.ones(real_monet.size(0), 1, device=device)
    g_loss_GAN = criterion_GAN(D(generated_photo), valid)
    g_loss_cycle = criterion_cycle(reconstructed_monet, real_monet)
    identity_photo = G(real_photo)
    g_loss_identity = criterion_identity(identity_photo, real_photo)
    g_loss = g_loss_GAN + lambda_cycle * g_loss_cycle + lambda_identity * g_loss_identity
    return g_loss

def d_loss_function(real_outputs, fake_outputs):
    real_labels = torch.ones(real_outputs.size(0), 1, device = device)
    real_loss   = criterion_discriminator(real_outputs, real_labels)
    fake_labels = torch.zeros(fake_outputs.size(0), 1, device = device)
    fake_loss   = criterion_discriminator(fake_outputs, fake_labels)
    d_loss = real_loss + fake_loss

    return d_loss
```



```python
for epoch in range(num_epochs):
  loader = zip(photo_loader, monet_loader)
  epoch_g_loss = 0.0
  epoch_d_loss = 0.0
  best_g_loss = 1.0

  for real_photos, real_monets in loader:
    real_photos = real_photos.to(device)
    real_monets = real_monets.to(device)

    # Generator forward pass
    G.zero_grad()
    monet_style_imgs = G(real_photos)
    reconstructed_photos = G(monet_style_imgs)

    # For identify loss (optional)
    # real_monets = next(iter(monet_loader)).to(device)

    # Calculate Generator Loss
    g_loss = g_loss_function(real_photos, monet_style_imgs, reconstructed_photos, real_monets, D)
    g_loss.backward()
    g_optimizer.step()

    # Discriminator Training
    # Randomly sample real Monet images from monet_loader for discriminator training
    D.zero_grad()

    # Discriminator outputs, for real and fake images
    real_outputs = D(real_monets)
    fake_outputs = D(monet_style_imgs.detach())

    # Compute Discriminator loss
    d_loss = d_loss_function(real_outputs, fake_outputs)
    d_loss.backward()
    d_optimizer.step()

    # Accumulate losses for epoch-level logging
    epoch_g_loss += g_loss.item()
    epoch_d_loss += d_loss.item()

  # Print epoch-level summrise
  print(
    f"Epoch {epoch+1}/{num_epochs} - Generator Loss : {epoch_g_loss / len(photo_loader):.4f}, Discriminator Loss : {epoch_d_loss / len(photo_loader):.4f}"
  )
#> Epoch 1/30 - Generator Loss : 4.7019, Discriminator Loss : 1.5295
#> Epoch 2/30 - Generator Loss : 3.1539, Discriminator Loss : 1.4993
#> Epoch 3/30 - Generator Loss : 2.9530, Discriminator Loss : 1.6271
#> Epoch 4/30 - Generator Loss : 2.8812, Discriminator Loss : 1.4234
#> Epoch 5/30 - Generator Loss : 2.7983, Discriminator Loss : 1.6110
#> Epoch 6/30 - Generator Loss : 2.7322, Discriminator Loss : 1.3266
#> Epoch 7/30 - Generator Loss : 2.6210, Discriminator Loss : 1.6879
#> Epoch 8/30 - Generator Loss : 2.8028, Discriminator Loss : 1.3344
#> Epoch 9/30 - Generator Loss : 2.8388, Discriminator Loss : 1.4804
#> Epoch 10/30 - Generator Loss : 2.6649, Discriminator Loss : 2.5616
#> Epoch 11/30 - Generator Loss : 2.7110, Discriminator Loss : 1.9228
#> Epoch 12/30 - Generator Loss : 2.6055, Discriminator Loss : 2.4581
#> Epoch 13/30 - Generator Loss : 2.5392, Discriminator Loss : 1.7270
#> Epoch 14/30 - Generator Loss : 2.3506, Discriminator Loss : 2.0267
#> Epoch 15/30 - Generator Loss : 2.4401, Discriminator Loss : 1.2807
#> Epoch 16/30 - Generator Loss : 2.3893, Discriminator Loss : 1.4658
#> Epoch 17/30 - Generator Loss : 2.3390, Discriminator Loss : 1.3588
#> Epoch 18/30 - Generator Loss : 2.3024, Discriminator Loss : 1.3720
#> Epoch 19/30 - Generator Loss : 2.3383, Discriminator Loss : 1.2395
#> Epoch 20/30 - Generator Loss : 2.1588, Discriminator Loss : 1.2999
#> Epoch 21/30 - Generator Loss : 2.1494, Discriminator Loss : 1.2109
#> Epoch 22/30 - Generator Loss : 2.1830, Discriminator Loss : 1.2221
#> Epoch 23/30 - Generator Loss : 2.2301, Discriminator Loss : 1.4188
#> Epoch 24/30 - Generator Loss : 2.3028, Discriminator Loss : 1.2354
#> Epoch 25/30 - Generator Loss : 2.1218, Discriminator Loss : 1.8879
#> Epoch 26/30 - Generator Loss : 2.0985, Discriminator Loss : 1.3165
#> Epoch 27/30 - Generator Loss : 2.0021, Discriminator Loss : 1.3461
#> Epoch 28/30 - Generator Loss : 1.9216, Discriminator Loss : 1.4205
#> Epoch 29/30 - Generator Loss : 1.9726, Discriminator Loss : 1.3580
#> Epoch 30/30 - Generator Loss : 1.9603, Discriminator Loss : 1.3185
```


```python
# Configuration for style transfer
transformed_save_dir = './images'

# Ensure the save directory exists
if not os.path.exists(transformed_save_dir) :
  os.makedirs(transformed_save_dir)

with HiddenPrints():
  G.eval()

# Process Monet images in batchs and save transformed images
for i, real_photos in enumerate(photo_loader):
  real_photos = real_photos.to(device)
  with torch.no_grad():
    monet_sthle_imgs = G(real_photos)
  # save each transformed image
  for j, img in enumerate(monet_style_imgs):
    save_path = os.path.join(transformed_save_dir, f'monet_style_image_{i * batch_size + j}.png')
    torchvision.utils.save_image(img, save_path)

print(f"Transformed images are saved in {transformed_save_dir}")
#> Transformed images are saved in ./images

shutil.make_archive("./images", "zip", "./images")
```






























