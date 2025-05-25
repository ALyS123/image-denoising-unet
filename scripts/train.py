import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
from pytorch_msssim import ssim

# -----------------------------
# U-Net Model for Denoising
# -----------------------------
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        self.dec3 = conv_block(256 + 128, 128)
        self.dec2 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d3 = F.interpolate(e3, scale_factor=2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, scale_factor=2)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        out = torch.sigmoid(self.final(d2))
        return out

# -----------------------------
# Dataset with Curriculum Noise
# -----------------------------
class DenoisingDataset(Dataset):
    def __init__(self, hr_dir, transform, epoch=0, total_epochs=50):
        self.hr_images = sorted(glob(os.path.join(hr_dir, '*.png')))
        self.transform = transform
        self.epoch = epoch
        self.total_epochs = total_epochs

    def __len__(self):
        return len(self.hr_images)

    def add_gaussian_noise(self, image, std):
        noise = torch.randn_like(image) * std
        noisy = image + noise
        return torch.clamp(noisy, 0., 1.)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_images[idx]).convert('RGB')
        clean_img = self.transform(hr_img)

        # Progressive noise: high → low
        max_std = 0.3
        min_std = 0.05
        std = max_std - ((max_std - min_std) * (self.epoch / self.total_epochs))
        noisy_img = self.add_gaussian_noise(clean_img, std)

        return noisy_img, clean_img

# -----------------------------
# Hybrid Loss: MSE + SSIM
# -----------------------------
def hybrid_loss(output, target):
    mse_loss = F.mse_loss(output, target)
    ssim_loss = 1 - ssim(output, target, data_range=1.0, size_average=True)
    return 0.5 * mse_loss + 0.5 * ssim_loss

# -----------------------------
# Training Loop
# -----------------------------
def train_unet_denoiser():
    hr_dir = 'data/DIV2K/DIV2K_train_HR/DIV2K_train_HR' # HR images for training 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        dataset = DenoisingDataset(hr_dir, transform, epoch=epoch, total_epochs=num_epochs)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"\n✅ Epoch {epoch+1}/{num_epochs} — Training on {len(dataset)} images")

        model.train()
        running_loss = 0.0

        for i, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)

            outputs = model(noisy)
            loss = hybrid_loss(outputs, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"  Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"✅ Epoch [{epoch+1}/{num_epochs}] complete — Avg Loss: {running_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), 'models/unet_denoiser.pth')
    print("\n🎉 Training complete. Model saved as 'unet_denoiser.pth'")

# -----------------------------
# Run Training
# -----------------------------
if __name__ == "__main__":
    train_unet_denoiser()
