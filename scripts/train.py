import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# -----------------------------
# U-Net Model (same as training)
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
# Load and Denoise Image
# -----------------------------
def run_denoising(model_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process output
    output_image = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output_image = (output_image * 255).astype('uint8')
    result = Image.fromarray(output_image)

    # Save and show
    result.save('denoised_result.png')
    print("✅ Denoised image saved as 'denoised_result.png'")

    # Display
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image)
    axs[0].set_title('Noisy Input')
    axs[0].axis('off')

    axs[1].imshow(result)
    axs[1].set_title('Denoised Output')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    model_path = 'unet_denoiser.pth'
    image_path = 'LR_image.png'  # Low resolution image path

    run_denoising(model_path, image_path)
