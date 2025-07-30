import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/content/drive/MyDrive/face_verification_project/data/Resized_plastic_dataset/PVC/pvc Type de dechet de plastique(300)"
SAVE_DIR = "/content/drive/MyDrive/face_verification_project/data"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "models"), exist_ok=True)

BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS = 3
Z_DIM = 128
LR = 1e-4
CRITIC_ITER = 5
LAMBDA_GP = 10
EPOCHS = 5000

# Seed for reproducibility
def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_all()

# Dataset
class SingleClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# Image transformations
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

dataset = SingleClassDataset(DATA_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)

# Critic
class Critic(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.net(x).view(-1)

# Gradient penalty for WGAN-GP
def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)
    mixed_scores = critic(interpolated)
    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

# Initialize models and optimizers
gen = Generator(Z_DIM, CHANNELS).to(DEVICE)
critic = Critic(CHANNELS).to(DEVICE)
opt_gen = torch.optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
opt_critic = torch.optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))
fixed_noise = torch.randn(64, Z_DIM, 1, 1, device=DEVICE)

# Training loop
print("Starting training...")

for epoch in range(EPOCHS):
    loop = tqdm(dataloader, leave=True)
    for real_images in loop:
        real_images = real_images.to(DEVICE)
        batch_size = real_images.size(0)

        # Train Critic
        for _ in range(CRITIC_ITER):
            noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
            fake_images = gen(noise).detach()
            critic_real = critic(real_images)
            critic_fake = critic(fake_images)
            gp = gradient_penalty(critic, real_images, fake_images, DEVICE)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

            opt_critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # Train Generator
        noise = torch.randn(batch_size, Z_DIM, 1, 1, device=DEVICE)
        fake_images = gen(noise)
        gen_fake = critic(fake_images)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        loop.set_postfix(loss_critic=loss_critic.item(), loss_gen=loss_gen.item())

    # Save models and images every 100 epochs
    if (epoch + 1) % 100 == 0:
        torch.save(gen.state_dict(), os.path.join(SAVE_DIR, "models", f"generator_epoch_{epoch+1}.pt"))
        torch.save(critic.state_dict(), os.path.join(SAVE_DIR, "models", f"critic_epoch_{epoch+1}.pt"))

        gen.eval()
        with torch.no_grad():
            samples = gen(fixed_noise).cpu()
            samples = (samples + 1) / 2
            save_image(samples, os.path.join(SAVE_DIR, "images", f"epoch_{epoch+1}.png"), nrow=8)
        gen.train()

print("Training complete!")
