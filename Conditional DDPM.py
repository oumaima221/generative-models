#itwas trained  on our plastic  data set 
import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# ---- CONFIG ----
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 7
EMBED_DIM = 64
EPOCHS = 1200
T_steps = 1000
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- SEED ----
def seed_all(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_all()

# ---- TIME EMBEDDING ----
def get_timestep_embedding(timesteps, dim):
    device = timesteps.device
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

# ---- MODEL ----
class SimpleUNet(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())

        self.conv1 = nn.Sequential(
            nn.Conv2d(3 + embed_dim, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU()
        )
        self.conv4 = nn.Conv2d(128, 3, 3, padding=1)

    def forward(self, x, t, y):
        t_emb = get_timestep_embedding(t, self.embed_dim).to(x.device)
        t_emb = self.time_mlp(t_emb)
        y_emb = self.label_emb(y)
        emb = t_emb + y_emb
        emb = emb[:, :, None, None].expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, emb], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.conv4(x)

# ---- SCHEDULE ----
betas = torch.linspace(1e-4, 0.02, T_steps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).to(DEVICE)

# ---- FORWARD DIFFUSION ----
def forward_diffusion_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None].to(x0.device)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None].to(x0.device)
    return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise, noise

# ---- LOSS ----
def diffusion_loss(model, x_start, t, y):
    noise = torch.randn_like(x_start)
    x_noisy, noise = forward_diffusion_sample(x_start, t, noise)
    predicted_noise = model(x_noisy, t, y)
    return 0.8 * F.mse_loss(predicted_noise, noise) + 0.2 * F.l1_loss(predicted_noise, noise)

# ---- DATA ----
transform = T.Compose([
    T.Resize((IMAGE_SIZE + 4, IMAGE_SIZE + 4)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DATA_PATH = '/content/drive/MyDrive/wasteGAN-main/Resized_plastic_dataset'
dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
reduced_dataset = Subset(dataset, random.sample(range(len(dataset)), int(1 * len(dataset))))
dataloader = DataLoader(reduced_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ---- INIT ----
model = SimpleUNet(num_classes=NUM_CLASSES, embed_dim=EMBED_DIM).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

# ---- TRAIN ----
def train():
    best_loss = float("inf")
    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        epoch_loss = 0
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            t = torch.randint(0, T_steps, (images.size(0),), device=DEVICE).long()
            loss = diffusion_loss(model, images, t, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Saved best model")

# ---- SAMPLE ----
@torch.no_grad()
def sample(class_label, n_samples=8):
    model.eval()
    x = torch.randn(n_samples, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    y = torch.full((n_samples,), class_label, dtype=torch.long, device=DEVICE)
    for t in reversed(range(T_steps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=DEVICE)
        pred_noise = model(x, t_batch, y)
        beta_t = betas[t].to(DEVICE)
        alpha_t = alphas[t].to(DEVICE)
        alpha_cumprod_t = alphas_cumprod[t].to(DEVICE)
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
        x = coef1 * (x - coef2 * pred_noise)
        if t > 0:
            noise = torch.randn_like(x)
            x += torch.sqrt(beta_t) * noise
    x = torch.clamp(x, -1, 1)
    x = (x + 1) / 2
    return x.cpu()

# ---- DISPLAY ----
def show_images(images, title="Generated Images"):
    grid = vutils.make_grid(images, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title(title)
    plt.show()

def save_images(images, class_label):
    # Update this path to your desired directory inside Drive
    drive_path = "/content/drive/MyDrive/generated_images"
    os.makedirs(drive_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(drive_path, f"class_{class_label}_{timestamp}.png")

    vutils.save_image(images, filepath, nrow=4)
    print(f"✅ Saved: {filepath}")


# ---- MAIN ----
if __name__ == "__main__":
    print("🚀 Starting training...")
    train()

    print("🎨 Sampling images...")
    for cls in range(NUM_CLASSES):
        samples = sample(class_label=cls, n_samples=8)
        show_images(samples, title=f"Class {cls} Samples")
        save_images(samples, cls)
