# custom layers 
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
EPOCHS = 300
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
        self.gain = gain

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = EqualizedDense(self.x_channels, gain=1)
        self.dense_2 = EqualizedDense(self.x_channels, gain=1)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
#A model mapping to map the random noise into style code

def Mapping(num_stages, input_shape=512):
    z = layers.Input(shape=(input_shape))
    w = pixel_norm(z)
    for i in range(8):
        w = EqualizedDense(512, learning_rate_multiplier=0.01)(w)
        w = layers.LeakyReLU(0.2)(w)
    w = tf.tile(tf.expand_dims(w, 1), (1, num_stages, 1))
    return keras.Model(z, w, name="mapping")
#Generator 
Generator

class Generator:
    def __init__(self, start_res_log2, target_res_log2, filter_nums=None):
        self.start_res_log2 = start_res_log2
        self.target_res_log2 = target_res_log2
        self.num_stages = target_res_log2 - start_res_log2 + 1
        self.g_blocks = []
        self.to_rgb = []
        self.noise_inputs = []
        self.filter_nums = filter_nums

        start_res = 2 ** start_res_log2
        self.input_shape = (start_res, start_res, self.filter_nums[start_res_log2])
        self.g_input = layers.Input(self.input_shape, name="generator_input")

        for i in range(start_res_log2, target_res_log2 + 1):
            filter_num = self.filter_nums[i]
            res = 2 ** i
            self.noise_inputs.append(
                layers.Input(shape=(res, res, 1), name=f"noise_{res}x{res}")
            )
            to_rgb = Sequential(
                [
                    layers.InputLayer(input_shape=(res, res, filter_num)),
                    EqualizedConv(3, 1, gain=1),
                ],
                name=f"to_rgb_{res}x{res}",
            )
            self.to_rgb.append(to_rgb)
            is_base = i == self.start_res_log2
            if is_base:
                input_shape = (res, res, self.filter_nums[i - 1])
            else:
                input_shape = (2 ** (i - 1), 2 ** (i - 1), self.filter_nums[i - 1])
            g_block = self.build_block(
                filter_num, res=res, input_shape=input_shape, is_base=is_base
            )
            self.g_blocks.append(g_block)

    def build_block(self, filter_num, res, input_shape, is_base):
        input_tensor = layers.Input(shape=input_shape, name=f"g_{res}")
        noise = layers.Input(shape=(res, res, 1), name=f"noise_{res}")
        w = layers.Input(shape=512)
        x = input_tensor

        if not is_base:
            x = layers.UpSampling2D((2, 2))(x)
            x = EqualizedConv(filter_num, 3)(x)

        x = AddNoise()([x, noise])
        x = layers.LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        x = AdaIN()([x, w])

        x = EqualizedConv(filter_num, 3)(x)
        x = AddNoise()([x, noise])
        x = layers.LeakyReLU(0.2)(x)
        x = InstanceNormalization()(x)
        x = AdaIN()([x, w])
        return keras.Model([input_tensor, w, noise], x, name=f"genblock_{res}x{res}")

    def grow(self, res_log2):
        res = 2 ** res_log2

        num_stages = res_log2 - self.start_res_log2 + 1
        w = layers.Input(shape=(self.num_stages, 512), name="w")

        alpha = layers.Input(shape=(1), name="g_alpha")
        x = self.g_blocks[0]([self.g_input, w[:, 0], self.noise_inputs[0]])

        if num_stages == 1:
            rgb = self.to_rgb[0](x)
        else:
            for i in range(1, num_stages - 1):

                x = self.g_blocks[i]([x, w[:, i], self.noise_inputs[i]])

            old_rgb = self.to_rgb[num_stages - 2](x)
            old_rgb = layers.UpSampling2D((2, 2))(old_rgb)

            i = num_stages - 1
            x = self.g_blocks[i]([x, w[:, i], self.noise_inputs[i]])

            new_rgb = self.to_rgb[i](x)

            rgb = fade_in(alpha[0], new_rgb, old_rgb)

        return keras.Model(
            [self.g_input, w, self.noise_inputs, alpha],
            rgb, 
          name=f"generator_{res}_x_{res}",
        )
#Discriminator

class Discriminator:
    def __init__(self, start_res_log2, target_res_log2, filter_nums):
        self.start_res_log2 = start_res_log2
        self.target_res_log2 = target_res_log2
        self.num_stages = target_res_log2 - start_res_log2 + 1
        self.filter_nums = filter_nums
        # list of discriminator blocks at increasing resolution
        self.d_blocks = []
        # list of layers to convert RGB into activation for d_blocks inputs
        self.from_rgb = []

        for res_log2 in range(self.start_res_log2, self.target_res_log2 + 1):
            res = 2 ** res_log2
            filter_num = self.filter_nums[res_log2]
            from_rgb = Sequential(
                [
                    layers.InputLayer(
                        input_shape=(res, res, 3), name=f"from_rgb_input_{res}"
                    ),
                    EqualizedConv(filter_num, 1),
                    layers.LeakyReLU(0.2),
                ],
                name=f"from_rgb_{res}",
            )

            self.from_rgb.append(from_rgb)

            input_shape = (res, res, filter_num)
            if len(self.d_blocks) == 0:
                d_block = self.build_base(filter_num, res)
            else:
                d_block = self.build_block(
                    filter_num, self.filter_nums[res_log2 - 1], res
                )

            self.d_blocks.append(d_block)

    def build_base(self, filter_num, res):
        input_tensor = layers.Input(shape=(res, res, filter_num), name=f"d_{res}")
        x = minibatch_std(input_tensor)
        x = EqualizedConv(filter_num, 3)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Flatten()(x)
        x = EqualizedDense(filter_num)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = EqualizedDense(1)(x)
        return keras.Model(input_tensor, x, name=f"d_{res}")

    def build_block(self, filter_num_1, filter_num_2, res):
        input_tensor = layers.Input(shape=(res, res, filter_num_1), name=f"d_{res}")
        x = EqualizedConv(filter_num_1, 3)(input_tensor)
        x = layers.LeakyReLU(0.2)(x)
        x = EqualizedConv(filter_num_2)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D((2, 2))(x)
        return keras.Model(input_tensor, x, name=f"d_{res}")

    def grow(self, res_log2):
        res = 2 ** res_log2
        idx = res_log2 - self.start_res_log2
        alpha = layers.Input(shape=(1), name="d_alpha")
        input_image = layers.Input(shape=(res, res, 3), name="input_image")
        x = self.from_rgb[idx](input_image)
        x = self.d_blocks[idx](x)
        if idx > 0:
            idx -= 1
            downsized_image = layers.AveragePooling2D((2, 2))(input_image)
            y = self.from_rgb[idx](downsized_image)
            x = fade_in(alpha[0], x, y)

            for i in range(idx, -1, -1):
                x = self.d_blocks[i](x)
        return keras.Model([input_image, alpha], x, name=f"discriminator_{res}_x_{res}")
  #Style GAN
  class StyleGAN(tf.keras.Model):
    def __init__(self, z_dim=512, target_res=64, start_res=4, filter_nums=None):
        super().__init__()
        self.z_dim = z_dim

        self.target_res_log2 = log2(target_res)
        self.start_res_log2 = log2(start_res)
        self.current_res_log2 = self.target_res_log2
        self.num_stages = self.target_res_log2 - self.start_res_log2 + 1

        self.alpha = tf.Variable(1.0, dtype=tf.float32, trainable=False, name="alpha")

        if filter_nums is None:
            self.filter_nums = {
                0: 32,
                1: 32,
                2: 32,  # 4x4
                3: 32,  # 8x8
                4: 32,  # 16x16
                5: 32,  # 32x32
            }
        else:
            self.filter_nums = filter_nums
        
        self.mapping = Mapping(num_stages=self.num_stages)
        self.d_builder = Discriminator(self.start_res_log2, self.target_res_log2, self.filter_nums)
        self.g_builder = Generator(self.start_res_log2, self.target_res_log2, self.filter_nums)
        self.g_input_shape = self.g_builder.input_shape

        self.phase = None
        self.train_step_counter = tf.Variable(0, dtype=tf.int32, trainable=False)

        self.loss_weights = {"gradient_penalty": 10, "drift": 0.001}

    def grow_model(self, res):
        tf.keras.backend.clear_session()
        res_log2 = log2(res)
        self.generator = self.g_builder.grow(res_log2)
        self.discriminator = self.d_builder.grow(res_log2)
        self.current_res_log2 = res_log2
        print(f"\nModel resolution:{res}x{res}")

    def compile(
        self, steps_per_epoch, phase, res, d_optimizer, g_optimizer, *args, **kwargs
    ):
        self.loss_weights = kwargs.pop("loss_weights", self.loss_weights)
        self.steps_per_epoch = steps_per_epoch
        if res != 2 ** self.current_res_log2:
            self.grow_model(res)
            self.d_optimizer = d_optimizer
            self.g_optimizer = g_optimizer

        self.train_step_counter.assign(0)
        self.phase = phase
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        super(StyleGAN, self).compile(*args, **kwargs)

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def generate_noise(self, batch_size):
        noise = [
            tf.random.normal((batch_size, 2 ** res, 2 ** res, 1))
            for res in range(self.start_res_log2, self.target_res_log2 + 1)
        ]
        return noise

    def gradient_loss(self, grad):
        loss = tf.square(grad)
        loss = tf.reduce_sum(loss, axis=tf.range(1, tf.size(tf.shape(loss))))
        loss = tf.sqrt(loss)
        loss = tf.reduce_mean(tf.square(loss - 1))
        return loss

    def train_step(self, real_images):
        self.train_step_counter.assign_add(1)

        if self.phase == "TRANSITION":
            self.alpha.assign(
                tf.cast(self.train_step_counter / self.steps_per_epoch, tf.float32)
            )
        elif self.phase == "STABLE":
            self.alpha.assign(1.0)
        else:
            raise NotImplementedError
        alpha = tf.expand_dims(self.alpha, 0)
        batch_size = tf.shape(real_images)[0]
        real_labels = tf.ones(batch_size)
        fake_labels = -tf.ones(batch_size)

        z = tf.random.normal((batch_size, self.z_dim))
        const_input = tf.ones(tuple([batch_size] + list(self.g_input_shape)))
        noise = self.generate_noise(batch_size)

        # generator
        with tf.GradientTape() as g_tape:
            w = self.mapping(z)
            fake_images = self.generator([const_input, w, noise, alpha])
            pred_fake = self.discriminator([fake_images, alpha])
            g_loss = wasserstein_loss(real_labels, pred_fake)

            trainable_weights = (
                self.mapping.trainable_weights + self.generator.trainable_weights
            )
#Training LOOP
START_RES = 4
TARGET_RES = 32

def train(label, start_res, target_res, steps_per_epoch):
    style_gan = StyleGAN(start_res=START_RES, target_res=TARGET_RES)
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    val_batch_size = 16
    val_z = tf.random.normal((val_batch_size, style_gan.z_dim))
    val_noise = style_gan.generate_noise(val_batch_size)

    start_res_log2 = int(np.log2(start_res))
    target_res_log2 = int(np.log2(target_res))

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == start_res and phase == "TRANSITION":
                continue

            train_dl = create_dataloader(res, label)

            steps = int(train_step_ratio[res_log2] * steps_per_epoch)

            style_gan.compile(
                d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )


            ckpt_cb = keras.callbacks.ModelCheckpoint(
                f"checkpoints_{label}/stylegan_{res}x{res}.ckpt",
                save_weights_only=True,
                verbose=0,
            )
            print(phase)
            style_gan.fit(
                train_dl, epochs=1, steps_per_epoch=steps, callbacks=[ckpt_cb]
            )
            
for i in range(10):
    print(f'\n\nTraining for Digit {i}')
    train(start_res=START_RES, target_res=TARGET_RES, steps_per_epoch=10, label=i)  
