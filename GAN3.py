import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
class CIFAR10LabelDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        label_onehot = F.one_hot(torch.tensor(label), num_classes=10).float()
        return img, label_onehot

class Generator(nn.Module):
    def __init__(self, text_dim, noise_dim=100, out_channels=3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LeakyReLU(0.2),
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
               nn.Linear(512, 2048 * 4 * 4),
                nn.BatchNorm1d(2048 * 4 * 4),
                nn.LeakyReLU(0.2),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(128, 64, 4, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, label_emb):
        text_feat = self.text_proj(label_emb)
        noise_feat = self.noise_proj(noise)
        x = torch.cat((text_feat, noise_feat), dim=1)
        x = self.fc(x).view(-1, 2048, 4, 4)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.image_net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2)
        )

        self.label_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1024 * 2 * 2),
            nn.LeakyReLU(0.2),
        )

        self.final = nn.Sequential(
            nn.Conv2d(2048, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, img, label_emb):
        img_feat = self.image_net(img)
        label_feat = self.label_proj(label_emb).view(-1, 1024, 2, 2)
        x = torch.cat((img_feat, label_feat), dim=1)
        return self.final(x).view(-1, 1)
def sample_images(generator, labels, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generator.eval()
    device = next(generator.parameters()).device
    noise = torch.randn(len(labels), 100).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    with torch.no_grad():
        imgs = generator(noise, labels).cpu()

    imgs = (imgs + 1) / 2
  
    num_imgs = len(labels)
    nrow = 3
    ncol = int(np.ceil(num_imgs / nrow))
    fig, axs = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3))
    axs = axs.flatten()

    for i in range(num_imgs):
        img = imgs[i].permute(1, 2, 0).numpy()
        axs[i].imshow(img)
        class_idx = torch.argmax(labels[i]).item()
        axs[i].set_title(CIFAR10_CLASSES[class_idx], fontsize=10)
        axs[i].axis('off')

    for j in range(num_imgs, len(axs)):
        axs[j].axis('off') 

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train(generator, discriminator, dataloader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.00003, betas=(0.5, 0.9))

    for epoch in range(epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)

            # Generate fake images
            noise = torch.randn(batch_size, 100, device=device)
            with torch.no_grad():
                fake_imgs = generator(noise, labels)

            # Optional noise injection
            real_imgs_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs)
            fake_imgs_noisy = fake_imgs + 0.05 * torch.randn_like(fake_imgs)

            # Discriminator forward
            real_logits = discriminator(real_imgs_noisy, labels)
            fake_logits = discriminator(fake_imgs_noisy, labels)

            # Hinge loss for Discriminator
            d_loss = torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # Generator update (3x)
            for _ in range(3):
                noise = torch.randn(batch_size, 100, device=device)
                fake_imgs = generator(noise, labels)
                fake_logits = discriminator(fake_imgs, labels)

                # Hinge loss for Generator
                g_loss = -torch.mean(fake_logits)

                optim_G.zero_grad()
                g_loss.backward()
                optim_G.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} Step {i} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), f"checkpoints/generator_epoch_{epoch:03d}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch_{epoch:03d}.pth")
            sample_images(generator, torch.eye(10, device=device), f"samples/sample_epoch_{epoch:03d}.png")
# ---------------- Run ----------------
dataset = CIFAR10LabelDataset(train=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
G = Generator(text_dim=10)
D = Discriminator(text_dim=10)
train(G, D, dataloader, epochs=201)
