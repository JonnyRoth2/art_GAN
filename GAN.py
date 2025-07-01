from torchvision.datasets import CocoCaptions
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import requests
from PIL import Image
from io import BytesIO
import torch.nn.utils.spectral_norm as spectral_norm
import random

# annFile = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
# local_ann_file = "captions_train2017.json"

class ArtCaptionDataset(Dataset):
    def __init__(self, image_dir, json_path, transform, text_encoder, max_samples=None):
        self.image_dir = image_dir
        self.transform = transform
        self.encoder = text_encoder

        with open(json_path, 'r') as f:
            data = json.load(f)
            # print(data)

        self.samples = []
        for file_name, captions in data.items():
            for caption in captions:
                self.samples.append((file_name, caption))
                #print(file_name, caption)

        if max_samples:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, caption = self.samples[idx]
        img_path = os.path.join(self.image_dir, file_name)

        try:
            img = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            return self.__getitem__((idx + 1) % len(self))  # fallback
        #print(img)
        #print(idx)
        img = self.transform(img)
        text_emb = self.encoder.encode(caption)
        return img, torch.tensor(text_emb, dtype=torch.float32)
    
class Generator(nn.Module):
    def __init__(self, text_dim, noise_dim=100, out_channels=3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU()
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 2048 * 4 * 4),
            nn.BatchNorm1d(2048 * 4 * 4),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(

            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

             nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, text_emb):

        text_feat = self.text_proj(text_emb)
        noise_feat = self.noise_proj(noise)
        x = text_feat + noise_feat
        x = self.fc(x).view(-1, 2048, 4, 4)
        return self.deconv(x)

class Discriminator(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.image_net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  
            nn.LeakyReLU(0.3), 
            nn.Dropout2d(0.25),    

            nn.Conv2d(32,64 , 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3),
            nn.Dropout2d(0.25),

            nn.Conv2d(128,256 , 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.3),
            nn.Dropout2d(0.3),

            # nn.Conv2d(256,512 , 4, 2, 1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(0.3),
            # nn.Dropout2d(0.25),
        )
        self.text_fc = nn.Linear(text_dim, 256 * 8 * 8)  
        self.final = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )

    def forward(self, img, text_emb):
        img_feat = self.image_net(img)
        img_feat = nn.functional.adaptive_avg_pool2d(img_feat, (8, 8))
        txt_feat = self.text_fc(text_emb).view(-1, 256, 8, 8)
        combined = torch.cat((img_feat, txt_feat), dim=1)
        out=self.final(combined)
        return out.view(-1,1)    
def train(generator, discriminator, dataloader, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    
    optim_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.9))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, (real_imgs, txt_emb) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs, txt_emb = real_imgs.to(device), txt_emb.to(device)
            real_imgs = real_imgs + torch.empty_like(real_imgs).uniform_(-0.2, 0.2)
      
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, 100, device=device)
            real_imgs_noisy = real_imgs + torch.randn_like(real_imgs) * 0.2
            

            with torch.no_grad():
                fake_imgs_detached = generator(noise, txt_emb).detach()
            fake_imgs_noisy = fake_imgs_detached + torch.randn_like(fake_imgs_detached) * 0.2

 
            real_validity = discriminator(real_imgs_noisy, txt_emb)
            fake_validity = discriminator(fake_imgs_noisy, txt_emb)
            d_loss_real = criterion(real_validity, real_labels)
            d_loss_fake = criterion(fake_validity, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            for _ in range(2):
                noise = torch.randn(batch_size, 100, device=device)
                fake_imgs = generator(noise, txt_emb)
                fake_validity = discriminator(fake_imgs, txt_emb)
                g_loss = criterion(fake_validity, torch.ones(batch_size, 1, device=device))
                diversity_loss = torch.mean(torch.std(fake_imgs.view(fake_imgs.size(0), -1), dim=0))
                g_loss += 0.1 * (1.0 - diversity_loss)

                optim_G.zero_grad()
                g_loss.backward()
                optim_G.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        sample_images(G, ["a man riding a horse", "a dog in a park", "a red car", "a policeman under a tree", "a woman with a bowl of fruit", "a blue boat" ], text_encoder)
def sample_images(generator, text_prompts, text_encoder, save_path="samples/sample.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generator.eval()
    device = next(generator.parameters()).device
    noise = torch.randn(len(text_prompts), 100).to(device)

    txt_emb = [text_encoder.encode(p) for p in text_prompts]
    txt_emb = torch.tensor(np.stack(txt_emb), dtype=torch.float32).to(device)

    with torch.no_grad():
        imgs = generator(noise, txt_emb).cpu()

    inv_norm = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    imgs = inv_norm(imgs)

    imgs = torch.clamp(imgs, 0, 1)

    grid = make_grid(imgs, nrow=3)

    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
image_dir = "images"
caption_path = "ArtCap.json"

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

from transformers import CLIPProcessor, CLIPModel

class CLIPTextEncoder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def encode(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs.squeeze(0).cpu()

text_encoder = CLIPTextEncoder()

dataset = ArtCaptionDataset(image_dir, caption_path, transform, text_encoder, max_samples=10000)



dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

G = Generator(text_dim=512, noise_dim=100)
D = Discriminator(text_dim=512)

train(G, D, dataloader, epochs=250)

G.eval()
device = next(G.parameters()).device

# Define test prompts
prompts = ["a dog", "a cat", "a horse", "a car", "a house"]
noise = torch.randn(len(prompts), 100).to(device)

# Get different text embeddings
embeddings_diff = [text_encoder.encode(p) for p in prompts]
embeddings_diff = torch.tensor(np.stack(embeddings_diff), dtype=torch.float32).to(device)

# Get same text embedding repeated
same_embedding = text_encoder.encode("a dog")
embeddings_same = torch.tensor([same_embedding] * len(prompts), dtype=torch.float32).to(device)

# Generate images
with torch.no_grad():
    imgs_diff = G(noise, embeddings_diff).cpu()
    imgs_same = G(noise, embeddings_same).cpu()

# Convert to displayable images
def denorm(imgs):
    inv = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    return torch.clamp(inv(imgs), 0, 1)

imgs_diff = denorm(imgs_diff)
imgs_same = denorm(imgs_same)

# Plot
def show_images(imgs, title):
    grid = make_grid(imgs, nrow=5)
    plt.figure(figsize=(15, 4))
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.title(title)
    plt.pause(2)
    plt.close()

show_images(imgs_diff, "Different Text, Same Noise")
show_images(imgs_same, "Same Text, Same Noise")

sample_images(G, ["a man riding a horse", "a dog in a park", "a red car", "a policeman under a tree", "a woman with a bowl of fruit", "a blue boat" ], text_encoder, save_path="samples/final.png")