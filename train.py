import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import datetime
import shutil
from model.MLPGAN import Discriminator, Generator

# conf
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = float(os.getenv("LR", 3e-4))
z_dim = int(os.getenv("Z_DIM", 64))
image_dim = 28 * 28 * 1
batch_size = int(os.getenv("BATCH_SIZE", 100))
num_epochs = int(os.getenv("EPOCHS", 2))

# Directorio de salida
output_dir = os.getenv("OUTPUT_DIR", "./output")
os.makedirs(output_dir, exist_ok=True)

# inicializacion de los modelos
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

# entrenamiento
print(f"Training GAN for {num_epochs} epochs on {device}")

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    print(f"[{epoch+1}/{num_epochs}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")

# guardado
torch.save(gen.state_dict(), os.path.join(output_dir, "generator.pth"))
torch.save(disc.state_dict(), os.path.join(output_dir, "discriminator.pth"))
print(f"✅ Modelos guardados en {output_dir}")
