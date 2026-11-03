import torch

from generator import Generator
from discriminator import Discriminator


# Configuration
latent_dim = 100
batch_size = 64

# Initialize both networks
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator()

# Generate random noise
noise = torch.randn(batch_size, latent_dim)

# Generator creates fake images
fake_images = generator(noise)

# Discriminator evaluates generated images
predictions = discriminator(fake_images)

print("Generated images:", fake_images.shape)
print("Discriminator predictions:", predictions.shape)

print("Sample logits:")
print(predictions[:5].detach().flatten())