
import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator


# ============================================================
# 1. Configuration and Hyperparameters
# ============================================================

LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0002

BETA1 = 0.5
BETA2 = 0.999

# Select GPU if available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ============================================================
# 2. Create Output Directories
# ============================================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


# ============================================================
# 3. Load and Preprocess Fashion-MNIST Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Number of batches: {len(train_loader)}")


# ============================================================
# 4. Initialize Generator and Discriminator
# ============================================================

generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

print("Generator and Discriminator initialized.")


# ============================================================
# 5. Define Loss Function and Optimizers
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer_G = torch.optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)

optimizer_D = torch.optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2)
)


# ============================================================
# 6. Create Fixed Noise for Monitoring Generated Images
# ============================================================

fixed_noise = torch.randn(64, LATENT_DIM, device=device)


# ============================================================
# 7. Main GAN Training Loop
# ============================================================

def train():

    for epoch in range(EPOCHS):

        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, (real_images, _) in enumerate(train_loader):

            # Move real images to the selected device
            real_images = real_images.to(device)

            current_batch_size = real_images.size(0)

            # Create labels for real and fake images
            real_labels = torch.ones(
                current_batch_size, 1, device=device
            )

            fake_labels = torch.zeros(
                current_batch_size, 1, device=device
            )


            # ------------------------------------------------
            # 7A. Train the Discriminator
            # ------------------------------------------------

            optimizer_D.zero_grad(set_to_none=True)

            # Discriminator learns to classify real images
            real_output = discriminator(real_images)

            real_loss = criterion(real_output, real_labels)

            # Generate fake images for Discriminator training
            noise = torch.randn(
                current_batch_size, LATENT_DIM, device=device
            )

            fake_images = generator(noise)

            # Detach prevents gradients from updating Generator
            fake_output = discriminator(fake_images.detach())

            fake_loss = criterion(fake_output, fake_labels)

            # Combine real and fake losses
            d_loss = real_loss + fake_loss

            # Calculate Discriminator gradients and update weights
            d_loss.backward()
            optimizer_D.step()


            # ------------------------------------------------
            # 7B. Train the Generator
            # ------------------------------------------------

            # Freeze Discriminator weights during Generator training
            for param in discriminator.parameters():
                param.requires_grad = False

            optimizer_G.zero_grad(set_to_none=True)

            # Generate a new batch of fake images
            noise = torch.randn(
                current_batch_size, LATENT_DIM, device=device
            )

            fake_images = generator(noise)

            # Generator wants Discriminator to classify fake images as real
            fake_output = discriminator(fake_images)

            g_loss = criterion(fake_output, real_labels)

            # Backpropagate through Discriminator into Generator
            g_loss.backward()
            optimizer_G.step()

            # Enable Discriminator weights for the next training iteration
            for param in discriminator.parameters():
                param.requires_grad = True


            # ------------------------------------------------
            # 7C. Track Training Losses
            # ------------------------------------------------

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"D Loss: {d_loss.item():.4f} "
                    f"G Loss: {g_loss.item():.4f}"
                )


        # ----------------------------------------------------
        # 7D. Calculate Average Loss for the Epoch
        # ----------------------------------------------------

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}] completed")
        print(f"Average D Loss: {avg_d_loss:.4f}")
        print(f"Average G Loss: {avg_g_loss:.4f}")


        # ----------------------------------------------------
        # 7E. Save Generated Images After Each Epoch
        # ----------------------------------------------------

        generator.eval()

        with torch.no_grad():
            generated_images = generator(fixed_noise)

            save_image(
                generated_images,
                f"outputs/epoch_{epoch + 1}.png",
                normalize=True,
                value_range=(-1, 1),
                nrow=8
            )

        generator.train()


        # ----------------------------------------------------
        # 7F. Save Model Checkpoints Every 10 Epochs
        # ----------------------------------------------------

        if (epoch + 1) % 10 == 0:

            checkpoint = {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
            }

            torch.save(
                checkpoint,
                f"checkpoints/dcgan_epoch_{epoch + 1}.pth"
            )

            print("Checkpoint saved.")


    # ========================================================
    # 8. Save Final Models
    # ========================================================

    torch.save(
        generator.state_dict(),
        "checkpoints/generator_final.pth"
    )

    torch.save(
        discriminator.state_dict(),
        "checkpoints/discriminator_final.pth"
    )

    print("\nTraining completed!")
    print("Final models saved in checkpoints/")


# ============================================================
# 9. Run Training
# ============================================================

if __name__ == "__main__":
    train()
