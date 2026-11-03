
import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator for Fashion-MNIST.

    Input:
        Random noise vector of shape (batch_size, latent_dim)

    Output:
        Generated images of shape (batch_size, 1, 28, 28)
    """

    def __init__(self, latent_dim=100, feature_maps=64):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(

            # Project latent vector into 7x7 feature maps
            nn.Linear(latent_dim, feature_maps * 2 * 7 * 7),
            nn.BatchNorm1d(feature_maps * 2 * 7 * 7),
            nn.ReLU(True),

            # Reshape: (batch_size, 128, 7, 7)
            nn.Unflatten(1, (feature_maps * 2, 7, 7)),

            # Upsample: 7x7 -> 14x14
            nn.ConvTranspose2d(
                feature_maps * 2,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # Upsample: 14x14 -> 28x28
            nn.ConvTranspose2d(
                feature_maps,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Output pixel range: [-1, 1]
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


if __name__ == "__main__":

    # Test the Generator
    latent_dim = 100
    batch_size = 64

    generator = Generator(latent_dim=latent_dim)

    noise = torch.randn(batch_size, latent_dim)

    generated_images = generator(noise)

    print("Input noise shape:", noise.shape)
    print("Generated images shape:", generated_images.shape)
    print("Minimum pixel value:", generated_images.min().item())
    print("Maximum pixel value:", generated_images.max().item())
