
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator for Fashion-MNIST.

    Input:
        Images of shape (batch_size, 1, 28, 28)

    Output:
        Raw logits of shape (batch_size, 1)
    """

    def __init__(self, feature_maps=64):
        super().__init__()

        self.model = nn.Sequential(

            # Input: (batch_size, 1, 28, 28)

            # 28x28 -> 14x14
            nn.Conv2d(
                1,
                feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 7x7
            nn.Conv2d(
                feature_maps,
                feature_maps * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Flatten: 128 * 7 * 7 = 6272
            nn.Flatten(),

            # Final classification
            nn.Linear(feature_maps * 2 * 7 * 7, 1)
        )

    def forward(self, images):
        return self.model(images)


if __name__ == "__main__":

    # Test the Discriminator
    batch_size = 64

    discriminator = Discriminator()

    # Simulate a batch of Fashion-MNIST images
    images = torch.randn(batch_size, 1, 28, 28)

    predictions = discriminator(images)

    print("Input images shape:", images.shape)
    print("Discriminator output shape:", predictions.shape)
    print("Sample logits:", predictions[:5].detach().flatten())
