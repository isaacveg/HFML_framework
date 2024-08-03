import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x

# Initialize the model
model = AlexNet()

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
memory_orginal = total_params * 4 / (1024 ** 2)
# Memory required for original model:
print(f"Memory required for the original model: {memory_orginal:.2f} MB")

# Calculate the size of the Hessian matrix
hessian_size = total_params ** 2

# Memory required for the Hessian matrix in bytes (assuming float32, 4 bytes per float)
memory_bytes = hessian_size * 4

# Convert to gigabytes
memory_gb = memory_bytes / (1024 ** 3)

print(f"Total number of parameters: {total_params}")
print(f"Memory required for the Hessian matrix: {memory_gb:.2f} GB")