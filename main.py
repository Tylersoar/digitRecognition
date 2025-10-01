import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 🔁 1. Data preparation
transform = transforms.Compose([
    transforms.ToTensor()  # Keeps image in 1x28x28 format
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)



# 🧠 2. Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 1→8 channels, same size output
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                        # Downsample 28x28 → 14x14

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 8→16 channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                         # Downsample 14x14 → 7x7
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),              # Flatten [16 x 7 x 7] → [784]
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)         # Output layer for 10 digits
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ⚙️ 3. Set up training tools
model = CNN().to(device)




loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📈 4. Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    model.train()  # Enable dropout / batchnorm if used
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f} - Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "mnist_cnn.pt")
# Get one batch of data
images, labels = next(iter(train_loader))

# Move data to the same device as the model
images = images.to(device)

# Pass through the first conv layer
model.eval()
with torch.no_grad():
    conv1_out = model.conv_layers[0](images)  # First Conv2d
    conv1_out = model.conv_layers[1](conv1_out)  # ReLU

# Move data back to CPU for visualization
images_cpu = images.cpu()
conv1_out_cpu = conv1_out.cpu()

# Plot original image and first few feature maps
fig, axs = plt.subplots(1, 6, figsize=(15, 3))

# Original image
axs[0].imshow(images_cpu[0][0], cmap='gray')
axs[0].set_title("Original")
axs[0].axis('off')

# First 5 feature maps
for i in range(5):
    axs[i+1].imshow(conv1_out_cpu[0][i], cmap='viridis')
    axs[i+1].set_title(f"Feature {i+1}")
    axs[i+1].axis('off')

plt.tight_layout()
plt.show()