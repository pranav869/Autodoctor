import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import io
# ================= CONFIG =================
DATA_DIR = "data/medical_images"   # must contain: NORMAL/, PNEUMONIA/
BATCH_SIZE = 16
EPOCHS = 5
NUM_CLASSES = 2  # NORMAL, PNEUMONIA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= IMAGE TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ================= DATASET =================
dataset = ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Class mapping:", dataset.class_to_idx)
# Expected: {'NORMAL': 0, 'PNEUMONIA': 1}

# ================= MODEL =================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ================= TRAINING LOOP =================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# ================= SAVE MODEL =================
BACKEND_DIR = "backend"
os.makedirs(BACKEND_DIR, exist_ok=True)

SAVE_PATH = os.path.join(BACKEND_DIR, "cnn_model.pt")
torch.save(model.state_dict(), SAVE_PATH)

print("✅ CNN model saved at:", SAVE_PATH)