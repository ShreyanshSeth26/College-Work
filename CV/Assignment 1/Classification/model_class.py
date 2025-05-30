import io
import random
import os
import torch
import torch.nn as nn
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.manifold import TSNE
from torchvision import models
from collections import Counter                     
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from dataset_class import RussianWildlifeDataset
from dataset_class import AugmentedRussianWildlifeDataset

print("For Convnet")
device = torch.device("cuda")
os.environ["WANDB_MODE"] = "online"

DATA_DIR = r"Classification\Dataset"
CLASS_MAPPING = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4,
                 'dog': 5, 'people': 6, 'roe_deer': 7, 'sika_deer': 8, 'wild_boar': 9}
CLASS_MAPPING_INV = {v: k for k, v in CLASS_MAPPING.items()}
torch.manual_seed(2022484)

transform = v2.Compose([
    v2.Resize((224, 224)), 
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = RussianWildlifeDataset(DATA_DIR, transform=transform)

def print_class_counts(dataset, dataset_name):
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]
    else:
        labels = dataset.labels

    class_counts = Counter(labels)
    
    print(f" {dataset_name} Class Counts:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"   {CLASS_MAPPING_INV[class_idx]}: {count} images")
    print(f"   Total: {len(dataset)} images\n")


train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=[dataset.labels[i] for i in range(len(dataset))])
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

print_class_counts(dataset, "Full Dataset")
print_class_counts(train_dataset, "Training Set")
print_class_counts(val_dataset, "Validation Set")

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

wandb.init(project="Wildlife Classification", name="Convnet", reinit=True, resume=False)
wandb.config.update({"batch_size": batch_size, "image_size": (224, 224)})

def plot_class_distribution(dataset, title):
    if isinstance(dataset, torch.utils.data.Subset):
        labels = [dataset.dataset.labels[i] for i in dataset.indices]  
    else:
        labels = dataset.labels  

    label_counts = Counter(labels)
    class_names = [CLASS_MAPPING_INV[i] for i in sorted(label_counts.keys())]

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, label_counts.values())
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.xticks(rotation=90)
    wandb.log({title: wandb.Image(plt)})
    plt.show()
    plt.close()

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

model = ConvNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler()

def train_convnet(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    all_preds, all_labels = [], []
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        wandb.log({"epoch": epoch + 1, 
                   "train_loss": train_loss, 
                   "train_accuracy": train_accuracy, 
                   "val_loss": val_loss, 
                   "val_accuracy": val_accuracy})
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
    return all_labels, all_preds
all_labels, all_preds = train_convnet(model, train_loader, val_loader, criterion, optimizer, epochs=10)

os.makedirs("Classification/weights/", exist_ok=True)
torch.save(model.state_dict(), "Classification/weights/convnet.pth")
print("Model saved successfully.")

def evaluate_convnet(model, val_loader, criterion, num_classes=10):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= val_total
    val_accuracy = val_correct / val_total
    
    f1 = f1_score(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    wandb.log({"final_val_accuracy": val_accuracy, "val_f1_score": f1})
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_MAPPING.keys(), yticklabels=CLASS_MAPPING.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    print(f"Validation Accuracy: {val_accuracy:.4f}, F1-Score: {f1:.4f}")
    
evaluate_convnet(model, val_loader, criterion)
plot_class_distribution(dataset, "Full Dataset Distribution")
plot_class_distribution(train_dataset, "Training Set Distribution")
plot_class_distribution(val_dataset, "Validation Set Distribution")

def misclassified_images(model, val_loader, num_images=3):
    model.eval()
    misclassified = {cls: [] for cls in CLASS_MAPPING.keys()}

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            misclassified_idx = (preds != labels).cpu().numpy()
            for i in range(len(images)):
                if misclassified_idx[i]:
                    true_label = CLASS_MAPPING_INV[int(labels[i].cpu().numpy())]
                    pred_label = CLASS_MAPPING_INV[int(preds[i].cpu().numpy())]

                    if len(misclassified[true_label]) < num_images:
                        misclassified[true_label].append((images[i].cpu(), pred_label))
    return misclassified

def plot_misclassified(misclassified):
    for true_label, images in misclassified.items():
        if len(images) == 0:
            continue

        fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
        if len(images) == 1:
            axes = [axes]

        for i, (img, pred_label) in enumerate(images):
            img = img.permute(1, 2, 0).numpy()
            img = (img * 0.5) + 0.5

            axes[i].imshow(img)
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=10)
            axes[i].axis("off")

        plt.suptitle(f"Misclassified Images for {true_label}", fontsize=12)
        plt.tight_layout()
        wandb.log({true_label: wandb.Image(plt)})
        plt.show()

misclassified_samples = misclassified_images(model, val_loader, num_images=3)
plot_misclassified(misclassified_samples)
wandb.finish()










print("\nFor Resnet18")
DATA_DIR = r"Classification\Dataset"
CLASS_MAPPING = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4, 
                 'dog': 5, 'people': 6, 'roe_deer': 7, 'sika_deer': 8, 'wild_boar': 9}
CLASS_MAPPING_INV = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

transform = v2.Compose([
    v2.Resize((224, 224)), 
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = RussianWildlifeDataset(DATA_DIR, transform=transform)
train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=[dataset.labels[i] for i in range(len(dataset))])
train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

wandb.init(project="Wildlife Classification", name="Resnet18", reinit=True, resume=False)
wandb.config.update({"batch_size": batch_size, "image_size": (224, 224)})

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_resnet(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()    
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

train_resnet(model, train_loader, val_loader, criterion, optimizer, epochs=10)

os.makedirs("Classification/weights/", exist_ok=True)
torch.save(model.state_dict(), "Classification/weights/resnet.pth")
print("Model saved successfully.")

def evaluate_resnet(model, val_loader, criterion):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= val_total
    val_accuracy = val_correct / val_total
    final_f1 = f1_score(all_labels, all_preds, average="weighted")
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final F1-Score: {final_f1:.4f}")

    wandb.log({"final_accuracy": val_accuracy, "final_f1_score": final_f1})
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_MAPPING.keys(), yticklabels=CLASS_MAPPING.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()

evaluate_resnet(model, val_loader, criterion)
print("Evaluation Completed!")

def extract_features(model, dataloader):
    model.fc = torch.nn.Identity()
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            output = model(images)  
            features.append(output.cpu().numpy())
            labels.extend(lbls.numpy())

    return np.concatenate(features, axis=0), np.array(labels)

train_features, train_labels = extract_features(model, train_loader)
val_features, val_labels = extract_features(model, val_loader)

def tsne_2d(features, labels, dataset_name, title):
    tsne = TSNE(n_components=2, random_state=2022484)
    reduced_features = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=[CLASS_MAPPING_INV[i] for i in labels], palette="tab10", alpha=0.7)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    wandb.log({f"tsne_2d_{dataset_name}": wandb.Image(plt)})
    plt.show()

def tsne_3d(features, labels, title):
    tsne = TSNE(n_components=3, random_state=2022484)
    reduced_features = tsne.fit_transform(features)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], c=labels, cmap="tab10", alpha=0.7)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_zlabel("t-SNE Dim 3")
    plt.title(title)
    wandb.log({"tsne_3d_Val": wandb.Image(fig)})
    plt.show()

tsne_2d(train_features, train_labels, "Train", "2D t-SNE - Training Set")
tsne_2d(val_features, val_labels, "Val", "2D t-SNE - Validation Set")
tsne_3d(val_features, val_labels, "3D t-SNE - Validation Set")
print("Feature Extraction & Visualization Completed!")
wandb.finish()



print("\nFor Resnet Aug")
DATA_DIR = r"Classification\Dataset"
CLASS_MAPPING = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear': 4,
                 'dog': 5, 'people': 6, 'roe_deer': 7, 'sika_deer': 8, 'wild_boar': 9}
CLASS_MAPPING_INV = {v: k for k, v in CLASS_MAPPING.items()}
NUM_CLASSES = len(CLASS_MAPPING)

transform = v2.Compose([
    v2.Resize((224, 224)), 
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

augmented_dataset = AugmentedRussianWildlifeDataset(DATA_DIR, transform=transform, augment=True)
train_idx, val_idx = train_test_split(range(len(augmented_dataset)), test_size=0.2, stratify=[augmented_dataset.labels[i] for i in range(len(augmented_dataset))])

train_dataset = torch.utils.data.Subset(augmented_dataset, train_idx)
val_dataset = torch.utils.data.Subset(augmented_dataset, val_idx)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

wandb.init(project="Wildlife Classification", name="Resnet Aug", reinit=True, resume=False)
wandb.config.update({"batch_size": batch_size, "image_size": (224, 224)})

def visualize_augmentations(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, num_samples * 3))

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) // 4 - 1)

        for aug_type in range(4):
            img, _ = dataset[idx + aug_type * (len(dataset) // 4)]
            img = img.permute(1, 2, 0).numpy()
            img = (img * 0.5) + 0.5

            axes[i, aug_type].imshow(img)
            axes[i, aug_type].set_title(["Original", "Gaussian Noise", "Rotated", "Flipped"][aug_type])
            axes[i, aug_type].axis("off")
    plt.tight_layout()
    wandb.log({"Augmented Samples": wandb.Image(plt)})
    plt.show()
visualize_augmentations(augmented_dataset)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_augresnet(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    final_f1 = f1_score(all_labels, all_preds, average="weighted")
    final_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

    print(f"\nFinal Validation Accuracy: {final_accuracy:.4f}")
    print(f"Final F1-Score: {final_f1:.4f}")
    wandb.log({"final_accuracy_aug": final_accuracy, "final_f1_score_aug": final_f1})

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_MAPPING.keys(), yticklabels=CLASS_MAPPING.keys())
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Final Confusion Matrix")
    wandb.log({"final_confusion_matrix": wandb.Image(plt)})
    plt.close()

train_augresnet(model, train_loader, val_loader, criterion, optimizer, epochs=10)
torch.save(model.state_dict(), "Classification/weights/resnet_aug.pth")
wandb.finish()