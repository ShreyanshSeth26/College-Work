import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torchvision.models as models
import wandb
from collections import Counter
from dataset_class import CamVidDataset

class_dict_path = r"Segmentation\Dataset\class_dict.csv"
train_image_dir = r"Segmentation\Dataset\train"
train_mask_dir  = r"Segmentation\Dataset\train_labels"
test_image_dir  = r"Segmentation\Dataset\test_images"
test_mask_dir   = r"Segmentation\Dataset\test_labels"
encoder_path    = r"Segmentation\encoder_model.pth"
decoder_path    = r"Segmentation\decoder.pth"
deeplabv3_path  = r"Segmentation\deeplabv3.pth"

class_dict_df = pd.read_csv(class_dict_path)
class_mapping = {(row["r"], row["g"], row["b"]): idx for idx, row in class_dict_df.iterrows()}
num_classes = len(class_dict_df)

image_transform = transforms.Compose([
    transforms.Resize((360, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = CamVidDataset(train_image_dir, train_mask_dir)
dataset_test  = CamVidDataset(test_image_dir,  test_mask_dir)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=4)
dataloader_test  = DataLoader(dataset_test,  batch_size=8, shuffle=False, num_workers=4)

def visualize_class_distribution(dataset):
    class_counts = Counter()
    for _, mask in dataset:
        unique, counts = torch.unique(mask, return_counts=True)
        for cls, cnt in zip(unique.numpy(), counts.numpy()):
            class_counts[int(cls)] += cnt
    plt.figure(figsize=(12, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel("Class ID")
    plt.ylabel("Pixel Count")
    plt.title("Class Distribution in CAMVid Dataset")
    plt.show()

def visualize_images_with_masks(dataset, num_samples):
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    for i in range(num_samples):
        image, mask = dataset[i]
        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        axes[i, 0].imshow(img_np)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(mask.numpy(), cmap="jet")
        axes[i, 1].axis("off")
    plt.show()

class SegNet_Encoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super().__init__()
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.ConvEn11 = nn.Conv2d(in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11   = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12   = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21   = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22   = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31   = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32   = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33   = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41   = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42   = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43   = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51   = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52   = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53   = nn.BatchNorm2d(512, momentum=BN_momentum)
    def forward(self, x):
        x = F.relu(self.BNEn11(self.ConvEn11(x)))
        x = F.relu(self.BNEn12(self.ConvEn12(x)))
        x, ind1 = self.MaxEn(x)
        size1 = x.size()
        x = F.relu(self.BNEn21(self.ConvEn21(x)))
        x = F.relu(self.BNEn22(self.ConvEn22(x)))
        x, ind2 = self.MaxEn(x)
        size2 = x.size()
        x = F.relu(self.BNEn31(self.ConvEn31(x)))
        x = F.relu(self.BNEn32(self.ConvEn32(x)))
        x = F.relu(self.BNEn33(self.ConvEn33(x)))
        x, ind3 = self.MaxEn(x)
        size3 = x.size()
        x = F.relu(self.BNEn41(self.ConvEn41(x)))
        x = F.relu(self.BNEn42(self.ConvEn42(x)))
        x = F.relu(self.BNEn43(self.ConvEn43(x)))
        x, ind4 = self.MaxEn(x)
        size4 = x.size()
        x = F.relu(self.BNEn51(self.ConvEn51(x)))
        x = F.relu(self.BNEn52(self.ConvEn52(x)))
        x = F.relu(self.BNEn53(self.ConvEn53(x)))
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        return x, [ind1, ind2, ind3, ind4, ind5], [size1, size2, size3, size4, size5]

class SegNet_Decoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=num_classes, BN_momentum=0.5):
        super().__init__()
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.stage5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=BN_momentum),
            nn.ReLU()
        )
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.stage4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=BN_momentum),
            nn.ReLU()
        )
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=BN_momentum),
            nn.ReLU()
        )
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=BN_momentum),
            nn.ReLU()
        )
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, out_chn, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chn, momentum=BN_momentum),
            nn.ReLU(),
            nn.Conv2d(out_chn, out_chn, kernel_size=3, padding=1)
        )
    def forward(self, x, indexes, sizes):
        ind1, ind2, ind3, ind4, ind5 = indexes
        size1, size2, size3, size4, size5 = sizes
        x = self.unpool5(x, ind5, output_size=size4)
        x = self.stage5(x)
        x = self.unpool4(x, ind4, output_size=size3)
        x = self.stage4(x)
        x = self.unpool3(x, ind3, output_size=size2)
        x = self.stage3(x)
        x = self.unpool2(x, ind2, output_size=size1)
        x = self.stage2(x)
        x = self.unpool1(x, ind1)
        x = self.stage1(x)
        return x

class SegNet_Pretrained(nn.Module):
    def __init__(self, encoder_path, in_chn=3, out_chn=32):
        super().__init__()
        self.encoder = SegNet_Encoder(in_chn, out_chn)
        self.decoder = SegNet_Decoder(in_chn, out_chn)
        encoder_state_dict = torch.load(encoder_path, map_location='cpu')
        self.encoder.load_state_dict(encoder_state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
    def forward(self, x):
        x, indexes, sizes = self.encoder(x)
        x = self.decoder(x, indexes, sizes)
        return x

def train_segnet_decoder(epochs):
    import wandb
    wandb.init(project="Segmentation", name="Decoder", reinit=True)
    device = torch.device("cuda")
    segnet_pretrained = SegNet_Pretrained(encoder_path, in_chn=3, out_chn=num_classes).to(device)
    decoder = SegNet_Decoder(in_chn=3, out_chn=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    for epoch in range(epochs):
        epoch_loss = 0.0
        decoder.train()
        for images, masks in dataloader_train:
            images, masks = images.to(device), masks.to(device)
            with torch.no_grad():
                encoded, indexes, sizes = segnet_pretrained.encoder(images)
            outputs = decoder(encoded, indexes, sizes)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader_train)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    torch.save(decoder.state_dict(), decoder_path)

def evaluate_segnet(decoder_ckpt_path):
    device = torch.device("cuda")
    segnet_pretrained = SegNet_Pretrained(encoder_path, in_chn=3, out_chn=num_classes).to(device)
    decoder_eval = SegNet_Decoder(in_chn=3, out_chn=num_classes).to(device)
    decoder_eval.load_state_dict(torch.load(decoder_ckpt_path, map_location=device))
    decoder_eval.eval()
    total_correct = 0
    total_pixels  = 0
    intersection_per_class = np.zeros(num_classes, dtype=np.float64)
    union_per_class        = np.zeros(num_classes, dtype=np.float64)
    tp_per_class = np.zeros(num_classes, dtype=np.float64)
    fp_per_class = np.zeros(num_classes, dtype=np.float64)
    fn_per_class = np.zeros(num_classes, dtype=np.float64)
    with torch.no_grad():
        for images, masks in dataloader_test:
            images = images.to(device)
            masks  = masks.to(device)
            encoded, indexes, sizes = segnet_pretrained.encoder(images)
            outputs = decoder_eval(encoded, indexes, sizes)
            preds   = torch.argmax(outputs, dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels  += masks.numel()
            for cls_id in range(num_classes):
                intersection = ((preds == cls_id) & (masks == cls_id)).sum().item()
                union = ((preds == cls_id) | (masks == cls_id)).sum().item()
                intersection_per_class[cls_id] += intersection
                union_per_class[cls_id]        += union
                tp = intersection
                fp = ((preds == cls_id) & (masks != cls_id)).sum().item()
                fn = ((preds != cls_id) & (masks == cls_id)).sum().item()
                tp_per_class[cls_id] += tp
                fp_per_class[cls_id] += fp
                fn_per_class[cls_id] += fn
    pixel_acc = total_correct / total_pixels
    iou_list   = []
    dice_list  = []
    prec_list  = []
    recall_list= []
    print("Class | Dice   | IoU   | Precision | Recall")
    for c in range(num_classes):
        i = intersection_per_class[c]
        u = union_per_class[c]
        iou_c = i / u if u > 0 else 0
        denom_dice = (2*i + fp_per_class[c] + fn_per_class[c])
        dice_c = 2*i / denom_dice if denom_dice > 0 else 0
        tp = tp_per_class[c]
        fp = fp_per_class[c]
        precision_c = tp/(tp+fp) if (tp+fp)>0 else 0
        fn = fn_per_class[c]
        recall_c = tp/(tp+fn) if (tp+fn)>0 else 0
        iou_list.append(iou_c)
        dice_list.append(dice_c)
        prec_list.append(precision_c)
        recall_list.append(recall_c)
        print(f"{c:5d} | {dice_c:.4f} | {iou_c:.4f} |   {precision_c:.4f}   |  {recall_c:.4f}")
    mIoU = np.mean(iou_list)
    print("\nPixel Accuracy =", pixel_acc)
    print("mIoU =", mIoU, "\n")
    print("IoU distribution with 0.1 intervals (for each class)")
    intervals = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for c in range(num_classes):
        iou_c = iou_list[c]
        idx = min(int(iou_c * 10), 9)
        print(f"Class {c}: IoU={iou_c:.2f} => in interval [{intervals[idx]}, {intervals[idx+1]})")

id_to_rgb = {}
for i, row in class_dict_df.iterrows():
    id_to_rgb[i] = (row["r"], row["g"], row["b"])

def colorize_mask(mask_tensor):
    mask_np = mask_tensor.cpu().numpy()
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, (r, g, b) in id_to_rgb.items():
        color_img[mask_np == cls_id] = (r, g, b)
    return color_img

def visualize_segnet(decoder_ckpt_path, class_interest, iou_threshold=0.5, max_samples=3):
    device = torch.device("cuda")
    segnet_pretrained = SegNet_Pretrained(encoder_path, in_chn=3, out_chn=num_classes).to(device)
    decoder_eval = SegNet_Decoder(in_chn=3, out_chn=num_classes).to(device)
    decoder_eval.load_state_dict(torch.load(decoder_ckpt_path, map_location=device))
    decoder_eval.eval()
    low_iou_samples = {cls: [] for cls in class_interest}
    with torch.no_grad():
        for images, masks in dataloader_test:
            images, masks = images.to(device), masks.to(device)
            encoded, indexes, sizes = segnet_pretrained.encoder(images)
            preds_raw = decoder_eval(encoded, indexes, sizes)
            preds     = torch.argmax(preds_raw, dim=1)
            for b in range(images.size(0)):
                pred_b = preds[b]
                mask_b = masks[b]
                for cls in class_interest:
                    if len(low_iou_samples[cls]) >= max_samples:
                        continue
                    intersection = ((pred_b == cls) & (mask_b == cls)).sum().item()
                    union        = ((pred_b == cls) | (mask_b == cls)).sum().item()
                    iou_c        = intersection / union if union > 0 else 0.0
                    if iou_c <= iou_threshold:
                        low_iou_samples[cls].append((images[b].cpu(), mask_b.cpu(), pred_b.cpu(), iou_c))
    for cls in class_interest:
        samples = low_iou_samples[cls]
        print(f"\nClass {cls}, found {len(samples)} images with IoU ≤ {iou_threshold}")
        if len(samples) == 0:
            print("No samples for this class at that IoU threshold.")
            continue
        fig, axes = plt.subplots(len(samples), 3, figsize=(10, 4*len(samples)))
        if len(samples) == 1:
            axes = [axes]
        for i, (img_tensor, gt_tensor, pred_tensor, iou_val) in enumerate(samples):
            row_axes = axes[i]
            img_np = img_tensor.permute(1,2,0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            gt_color   = colorize_mask(gt_tensor)
            pred_color = colorize_mask(pred_tensor)
            row_axes[0].imshow(img_np)
            row_axes[0].set_title(f"Image (IoU={iou_val:.2f})")
            row_axes[0].axis("off")
            row_axes[1].imshow(gt_color)
            row_axes[1].set_title(f"GT Mask (class {cls})")
            row_axes[1].axis("off")
            row_axes[2].imshow(pred_color)
            row_axes[2].set_title("Predicted Mask")
            row_axes[2].axis("off")
        plt.tight_layout()
        plt.show()
    
class DeepLabV3(nn.Module):
    def __init__(self, num_classes=32):
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']


def train_deeplabv3(num_epochs=10, lr=0.0001, save_path=deeplabv3_path):
    import wandb
    wandb.init(project="Segmentation", name="Deeplabv3 Finetune", reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3(num_classes=num_classes).to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks in dataloader_train:
            if images.shape[0] == 1:
                continue
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader_train)
        wandb.log({"epoch": epoch + 1, "deeplabv3_train_loss": avg_loss})
        print(f"[DeepLabV3] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"DeepLabV3 weights saved to {save_path}")

    
def evaluate_deeplabv3(ckpt_path= deeplabv3_path):
    device = torch.device("cuda")
    model = DeepLabV3(num_classes=num_classes).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    total_correct = 0
    total_pixels  = 0

    intersection_per_class = np.zeros(num_classes, dtype=np.float64)
    union_per_class = np.zeros(num_classes, dtype=np.float64)
    tp_per_class = np.zeros(num_classes, dtype=np.float64)
    fp_per_class = np.zeros(num_classes, dtype=np.float64)
    fn_per_class = np.zeros(num_classes, dtype=np.float64)

    with torch.no_grad():
        for images, masks in dataloader_test:
            images = images.to(device)
            masks  = masks.to(device)

            preds_raw = model(images)
            preds = torch.argmax(preds_raw, dim=1)

            total_correct += (preds == masks).sum().item()
            total_pixels  += masks.numel()

            for c in range(num_classes):
                intersection = ((preds == c) & (masks == c)).sum().item()
                union = ((preds == c) | (masks == c)).sum().item()
                intersection_per_class[c] += intersection
                union_per_class[c]  += union

                tp = intersection
                fp = ((preds == c) & (masks != c)).sum().item()
                fn = ((preds != c) & (masks == c)).sum().item()
                tp_per_class[c] += tp
                fp_per_class[c] += fp
                fn_per_class[c] += fn

    pixel_acc = total_correct / total_pixels

    print("Class |   Dice   |   IoU    | Precision | Recall")
    iou_list  = []
    dice_list = []

    for c in range(num_classes):
        i = intersection_per_class[c]
        u = union_per_class[c]
        iou_c = i / u if u > 0 else 0

        denom_dice = (2 * i + fp_per_class[c] + fn_per_class[c])
        dice_c = 2.0 * i / denom_dice if denom_dice > 0 else 0

        tp = tp_per_class[c]
        fp = fp_per_class[c]
        precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        fn = fn_per_class[c]
        recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        iou_list.append(iou_c)
        dice_list.append(dice_c)

        print(f"{c:5d} | {dice_c:8.4f} | {iou_c:8.4f} | {precision_c:9.4f} | {recall_c:6.4f}")

    mIoU = np.mean(iou_list)
    pixel_acc_percent = 100.0 * pixel_acc

    print(f"\nPixel Accuracy = {pixel_acc_percent:.2f}%")
    print(f"mIoU = {mIoU:.4f}\n")

    intervals = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    print("IoU distribution (0.1 intervals)")
    for c in range(num_classes):
        iou_c = iou_list[c]
        idx   = min(int(iou_c * 10), 9)
        print(f"Class {c}: IoU={iou_c:.2f} => in interval [{intervals[idx]}, {intervals[idx+1]})")
    
def visualize_deeplab(ckpt_path= deeplabv3_path, class_interest=[0,1,2], iou_threshold=0.5, max_samples=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    low_iou_samples = {cls: [] for cls in class_interest}

    with torch.no_grad():
        for images, masks in dataloader_test:
            images, masks = images.to(device), masks.to(device)

            preds_raw = model(images)
            preds     = torch.argmax(preds_raw, dim=1)

            for i in range(images.size(0)):
                img_i   = images[i].cpu()
                mask_i  = masks[i]
                pred_i  = preds[i]

                for cls in class_interest:
                    if len(low_iou_samples[cls]) >= max_samples:
                        continue
                    intersection = ((pred_i == cls) & (mask_i == cls)).sum().item()
                    union        = ((pred_i == cls) | (mask_i == cls)).sum().item()
                    iou_c        = intersection / union if union > 0 else 0.0

                    if iou_c <= iou_threshold:
                        low_iou_samples[cls].append((img_i, mask_i.cpu(), pred_i.cpu(), iou_c))
                        
    for cls in class_interest:
        samples = low_iou_samples[cls]
        print(f"\nClass {cls}, found {len(samples)} images with IoU ≤ {iou_threshold}")
        if len(samples) == 0:
            print("No samples for this class at that IoU threshold.")
            continue
        
        fig, axes = plt.subplots(len(samples), 3, figsize=(10, 4 * len(samples)))
        if len(samples) == 1:
            axes = [axes]

        for i, (img_tensor, gt_tensor, pred_tensor, iou_val) in enumerate(samples):
            row_axes = axes[i]
            img_np = img_tensor.permute(1,2,0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            gt_color   = colorize_mask(gt_tensor)
            pred_color = colorize_mask(pred_tensor)
            row_axes[0].imshow(img_np)
            row_axes[0].set_title(f"Image (IoU={iou_val:.2f})")
            row_axes[0].axis("off")
            row_axes[1].imshow(gt_color)
            row_axes[1].set_title(f"GT Mask (class {cls})")
            row_axes[1].axis("off")
            row_axes[2].imshow(pred_color)
            row_axes[2].set_title("Pred Mask")
            row_axes[2].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    visualize_class_distribution(dataset_train)
    visualize_images_with_masks(dataset_train, num_samples=7)
    print("TRAINING DECODER")
    train_segnet_decoder(25)
    print("\nEVALUATING ON TEST SET")
    evaluate_segnet(decoder_path)
    print("\nVISUALIZING LOW-IOU CASES")
    visualize_segnet(decoder_ckpt_path=decoder_path, class_interest=[0,1,2], iou_threshold=0.5, max_samples=3)
    print("Training DeepLabV3")
    train_deeplabv3(num_epochs=10, lr=0.0001, save_path=deeplabv3_path)
    evaluate_deeplabv3(deeplabv3_path)
    visualize_deeplab(ckpt_path=deeplabv3_path, class_interest=[0,1,2], iou_threshold=0.5, max_samples=3)