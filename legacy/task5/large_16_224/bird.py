import torch
import os
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from timm import create_model
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from dataloader import BirdsDataset

def preprocess(image):
    width, height = image.size
    if width > height and width > 224:
        height = math.floor(224 * height / width)
        width = 224
    elif width < height and height > 224:
        width = math.floor(224 * width / height)
        height = 224
    pad_values = (
        (224 - width) // 2 + (0 if width % 2 == 0 else 1),
        (224 - height) // 2 + (0 if height % 2 == 0 else 1),
        (224 - width) // 2,
        (224 - height) // 2,
    )
    return transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)

def train_one_epoch(model, dataloader, loss_func, optimizer, device, scheduler=None, scaler=None, mixup_fn=None):
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Apply MixUp
        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # Use autocast to enable mixed-precision training
            with autocast():
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

            # Scale the loss and call backward to update gradients
            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params and call optimizer.step() to adjust params
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            _, preds = torch.max(outputs, 1)

        # running_loss += loss.item() * inputs.size(0)
        # running_corrects += torch.sum(preds == labels.data)
        preds = torch.softmax(outputs, dim=1)
        preds = preds.argmax(dim=1, keepdim=True)
        running_corrects += preds.eq(labels.argmax(dim=1, keepdim=True)).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)

    return epoch_loss, epoch_acc

def val_one_epoch(model, dataloader, loss_func, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc

train_set = BirdsDataset(root_dir=os.path.join(os.getcwd(), "data/CUB-200-2011/CUB_200_2011"), split="train",transform=preprocess)
test_set = BirdsDataset(root_dir=os.path.join(os.getcwd(), "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess)

dataloaders = {
    'train': DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8),
    'val': DataLoader(test_set, batch_size=16, shuffle=True, num_workers=8)
}
dataset_sizes = {
    'train': len(train_set),
    'val': len(test_set)
}

model = create_model('vit_large_patch16_224', pretrained=True, num_classes=1000)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_loss_func = SoftTargetCrossEntropy().to(device)
val_loss_func = nn.CrossEntropyLoss().to(device)

writer = SummaryWriter('runs/bird_experiment_1')
num_epochs = 25

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=num_epochs, steps_per_epoch=len(dataloaders['train']))
scaler = GradScaler()
mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=0.2, prob=0.5, switch_prob=0.5, mode='batch')

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, dataloaders['train'], train_loss_func, optimizer, device, scheduler, scaler, mixup_fn)
    val_loss, val_acc = val_one_epoch(model, dataloaders['val'], val_loss_func, device)

    print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')

    # Write to tensorboard
    writer.add_scalar('Train loss', train_loss, epoch)
    writer.add_scalar('Train Accuracy', train_acc, epoch)

    writer.add_scalar('Validation Loss', val_loss, epoch)
    writer.add_scalar('Validation Accuracy', val_acc, epoch)

writer.close()