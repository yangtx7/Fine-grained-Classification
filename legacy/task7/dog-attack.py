import torch
import os
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from dataloader import StanfordDogsDataset

def preprocess(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

train_set = StanfordDogsDataset(root_dir=os.path.join(os.getcwd(), "data/StanfordDogs"), split="train",transform=preprocess)
test_set = StanfordDogsDataset(root_dir=os.path.join(os.getcwd(), "data/StanfordDogs"), split="test", transform=preprocess)
dataloaders = {
    'train': DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8),
    'val': DataLoader(test_set, batch_size=16, shuffle=True, num_workers=8)
}
dataset_sizes = {
    'train': len(train_set),
    'val': len(test_set)
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def gaussian_noise(inputs, mean=0., std=1.):
    noise = torch.randn_like(inputs)*std + mean
    return inputs + noise

writer = SummaryWriter('runs/dog_breed_experiment_53')
num_epochs = 25
for epoch in range(num_epochs):

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if phase == 'train':
                inputs = gaussian_noise(inputs)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                if phase == 'train':
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0)
                    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                    outputs = model(inputs)
                    loss_func = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            if phase == 'train':
                loss_func.backward()
                optimizer.step()

            running_loss += loss_func.item() * inputs.size(0)
            _, preds = outputs.max(1)
            running_corrects += preds.eq(labels).sum().item()

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Record loss and accuracy into TensorBoard
        writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
        writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)

    print()

writer.close()
