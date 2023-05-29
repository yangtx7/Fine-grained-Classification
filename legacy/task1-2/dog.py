import torch
import os
import math
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

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
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

writer = SummaryWriter('runs/dog_breed_experiment_1')
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

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Record loss and accuracy into TensorBoard
        writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
        writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)

    print()

writer.close()
