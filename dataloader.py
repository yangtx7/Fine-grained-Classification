import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        self.labels = data_split["labels"].squeeze() - 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
    
class SquareCroppedStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        self.labels = data_split["labels"].squeeze() - 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img_name = img_name.replace("Images", "squareCroppedImages")
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
    
class SingleClassSquareCroppedStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, classid, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        data_split_file = os.path.join(root_dir, "train_list.mat")

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list2 = data_split["file_list"]
        self.labels2 = data_split["labels"].squeeze() - 1

        self.file_list = []
        self.labels = []
        for i in range(len(self.labels2)):
            if self.labels2[i] != classid:
                continue
            self.labels.append(self.labels2[i])
            self.file_list.append(self.file_list2[i])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img_name = img_name.replace("Images", "squareCroppedImages")
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img.unsqueeze(0), label
    
class CroppedStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        self.labels = data_split["labels"].squeeze() - 1

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img_name = img_name.replace("Images", "croppedImages")
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
    
class BirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1 
    
class CroppedBirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'croppedimages', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1

class SquareCroppedBirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'squarecroppedimages', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1
    
class SingleClassSquareCroppedBirdsDataset(Dataset):
    def __init__(self, root_dir, classid, transform=None):
        self.root_dir = root_dir
        self.classid = classid
        self.transform = transform

        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')

        self.data_info = self.data_info[(self.data_info.is_training_image == 1) & (self.data_info.class_id == self.classid)]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'squarecroppedimages', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image.unsqueeze(0), label - 1
  