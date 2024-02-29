import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, DataLoader

class CustomISICDataset(Dataset):
    def __init__(self, csv_file, root_dir, split, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, 'img')
        self.img_labels = pd.read_csv(csv_file)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.img_labels['diagnostic'].unique())}

        # Filter img_labels to include only files that exist
        self.img_labels['img_path'] = self.img_labels['image'].apply(lambda x: os.path.join(self.img_dir, x + '.jpeg'))
        self.img_labels = self.img_labels[self.img_labels['img_path'].apply(os.path.exists)]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx]['img_path']
        image = Image.open(img_path)
        label_name = self.img_labels.iloc[idx, 1]
        label = self.label_to_idx[label_name]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


class SkinCancerDataLoaders:
    def __init__(self, images_dir, image_size, batch_size, num_workers, interpolation):
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Assuming separate train/val/test directories under images_dir and a single labels.csv at images_dir level
        self.train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.val_transforms = self.train_transforms  # Assuming same transforms for simplicity; adjust if necessary

        self.train_loader = self._create_loader('train', self.train_transforms)
        self.val_loader = self._create_loader('val', self.val_transforms)
        self.test_loader = self._create_loader('test', self.val_transforms)

    def _create_loader(self, split, transforms):
        csv_file = os.path.join(self.images_dir, 'labels.csv')
        dataset = CustomISICDataset(csv_file=csv_file, root_dir=self.images_dir, split=split, transform=transforms)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=(split == 'train'), num_workers=self.num_workers, pin_memory=True)
