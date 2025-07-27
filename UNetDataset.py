from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import tifffile as tiff
import numpy as np
import random

class UNetDataset(Dataset):
    def __init__(self, volume_path, label_path, augment=False):
        # Load full volume and label stacks (as numpy arrays)
        self.volume = tiff.imread(volume_path)  # shape: (N, H, W)
        self.labels = tiff.imread(label_path)   # shape: (N, H, W)
        self.augment = augment

        # Image transform: resize, make 3 channels, convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Grayscale(num_output_channels=3),  # fake 3 channels
            transforms.ToTensor()
        ])

        # Label transform: resize with NEAREST for masks, convert to tensor
        self.label_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def augment_image_and_mask(self, img, mask):
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)

        return img, mask

    def __getitem__(self, index):
        # Load single slice
        img = self.volume[index]   # shape: (H, W)
        mask = self.labels[index]  # shape: (H, W)

        # Convert to PIL
        img = Image.fromarray(img).convert("L")  # grayscale
        mask = Image.fromarray(mask).convert("L")  # grayscale

        # Apply augmentations
        if self.augment:
            img, mask = self.augment_image_and_mask(img, mask)

        # Apply transforms
        img = self.transform(img)  # [3, 512, 512]
        mask = self.label_transform(mask)  # [1, 512, 512]

        # Convert mask to binary float for BCEWithLogitsLoss
        mask = (mask > 0).float()  # values in [0.0, 1.0]

        return img, mask

    def __len__(self):
        return self.volume.shape[0]
