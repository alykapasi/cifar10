import torch
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10_Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        images: numpy array of shape [N, 32, 32, 3]
        labels: list or numpy array of label indices
        transform: optional torchvision transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]

        img = self.transform(Image.fromarray(img)) if self.transform else torch.tensor(img).permute(2, 0, 1).float() / 255.0

        return img, lbl