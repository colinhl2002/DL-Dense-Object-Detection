# fonction d'affichage,...

import torch
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, target_transform=None):
        """
        hf_dataset: l'objet chargé via load_dataset
        transform: tes transformations torchvision
        """
        self.dataset = hf_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. On récupère l'entrée brute (dict)
        item = self.dataset[idx]
        
        # 2. On extrait l'image (déjà un objet PIL via HF) et le label
        image = item['image'].convert("RGB")
        label = item['label']

        # 3. Application des transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(int(label))
        # 4. On retourne le format standard PyTorch (image, label)
        return image, torch.tensor(int(label))