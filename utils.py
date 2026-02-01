import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Callable

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

        orig_w, orig_h = image.size
        print((orig_w, orig_h))

        # 3. Application des transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(int(label))
        # 4. On retourne le format standard PyTorch (image, label)
        return image, torch.tensor(int(label))
    
class BinaryDataset(Dataset):
    def __init__(self, subset, mapping):
        self.subset = subset
        self.mapping = mapping
        
    def __getitem__(self, index):
        image, label = self.subset[index]
        # On transforme le label (3 -> 0, 5 -> 1)
        return image, self.mapping[label]
        
    def __len__(self):
        return len(self.subset)
    

class NpArrayDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        image_transforms: Callable = None,
        label_transforms: Callable = None,
    ):
        self.images = images
        self.labels = labels
        self.image_transforms = image_transforms
        self.label_transforms = label_transforms

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index: int):
        x = self.images[index]
        y = self.labels[index]

        if self.image_transforms is not None:
            x = self.image_transforms(x)
        else:
            x = torch.tensor(x)

        if self.label_transforms is not None:
            y = self.label_transforms(y)
        else:
            y = torch.tensor(y)

        return x, y


def reduce_datasets(X, Y, ratio=0.1):

    y = Y.flatten()

    # 1. On récupère les indices de chaque classe
    ids_classe_0 = np.where(y == 0)[0]
    ids_classe_1 = np.where(y == 1)[0]

    # 2. On calcule combien de la classe positive "1" on veut garder (ratio)
    nb_img = int(len(ids_classe_1) * ratio)

    # 3. On tire au sort des indices de la classe 1
    np.random.seed(42)
    subids_classe_1 = np.random.choice(ids_classe_1, nb_img, replace=False)

    # 4. On combine les indices des "0" et "1"
    ids_finaux = np.concatenate([ids_classe_0, subids_classe_1])

    # 5. On mélange les indices pour ne pas avoir tous les 0 puis tous les 1
    np.random.shuffle(ids_finaux)

    # 6. On crée les nouveaux datasets
    X_unbalanced = X[ids_finaux]
    Y_unbalanced = Y[ids_finaux]

    return (X_unbalanced, Y_unbalanced)