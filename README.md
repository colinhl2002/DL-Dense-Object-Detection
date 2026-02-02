# 🚀 Classification d'images avec l'utilisation de la Focal Loss

Ce repository GitHub contient une implementation de la fonction **Focal Loss** pour améliorer la **classification d'images** dans le cas d'un dataset présentant un **déséquilibre de classe**.
Il permet de se familiariser avec l'utilisation de la Focal Loss.

Le dataset retenu pour tester notre implémentation est un dataset binaire "Dog vs Cat" que l'on a volontairement rendu déséquilibré. Ainsi, la problématique est analogue au déséquilibre de classe "Background vs Foreground" que l'on retrouve dans la détection d'objet et qui est abordée dans l'article : 
https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf

## Détail du repository

Ce repository GitHub contient plusieurs fonctions Python qui servent à stocker nos fonctions / classes d'intérêts, un dossier qui contient les ressources et un notebook jupyter qui permet de parcourir le sujet :

* **eval.py** : fonction predict_one_epoch
* **loss.py** : classe FocalLoss qui implémente la fonction loss d'intérêt pour la classification Binaire et Multi-classes
* **model.py** : fonction cnn_model qui définie l'architecture de notre réseau de neurones CNN
* **train.py** : fonction train_one_epoch
* **utils.py** : classe NpArrayDataset qui permet de charger des images numpy dans un format compatible avec Pytorch et fonction reduce_datasets qui permet de créer un déséquilibre de classe dans notre dataset

* **image_classification.ipynb** : notebook jupyter dans lequel on va entraîner notre modèle CNN sur le dataset "Dog vs Cat" déséquilibré avec l'utilisation de la fonction BCELoss() dans un premier temps, puis FocalLoss() dans un second temps
**Note** : Ce notebook Jupyter peut facilement être exécuté en local.

* **ressources** : dossier qui contient la présentation powerpoint du sujet et le pdf de l'article "Focal Loss for Dense Object Detection"

## Installation

1) Clonez le dépôt et installez les dépendances :
```bash
git clone [https://github.com/colinhl2002/Image-Classification-with-FocalLoss.git](https://github.com/colinhl2002/Image-Classification-with-FocalLoss.git)
cd Image-Classification-with-FocalLoss
pip install torch torchvision matplotlib numpy pandas seaborn tqdm sklearn
```

2) Télécharger en local le dataset via le lien :
https://drive.google.com/drive/u/0/folders/1dZvL1gi5QLwOGrfdn9XEsi4EnXx535bD (Dog vs Cat dataset)

3) Créer un dossier **data** à la racine du projet :
```bash
mkdir data
```
et y mettre les csv suivants :
- input_test.csv
- input.csv
- labels_test.csv
- labels.csv

## Ressources additionnels

Pour compléter ce notebook :
- une présentation Powerpoint du sujet (dans le dossier ressources de ce repo)
- le PDF de l'article 'Focal Loss for Dense Object Detection' (dans le dossier ressources de ce repo)
- une vidéo de présentation du sujet (accessiblke sur YouTube via le lien : https://youtu.be/FZB2hb3dPGI)

## Aperçu

![Capture d'écran de l'application](votre-lien-image.png)