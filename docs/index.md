# TP Deep Learning : de PyTorch à votre baseline

Module **SE & IA**, partie *Efficient Deep Learning*.

## L'objectif

À la fin de ce TP, chaque binôme a **un ResNet18 entraîné sur CIFAR-10**, avec ses
courbes d'apprentissage et ses poids sauvegardés.

Ce modèle est votre **baseline** : 11 millions de paramètres, environ 93 % de
précision. Toute la suite du module consiste à le rendre plus petit et moins
coûteux (quantification, élagage, factorisation, distillation) **en perdant le
moins de précision possible**. Gardez-le précieusement.

## Le déroulé

| Partie | Contenu |
|---|---|
| [0. Google Colab](00_colab.md) | Un GPU dans le navigateur, et les données |
| [1. Prise en main de PyTorch](01_pytorch.md) | Le tutoriel officiel *Training a Classifier*, et quelques questions |
| [2. Votre baseline](02_baseline.md) | Écrire vous-mêmes l'entraînement d'un ResNet18 |
| [3. Data augmentation](03_data_augmentation.md) | Visualiser l'augmentation, mesurer son effet, mixup |

On commence aujourd'hui et on avance à votre rythme : ce qui n'est pas fini
aujourd'hui se poursuivra à la séance suivante.

En parallèle : [la lecture d'articles](articles.md), à présenter en binôme.

## Comment on travaille

- **En binôme.**
- **Vous écrivez le code.** On vous fournit la description du réseau
  (`resnet.py`) et un notebook à trous, avec des cellules de vérification qui
  vous disent si vous êtes sur la bonne voie.
- Le but n'est pas d'aller vite, c'est de **comprendre ce que fait votre code** :
  c'est lui que vous allez transformer pendant tout le reste du module pour
  compresser votre réseau. Aidez-vous de ce que vous voulez (documentation,
  voisins, assistants IA) tant que vous comprenez ce que vous écrivez.

!!! note "Crédits"
    Ce TP reprend et adapte les labs 1 et 2 du cours
    [Efficient Deep Learning](https://github.com/brain-bzh/efficient-deep-learning)
    d'IMT Atlantique. Le code du ResNet est adapté de
    [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (licence MIT).
