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

| Partie | Contenu | Quand |
|---|---|---|
| [0. Le serveur](00_serveur.md) | Se connecter au serveur GPU de l'ENSTA | en séance, 10 min |
| [1. Prise en main de PyTorch](01_pytorch.md) | Le tutoriel officiel *Training a Classifier*, et quelques questions | en séance |
| [2. Votre baseline](02_baseline.md) | Écrire vous-mêmes l'entraînement d'un ResNet18 | en séance, puis entraînement long |
| [3. Data augmentation](03_data_augmentation.md) | Visualiser l'augmentation, mesurer son effet, mixup | après le cours sur la data augmentation, à finir chez vous |

En parallèle : [la lecture d'articles](articles.md), à présenter en binôme.

## Les règles du jeu

- On travaille **en binôme**.
- **Vous écrivez le code.** On ne vous fournit que la description du réseau
  (`resnet.py`) et un notebook à trous, avec des cellules de vérification qui
  vous disent si vous êtes sur la bonne voie.
- Vous pouvez vous faire aider (documentation, camarades, assistant IA). Une
  seule condition : **être capables d'expliquer chaque ligne** que vous rendez.
  C'est la question qu'on vous posera.
- Le GPU est **partagé** entre toute la promotion : lisez les règles de la
  [partie 0](00_serveur.md#les-regles-dun-gpu-partage).

!!! note "Crédits"
    Ce TP reprend et adapte les labs 1 et 2 du cours
    [Efficient Deep Learning](https://github.com/brain-bzh/efficient-deep-learning)
    d'IMT Atlantique. Le code du ResNet est adapté de
    [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) (licence MIT).
