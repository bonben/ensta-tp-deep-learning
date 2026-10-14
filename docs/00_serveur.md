# 0. Le serveur

L'ENSTA met à votre disposition un serveur JupyterHub équipé de GPU. Tout le TP
s'y déroule : rien à installer sur votre machine.

## Se connecter

1. Ouvrez <https://soiahub.ensta-bretagne.fr/jupyter> et connectez-vous avec vos
   identifiants ENSTA.
2. Dans le *Launcher*, ouvrez un **Terminal**.
3. Vérifiez que vous voyez un GPU, et que PyTorch le voit aussi :

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

!!! success "Résultat attendu"
    `nvidia-smi` affiche une ou plusieurs cartes, avec la mémoire utilisée et les
    processus en cours. La seconde commande affiche la version de PyTorch, puis
    `True`.

    Si vous obtenez `False`, appelez l'enseignant avant d'aller plus loin.

## Récupérer les fichiers du TP

Toujours dans le terminal :

```bash
mkdir -p tp_dl && cd tp_dl
wget https://bonben.github.io/ensta-tp-deep-learning/code/resnet.py
wget https://bonben.github.io/ensta-tp-deep-learning/code/baseline.ipynb
```

Puis les données, CIFAR-10 (170 Mo). Ne passez pas par le téléchargement
automatique de PyTorch, qui va chercher les données sur un serveur universitaire
très lent : prenez cette copie.

```bash
wget https://github.com/bonben/ensta-tp-deep-learning/releases/download/cifar10/cifar-10-python.tar.gz
mkdir -p data && tar xzf cifar-10-python.tar.gz -C data && rm cifar-10-python.tar.gz
ls data/cifar-10-batches-py      # doit lister data_batch_1 à data_batch_5, test_batch, etc.
```

Vos notebooks doivent être **dans le dossier `tp_dl`** : ils cherchent les données
dans `./data`, et PyTorch ne les retéléchargera pas puisqu'elles y sont déjà.

!!! tip "Si `wget` échoue"
    Téléchargez les deux fichiers depuis votre navigateur
    ([resnet.py](code/resnet.py), [baseline.ipynb](code/baseline.ipynb)), puis
    déposez-les dans le dossier `tp_dl` avec le bouton **Upload** de l'explorateur
    de fichiers de Jupyter (flèche vers le haut, en haut à gauche).

## Les règles d'un GPU partagé

Toute la promotion partage les mêmes cartes. Un binôme négligent peut bloquer
tous les autres.

- **Un seul entraînement à la fois par binôme.**
- **Arrêtez vos noyaux** quand vous avez fini avec un notebook : menu
  *Kernel > Shut Down Kernel*, ou l'onglet *Running* dans la barre de gauche. Un
  noyau ouvert garde sa mémoire GPU, même s'il ne calcule plus rien.
- Avant de lancer un entraînement long, regardez `nvidia-smi` : si la mémoire de
  la carte est presque pleine, attendez ou prévenez l'enseignant.
- Pour retrouver et arrêter un entraînement lancé en arrière-plan :

```bash
ps -u $USER -f | grep python     # repérer le numéro (PID) du processus
kill PID                         # l'arrêter
```

!!! warning "Erreur de mémoire partagée"
    Si un `DataLoader` plante avec un message parlant de *shared memory* ou de
    *bus error*, passez `num_workers=0` dans vos `DataLoader`.

## Notebook ou terminal ?

- Le **notebook** sert à mettre au point : on exécute cellule par cellule, on
  regarde les résultats, on corrige.
- Le **terminal** sert aux entraînements longs : un script lancé avec `nohup`
  continue de tourner même si vous fermez votre navigateur. La
  [partie 2](02_baseline.md#9-lentrainement-long) explique comment faire.
