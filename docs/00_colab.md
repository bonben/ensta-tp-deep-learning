# 0. Google Colab

On travaille sur **Google Colab** : des notebooks dans le navigateur, avec un GPU
gratuit. Rien à installer, il suffit d'un compte Google.

## Démarrer

1. Ouvrez <https://colab.research.google.com> et connectez-vous avec un compte Google.
2. Dans chaque notebook, **passez sur GPU** avant de lancer quoi que ce soit :
   menu *Exécution > Modifier le type d'exécution > GPU T4*
   (en anglais : *Runtime > Change runtime type > T4 GPU*).
3. Vérifiez dans une cellule :

```python
import torch
print(torch.__version__, torch.cuda.is_available())
```

!!! success "Résultat attendu"
    La version de PyTorch, puis `True`. Si vous obtenez `False`, refaites l'étape 2.

## Les données : toujours la copie rapide

PyTorch télécharge normalement CIFAR-10 depuis un serveur universitaire qui est
**extrêmement lent**. Utilisez toujours notre copie. Dans un notebook Colab,
la cellule suivante la télécharge en quelques secondes :

```python
!wget -q https://github.com/bonben/ensta-tp-deep-learning/releases/download/cifar10/cifar-10-python.tar.gz
!mkdir -p data && tar xzf cifar-10-python.tar.gz -C data && rm cifar-10-python.tar.gz
!ls data/cifar-10-batches-py
```

Ensuite, `torchvision.datasets.CIFAR10("./data", download=True)` trouve les
données sur place et ne télécharge plus rien.

Le notebook de la partie 2 fait ce téléchargement tout seul, dans sa première cellule.

## Ce qu'il faut savoir sur Colab

- **Les fichiers disparaissent** quand la session se termine. Téléchargez ce que
  vous voulez garder (vos poids `best.pt`, vos courbes) : panneau *Fichiers* à
  gauche, clic droit, *Télécharger*.
- Une session inactive est **déconnectée** au bout d'un moment : gardez l'onglet
  ouvert pendant un entraînement.
- Le GPU gratuit est **limité en temps** par jour. Ne laissez pas tourner une
  session pour rien : *Exécution > Déconnecter et supprimer l'exécution* quand
  vous avez fini.

!!! note "Et plus tard ?"
    Pour les entraînements plus longs de la suite du module, l'école met à
    disposition un serveur GPU. Votre notebook y fonctionnera tel quel : la
    migration consiste à y déposer le notebook et à le relancer.
