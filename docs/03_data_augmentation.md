# 3. Data augmentation

À faire **après le cours** sur la régularisation et la data augmentation, et à
finir chez vous. Vous repartez de votre notebook de la partie 2.

!!! warning "Partagez le GPU"
    Les expériences de cette partie demandent plusieurs entraînements. Faites-les
    **sur 15 époques**, une à la fois, et lancez les longues dans le terminal
    comme à l'[étape 9](02_baseline.md#9-lentrainement-long).

## 3.1 Voir l'augmentation

Avant de mesurer l'effet d'une augmentation, regardez ce qu'elle fait aux images.

1. Créez un jeu d'entraînement avec la transformation de la partie 2, **sans**
   `Normalize` (sinon les couleurs sont illisibles), et un `DataLoader` de
   8 images **non mélangées** (`shuffle=False`).
2. Affichez ce batch sous forme de grille, et sauvegardez la figure dans `train_DA.png`.
3. Réexécutez la cellule plusieurs fois. Les images sont-elles toujours les
   mêmes ? Pourquoi ?

??? tip "Indice"
    `torchvision.utils.make_grid(images, nrow=4)` assemble un batch en une seule
    image de forme `(3, H, W)`. Pour l'afficher avec `plt.imshow`, les canaux
    doivent être en dernier : `.permute(1, 2, 0)`.

Essayez ensuite d'autres transformations de
[`torchvision.transforms`](https://docs.pytorch.org/vision/stable/transforms.html) :
`ColorJitter`, `RandomRotation`, `RandomErasing` (à placer **après** `ToTensor`),
`TrivialAugmentWide`… Lesquelles gardent une image reconnaissable ?

## 3.2 Avec ou sans augmentation ?

Entraînez deux ResNet18 pendant **15 époques** :

- l'un **sans** data augmentation (la transformation d'entraînement est la même
  que celle de validation) ;
- l'autre **avec** la data augmentation de la partie 2.

Tracez les quatre courbes de précision (train et val, avec et sans) sur un même
graphique.

**Q1.** Sans augmentation, que devient la précision d'entraînement ? Et celle de
validation ? Où voit-on le **surapprentissage** ?

**Q2.** L'augmentation rend-elle l'entraînement plus facile ou plus difficile ?
Pourquoi est-ce une bonne chose ?

## 3.3 Votre propre augmentation

Ajoutez une ou deux transformations de votre choix à la transformation
d'entraînement, et comparez à 15 époques.

**Q3.** Votre augmentation améliore-t-elle la précision de validation ? Si non,
proposez une explication. Pensez à la durée d'entraînement : une augmentation
plus forte demande souvent **plus d'époques** pour porter ses fruits.

## 3.4 Bonus : mixup

Mixup ne transforme pas une image seule : il **mélange deux images** et leurs
étiquettes. Pour chaque batch $(X, y)$ :

1. tirer une permutation aléatoire du batch : `perm = torch.randperm(X.size(0))` ;
2. tirer $\lambda$ uniformément dans $[0, 1]$ ;
3. former le batch mélangé $X_m = \lambda X + (1 - \lambda) X_{perm}$ ;
4. calculer la sortie $\hat{y} = \text{model}(X_m)$ ;
5. calculer la perte
   $\lambda \, \text{CE}(\hat{y}, y) + (1 - \lambda) \, \text{CE}(\hat{y}, y_{perm})$ ;
6. rétropropager et faire le pas d'optimisation, comme d'habitude.

Écrivez une fonction `train_one_epoch_mixup`, sur le modèle de `train_one_epoch`,
et comparez avec votre baseline.

**Q4.** Que signifie la « précision d'entraînement » quand les images sont
mélangées ? Comment la mesureriez-vous ?

??? tip "Pour aller plus loin"
    L'article original : [mixup, Zhang et al. 2017](https://arxiv.org/abs/1710.09412).
    Une variante qui découpe et colle au lieu de mélanger :
    [CutMix, Yun et al. 2019](https://arxiv.org/abs/1905.04899).

---

Votre baseline et vos courbes seront le point de départ des séances de
compression. Gardez `baseline_resnet18.pt` et notez sa précision.
