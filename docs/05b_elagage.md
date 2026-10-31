# 5b. Élagage

> *Combien de poids de votre réseau peut-on mettre à zéro, et à quel prix ?*

## L'amorce

**[Ouvrir l'amorce dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/amorce_elagage.ipynb){ .md-button .md-button--primary }**

Puis *Fichier > Enregistrer une copie dans Drive*, et *GPU T4*.

Le notebook reprend la boîte à outils du fil rouge et y ajoute :

- `prune_global(model, amount)` : met à zéro la fraction `amount` des poids de **plus petite
  valeur absolue**, en comparant tous les poids du réseau entre eux (élagage global, non
  structuré, avec `torch.nn.utils.prune`) ;
- `count_nonzero(model)` : le nombre de paramètres non nuls ;
- `make_permanent(model)` : écrit les zéros dans les poids et retire les masques, à faire avant
  de sauvegarder.

```python
pruned = copy.deepcopy(model)          # on garde l'original intact
prune_global(pruned, 0.7)              # 70 % des poids à zéro
h = fit(pruned, epochs=5, lr=0.01)     # réentraîner un peu, le masque reste en place
make_permanent(pruned)
print(count_nonzero(pruned), evaluate(pruned))
```

PyTorch ne supprime pas les poids : il garde le poids d'origine et un **masque** de 0 et de 1,
et le réseau utilise leur produit. Pendant le réentraînement, les poids élagués restent à zéro.

Partez de **votre réseau entraîné** : rechargez ses poids dans la cellule « Votre réseau de
départ ».

## Repères mesurés

RTX 5090, réseau de démonstration : ResNet18 factorisé, `width=32`, 15 époques
(371 658 paramètres, 89,1 %).

| Poids élagués | Paramètres non nuls | Sans réentraîner | + 5 époques (`lr=0.01`) |
|---:|---:|---:|---:|
| 0 % | 371 658 | 89,1 % | |
| 30 % | 261 604 | 88,9 % | |
| 50 % | 188 234 | 88,0 % | 89,1 % |
| 70 % | 114 864 | 79,8 % | 88,6 % |
| 80 % | 78 180 | 49,5 % | |
| 90 % | 41 495 | 15,6 % | 86,5 % |

En plusieurs étapes (élaguer 50 % des poids restants, réentraîner 3 époques, trois fois) :
50 666 paramètres non nuls, 87,3 %.

Pour comparer : le même réseau en `width=16` a 98 794 paramètres et fait 84,9 %. Élaguer un
réseau plus large a fait mieux que le rétrécir d'avance.

## Pistes

- **Jusqu'où avec un réentraînement ?** Tracez la courbe précision / paramètres non nuls de
  votre réseau, avec et sans réentraînement.
- **En une fois ou en plusieurs étapes**, à nombre total d'époques égal ?
- **Quelles couches sont sensibles ?** Élaguez une couche à la fois
  (`prune.l1_unstructured(module, "weight", amount=0.5)`) et mesurez. Faut-il épargner la
  première convolution, la couche finale, les convolutions par FM (`depthwise`) ?
- **Global ou local** : la même fraction dans chaque couche, est-ce mieux ou pire ?
- **Structuré** : `prune.ln_structured(module, "weight", amount, n=2, dim=0)` retire des
  filtres entiers. C'est le seul élagage qui rend la couche vraiment plus petite. À quel prix ?

## Pièges

- **Des zéros ne sont pas un gain.** Une matrice dense à 90 % de zéros prend autant de place
  et de temps qu'avant. Pour en profiter, il faut un format creux (chaque poids gardé coûte
  alors sa valeur **et** son indice) ou un élagage structuré. Dites ce que vous comptez.
- `count_params` compte toujours tous les paramètres : c'est `count_nonzero` qui mesure
  l'élagage.
- Sans `copy.deepcopy`, vous élaguez votre réseau de départ, et il n'y a plus de référence.

## À rapporter

Fraction élaguée, paramètres non nuls, époques de réentraînement et `lr`, précision au dernier
epoch. Pour la taille mémoire, dites quelle hypothèse vous faites (dense, ou creux avec indices).
