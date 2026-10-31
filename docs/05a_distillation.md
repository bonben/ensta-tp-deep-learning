# 5a. Distillation

> *Un enseignant fait-il gagner de la précision à votre réseau, ou lui permet-il de rétrécir ?*

## L'amorce

**[Ouvrir l'amorce dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/amorce_distillation.ipynb){ .md-button .md-button--primary }**

Puis *Fichier > Enregistrer une copie dans Drive*, et *GPU T4*.

Le notebook reprend la boîte à outils du fil rouge et y ajoute :

- un **enseignant** déjà entraîné, téléchargé : un ResNet18 de largeur 32 (2,8 M paramètres,
  94,9 %, 200 époques) ;
- `kd_loss(teacher, T=4.0, alpha=0.9)` : la perte de distillation de Hinton, à passer à `fit`
  par l'argument `loss_fn`.

```python
student = ResNet(SepBasicBlock, [2, 2, 2, 2], width=32)
h_kd = fit(student, epochs=15, loss_fn=kd_loss(teacher, T=4.0, alpha=0.9))
```

L'élève apprend à reproduire les **sorties adoucies** de l'enseignant (softmax à température
`T`) et, pour une part `1 - alpha`, les vraies étiquettes. **Seule la perte change** : même
architecture, même recette, même nombre d'époques que votre réseau de départ. C'est ce qui
rend la comparaison honnête.

Remplacez le réseau de démonstration par le vôtre dans la cellule « Votre réseau de départ ».

## Repères mesurés

RTX 5090, 15 époques, `T = 4`, `alpha = 0.9`.

Élève : ResNet18 factorisé, `width=32` (371 658 paramètres), avec l'enseignant de l'amorce.

| Graine | Sans distillation | Avec distillation | Gain |
|---:|---:|---:|---:|
| 0 | 89,1 % | 89,7 % | +0,6 |
| 1 | 88,8 % | 89,8 % | +1,0 |
| 2 | 88,5 % | 89,5 % | +1,0 |
| 3 | 89,0 % | 90,0 % | +1,1 |
| 4 | 89,1 % | 89,9 % | +0,8 |
| 5 | 88,5 % | 89,3 % | +0,8 |
| **moyenne** | **88,8 %** | **89,7 %** | **+0,9** |

L'élève `width=16` (98 794 paramètres) : 84,9 % sans, 85,6 % avec (une graine).

D'une graine à l'autre, la précision varie de 0,6 point, autant que le gain. Ce qui permet de
conclure, c'est que le gain est là **à chaque graine**. **Ne concluez pas sur un seul essai.**

Chaque époque coûte un peu plus cher que sans distillation, car l'enseignant calcule aussi.

**Un meilleur enseignant ne fait pas un meilleur élève.** Trois enseignants, même élève,
moyenne de 6 graines :

| Enseignant | Paramètres | Sa précision | L'élève distillé |
|---|---:|---:|---:|
| aucun | | | 88,8 % |
| ResNet18, 30 époques | 11,2 M | 93,4 % | 89,4 % |
| ResNet18, 200 époques | 11,2 M | **96,5 %** | 89,5 % |
| ResNet18 `width=32`, 200 époques (celui de l'amorce) | 2,8 M | 94,9 % | **89,7 %** |

Trois points de plus chez l'enseignant ne rapportent rien à l'élève. Le petit enseignant fait un
peu mieux, mais l'écart (0,3 point) reste à la limite du bruit. Le gain de l'élève n'est pas
mécaniquement lié à la précision de l'enseignant.

## Pistes

Une question à la fois, toujours comparée à votre réseau entraîné **sans** distillation, au même
nombre d'époques.

- **La température** : `T` = 1, 2, 4, 8. À `T = 1`, les sorties de l'enseignant sont presque
  aussi tranchées que les étiquettes : que reste-t-il du gain ?
- **Le dosage** `alpha` : 0,5, 0,9, 1 (l'élève n'apprend plus que de l'enseignant).
- **La taille de l'élève** : le gain est-il plus grand pour un petit élève ? La distillation
  permet-elle de garder la précision de votre réseau avec un élève plus petit ?
- **Plus d'époques** : l'écart avec l'entraînement normal se maintient-il à 30 ou 60 époques ?
- **Un autre enseignant** : votre meilleur réseau du fil rouge, ou un enseignant plus petit.
  Le tableau ci-dessus montre qu'un enseignant plus précis ne fait pas forcément un meilleur élève.

## Pièges

- **Comparer à budget égal** : un élève distillé en 30 époques contre une référence de 15, ce
  n'est pas la distillation que vous mesurez.
- L'agent de la démonstration avait trouvé que la distillation **faisait perdre** de 0,7 à
  1,7 point à ses très petits réseaux. La distillation régularise : un réseau qui n'arrive déjà
  pas à apprendre par cœur peut en souffrir. Regardez vos courbes d'entraînement.

## À rapporter

Paramètres, `T`, `alpha`, époques, graine, précision au dernier epoch, avec et sans distillation.
La taille mémoire ne change pas : votre gain se lit en **précision à taille égale**, ou en
**taille à précision égale**.
