# 5a. Distillation

> *Un teacher fait-il gagner de la précision à votre réseau, ou lui permet-il de rétrécir ?*

## L'amorce

**[Ouvrir l'amorce dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/amorce_distillation.ipynb){ .md-button .md-button--primary }**

Puis *Fichier > Enregistrer une copie dans Drive*, et *GPU T4*.

Le notebook reprend la boîte à outils du fil rouge et y ajoute :

- un **teacher** déjà entraîné, téléchargé : un ResNet18 `width=32` (2,8 M paramètres,
  94,9 %, 200 époques) ;
- `kd_loss(teacher, T=4.0, alpha=0.9)` : la perte de distillation de Hinton, à passer à `fit`
  par l'argument `loss_fn`.

```python
student = ResNet(SepBasicBlock, [2, 2, 2, 2], width=32)
h_kd = fit(student, epochs=15, loss_fn=kd_loss(teacher, T=4.0, alpha=0.9))
```

Le student apprend à reproduire les **soft labels** du teacher (softmax à température
`T`) et, pour une part `1 - alpha`, les vrais labels. **Seule la perte change** : même
architecture, même recette, même nombre d'époques que votre réseau de départ. C'est ce qui
rend la comparaison honnête.

Remplacez le réseau de démonstration par le vôtre dans la cellule « Votre réseau de départ ».

## Repères mesurés

RTX 5090, 15 époques, `T = 4`, `alpha = 0.9`.

Student : ResNet18 factorisé, `width=32` (371 658 paramètres), avec le teacher de l'amorce.

| Seed | Sans distillation | Avec distillation | Gain |
|---:|---:|---:|---:|
| 0 | 89,1 % | 89,7 % | +0,6 |
| 1 | 88,8 % | 89,8 % | +1,0 |
| 2 | 88,5 % | 89,5 % | +1,0 |
| 3 | 89,0 % | 90,0 % | +1,1 |
| 4 | 89,1 % | 89,9 % | +0,8 |
| 5 | 88,5 % | 89,3 % | +0,8 |
| **moyenne** | **88,8 %** | **89,7 %** | **+0,9** |

Le student `width=16` (98 794 paramètres) : 84,9 % sans, 85,6 % avec (une seed).

D'une seed à l'autre, la précision varie de 0,6 point, autant que le gain. Ce qui permet de
conclure, c'est que le gain est là **à chaque seed**. **Ne concluez pas sur un seul essai.**

Chaque époque coûte un peu plus cher que sans distillation, car le teacher calcule aussi.

**Un meilleur teacher ne fait pas un meilleur student.** Trois teachers, même student,
moyenne de 6 seeds :

| Teacher | Paramètres | Sa précision | Le student distillé |
|---|---:|---:|---:|
| aucun | | | 88,8 % |
| ResNet18, 30 époques | 11,2 M | 93,4 % | 89,4 % |
| ResNet18, 200 époques | 11,2 M | **96,5 %** | 89,5 % |
| ResNet18 `width=32`, 200 époques (celui de l'amorce) | 2,8 M | 94,9 % | **89,7 %** |

Trois points de plus chez le teacher ne rapportent rien au student. Le petit teacher fait un
peu mieux, mais l'écart (0,3 point) reste à la limite du bruit. Le gain du student n'est pas
mécaniquement lié à la précision du teacher.

## Pistes

Une question à la fois, toujours comparée à votre réseau entraîné **sans** distillation, au même
nombre d'époques.

- **La température** : `T` = 1, 2, 4, 8. À `T = 1`, les sorties du teacher sont presque
  aussi tranchées que les labels : que reste-t-il du gain ?
- **Le dosage** `alpha` : 0,5, 0,9, 1 (le student n'apprend plus que du teacher).
- **La taille du student** : le gain est-il plus grand pour un petit student ? La distillation
  permet-elle de garder la précision de votre réseau avec un student plus petit ?
- **Plus d'époques** : l'écart avec l'entraînement normal se maintient-il à 30 ou 60 époques ?
- **Un autre teacher** : votre meilleur réseau du fil rouge, ou un teacher plus petit.
  Le tableau ci-dessus montre qu'un teacher plus précis ne fait pas forcément un meilleur student.

## Pièges

- **Comparer à budget égal** : un student distillé en 30 époques contre une référence de 15, ce
  n'est pas la distillation que vous mesurez.
- L'agent de la démonstration avait trouvé que la distillation **faisait perdre** de 0,7 à
  1,7 point à ses très petits réseaux. La distillation régularise : un réseau qui n'arrive déjà
  pas à apprendre par cœur peut en souffrir. Regardez vos courbes d'entraînement.

## À rapporter

Paramètres, `T`, `alpha`, époques, seed, précision au dernier epoch, avec et sans distillation.
La taille mémoire ne change pas : votre gain se lit en **précision à taille égale**, ou en
**taille à précision égale**.
