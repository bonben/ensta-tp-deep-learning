# 4. Le projet fil rouge

## Le projet

!!! abstract "Objectif"
    **Au moins 90 % de précision en validation sur CIFAR-10, avec le réseau le plus petit
    possible** (en nombre de paramètres), en utilisant les techniques du cours.

C'est un projet au long cours. Aujourd'hui, vous disposez de trois familles de techniques :
la **recette d'entraînement** (cours 1), la **régularisation et l'augmentation** (cours 2) et la
**factorisation** (cours 3). Chaque nouveau cours (quantification, élagage, distillation)
ajoutera des leviers, et tous vos essais iront dans le même tableau.

Deux points de repère :

| | paramètres | précision |
|---|---:|---:|
| votre baseline, ResNet18 | 11 173 962 | environ 93 % (30 époques) |
| l'agent de la démonstration | **25 322** | 90,01 % (moyenne de 7 entraînements, 200 époques) |

Entre les deux, il y a un facteur 440. Le chemin est à vous.

## La boîte à outils

**[Ouvrir la boîte à outils dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/fil_rouge.ipynb){ .md-button .md-button--primary }**

Puis *Fichier > Enregistrer une copie dans Drive*, et *GPU T4*.

Ce notebook n'a pas de trous : il **fonctionne tel quel** et entraîne votre baseline. Il fournit :

- `count_params(model)` et `count_macs(model)` : la taille et le calcul d'un réseau ;
- `fit(model, epochs, lr=0.1, weight_decay=5e-4, batch_size=128, label_smoothing=0.0, transform=..., seed=0, name=...)` :
  la recette de la baseline, dont **tous les réglages sont des arguments**. À la fin, il
  affiche une ligne à recopier dans le tableau partagé ;
- `plot(h1, h2, ...)` : superpose les courbes de validation de plusieurs essais.

Le modèle se construit avec `resnet.py` : `ResNet(BasicBlock, [2, 2, 2, 2], width=64)` est
votre ResNet18. `[2, 2, 2, 2]` est le nombre de blocs dans chacun des 4 étages, et `width`
le nombre de FM du premier étage (puis 2×, 4×, 8×).

## Comment travailler

Faites comme un ingénieur, et comme l'agent quand il a bien travaillé :

1. **Choisissez une piste** ci-dessous, et une **question** précise.
2. **Prédisez** avant de lancer : qu'attendez-vous ? Écrivez-le.
3. **Changez une seule chose à la fois**, par rapport à un essai de référence.
4. **Comparez à budget égal** : le même nombre d'époques. 15 époques pour explorer, plus
   pour confirmer.
5. **Notez tout** dans le tableau partagé, y compris les essais décevants : paramètres, MAC,
   époques, précision **au dernier epoch**.
6. **Décidez** : garder ou jeter, et pourquoi. À moins d'un demi-point du seuil, relancez avec
   une autre graine (`seed=1`) avant de conclure.

!!! tip "Lire ses courbes pour décider"
    - **Entraînement bien au-dessus de la validation** : le réseau apprend par cœur. Régularisez
      plus (cours 2), ou il est plus gros que nécessaire : rétrécissez-le.
    - **Entraînement et validation proches, et trop bas** : le réseau est trop petit ou
      sous-entraîné. Plus d'époques, moins de régularisation, ou un peu plus de capacité.
    - **Précision qui monte encore à la dernière époque** : vous n'avez pas fini d'entraîner.

Combien de temps dure un essai ? La durée d'une époque s'affiche dès la première : multipliez
par le nombre d'époques **avant** de lancer un long entraînement.

---

## Les pistes

Choisissez-en une. Elles se combinent : une fois votre question tranchée, gardez ce qui
marche et passez à une autre.

### Piste 1 : la recette d'entraînement

> *Avec un réseau fixé, combien la recette peut-elle lui faire gagner ?*

**Leviers** : les arguments de `fit` : `epochs`, `lr`, `batch_size`, `weight_decay`,
`label_smoothing`.

**Par où commencer** : prenez un petit réseau qui n'atteint pas 90 %, par exemple
`ResNet(BasicBlock, [1, 1, 1, 1], width=16)` (308 826 paramètres, 86,1 % en 15 époques), et faites varier **le nombre d'époques**
(15, 30, 60), puis **le taux d'apprentissage** (0,05, 0,1, 0,2).

**Ce qu'on sait déjà** : chez l'agent, passer de 50 à 200 époques a rapporté environ 2 points
sur les petits réseaux, et au-delà plus rien. Le taux d'apprentissage est le réglage qui compte
le plus. Le reste (taille de batch, weight decay, label smoothing) joue souvent moins que le
bruit d'une graine à l'autre.

**Piège** : un taux trop fort fait diverger l'entraînement (la perte devient `nan`).

### Piste 2 : factoriser les convolutions (cours 3)

> *Combien de paramètres gagne-t-on en factorisant, et combien de précision perd-on ?*

**Leviers** : la convolution séparable en profondeur ; les convolutions 1×1 groupées.

**Par où commencer** : remplacez chaque convolution 3×3 des blocs par une convolution
séparable. Deux classes suffisent :

```python
class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 3×3 par FM : groups=in_ch, chaque filtre ne voit qu'une FM
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        # 1×1 : mélange les FM
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SepBasicBlock(BasicBlock):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__(in_planes, planes, stride)
        self.conv1 = SeparableConv2d(in_planes, planes, stride)
        self.conv2 = SeparableConv2d(planes, planes, 1)

model = ResNet(SepBasicBlock, [2, 2, 2, 2], width=64)
```

**Ce qu'on sait déjà** (15 époques, une graine) :

| | paramètres | MAC | précision |
|---|---:|---:|---:|
| ResNet18 | 11 173 962 | 555 M | 90,6 % |
| ResNet18 factorisé | 1 439 626 | 73 M | 90,2 % |
| factorisé, `width=32` | 371 658 | 20 M | 88,9 % |
| factorisé, `width=16` | 98 794 | 5,7 M | 84,4 % |

Après factorisation, 85 % des paramètres sont dans les convolutions 1×1 : c'est là qu'il faut
chercher la suite.

**Pour aller plus loin** : ajoutez `groups=g` à la convolution 1×1 (paramètres divisés par
$g$ ; il faut que $C_{in}$ et $C_{out}$ soient divisibles par $g$). Ou supprimez les
convolutions 1×1 des raccourcis, qui pèsent 12 % du réseau factorisé.

### Piste 3 : mettre le réseau à l'échelle

> *Pour diviser la taille par 10, vaut-il mieux rétrécir le réseau ou le raccourcir ?*

C'est la question de l'article EfficientNet (article 4).

**Leviers** : la largeur `width` (64, 32, 16, 8) et la profondeur (`[2, 2, 2, 2]`,
`[1, 1, 1, 1]`, `[2, 2, 1, 1]`…).

**Par où commencer** : une petite grille, par exemple `width` ∈ {32, 16} × profondeur ∈
{`[2, 2, 2, 2]`, `[1, 1, 1, 1]`}, soit 4 essais de 15 époques. Tracez la précision en
fonction du nombre de paramètres, en **échelle logarithmique** : `plt.xscale("log")`.

**Ce qu'on sait déjà** : diviser `width` par 2 divise les paramètres par 4 environ. Un ResNet18
non factorisé à `width=16` a 701 466 paramètres et fait 88,0 % en 15 époques. Chez l'agent, la
précision croît à peu près comme le logarithme de la taille : chaque point de précision coûte
de plus en plus cher.

**Une remarque** : les étages n'ont pas forcément besoin d'être aussi larges les uns que les
autres. L'agent a gagné ses derniers paramètres en **élargissant le premier étage et en
rétrécissant le dernier**. Pour l'essayer, il faut modifier `resnet.py`.

### Piste 4 : régulariser et augmenter (cours 2)

> *Un petit réseau a-t-il besoin d'autant de régularisation qu'un gros ?*

**Leviers** : l'argument `transform` de `fit` (ajoutez `v2.ColorJitter`, `v2.RandomErasing`,
`v2.TrivialAugmentWide`…, voir la partie 3), `label_smoothing`, `weight_decay`, mixup (partie 3).

**Par où commencer** : prenez **deux** réseaux de tailles très différentes (le ResNet18 et un
réseau 50 fois plus petit), et comparez chacun avec et sans une augmentation plus forte.

```python
fort = v2.Compose([v2.RandomCrop(32, padding=4), v2.RandomHorizontalFlip(),
                   v2.TrivialAugmentWide()] + NORMALIZE)
h = fit(model, epochs=15, transform=fort, name="TrivialAugment")
```

**Ce qu'on sait déjà** : sur le ResNet18, l'augmentation de base fait gagner près de 5 points
(partie 3). Mais les petits réseaux de l'agent n'arrivaient même pas à apprendre parfaitement
les images d'entraînement : chez eux, une régularisation supplémentaire (la distillation, en
l'occurrence) a **fait perdre** de 0,7 à 1,7 point. Regardez l'écart entre entraînement et
validation pour savoir dans quel cas vous êtes.

### Piste 5 (pour les plus à l'aise) : changer l'architecture

> *Peut-on faire mieux qu'un ResNet18 rétréci ?*

Le réseau de l'agent n'est pas un ResNet18 : il n'a que **3 étages** (résolutions 32, 16 et
8), des **raccourcis sans aucun paramètre** et l'activation **SiLU**. Lisez son code
(`train.py`, classes `DSBlock` et `PadShort`), reprenez une idée à la fois dans `resnet.py`,
et mesurez ce que chacune apporte.

---

Le projet continue aux séances suivantes : gardez votre notebook (il est dans votre Drive)
et continuez à remplir le tableau.
