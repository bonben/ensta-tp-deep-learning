# 4. Factoriser son réseau

Après le cours sur la factorisation, on la met en pratique sur votre ResNet18, puis on
lance le **fil rouge** du module : le plus petit réseau possible à 90 % sur CIFAR-10.

**[Ouvrir le notebook dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/factorisation.ipynb){ .md-button .md-button--primary }**

Puis, tout de suite : *Fichier > Enregistrer une copie dans Drive*, et passez sur GPU
(*Exécution > Modifier le type d'exécution > GPU T4*).

Le notebook est autonome : il télécharge les données et `resnet.py`, et il **fournit**
les fonctions d'entraînement (c'est le corrigé de la partie 2). Il fournit aussi deux
outils de mesure :

- `count_params(model)` : le nombre de paramètres ;
- `count_macs(model)` : le nombre de multiplications-additions (MAC) pour classer **une** image.

## 1. Compter à la main

Calculez, pour 64 input FM et 64 output FM, le nombre de paramètres d'une convolution 3×3
pleine, puis d'une convolution séparable (3×3 par FM, puis 1×1).

??? success "Résultat attendu"
    Pleine : $9 \times 64 \times 64 = 36\,864$. Séparable : $9 \times 64 + 64 \times 64 = 576 + 4\,096 = 4\,672$,
    soit **7,9 fois moins**.

## 2. Une convolution séparable

Vous écrivez une classe `SeparableConv2d` avec deux `nn.Conv2d`. Toute l'astuce tient dans
le paramètre `groups` : avec `groups=C_in`, chaque filtre ne regarde **qu'une seule** FM.

## 3. Un bloc résiduel factorisé

Vous créez `SepBasicBlock`, qui hérite du `BasicBlock` de `resnet.py` et remplace ses deux
convolutions 3×3 par des `SeparableConv2d`. Deux lignes suffisent.

??? success "Résultat attendu"
    | | paramètres | MAC par image |
    |---|---:|---:|
    | ResNet18 | 11 173 962 | 555 M |
    | ResNet18 factorisé | 1 439 626 | 73 M |
    | rapport | 7,8 | 7,6 |

**Q1.** Où sont les paramètres qui n'ont pas été factorisés ?

??? success "Réponse"
    Dans le réseau factorisé, les paramètres se répartissent ainsi :

    | | paramètres | part |
    |---|---:|---:|
    | convolutions 1×1 des blocs | 1 220 608 | 85 % |
    | convolutions 1×1 des raccourcis (changements d'étage) | 172 032 | 12 % |
    | convolutions 3×3 par FM | 30 528 | 2 % |
    | BatchNorm, première convolution, couche finale | 16 458 | 1 % |

    Les convolutions 3×3, celles qui regardent le voisinage, ne coûtent presque plus rien :
    **c'est le mélange des FM (les 1×1) qui coûte**. C'est là qu'il faut chercher les
    prochaines économies.

## 4. Entraîner le réseau factorisé

15 époques, même recette que la baseline.

**Q2.** Combien de points perdez-vous pour un réseau près de 8 fois plus petit ?

**Q3.** Une époque est-elle 8 fois plus rapide ? Pourquoi ?

??? success "Résultat attendu"
    Mesuré sur une RTX 5090, 15 époques, une seule graine :

    | | paramètres | MAC | précision (dernier epoch) | inférence (256 images) |
    |---|---:|---:|---:|---:|
    | ResNet18 | 11,2 M | 555 M | 90,6 % | 5,7 ms |
    | ResNet18 factorisé | 1,4 M | 73 M | 90,2 % | 4,0 ms |

    **Q2** : 0,4 point, à peine plus que le bruit d'un entraînement à l'autre.

    **Q3** : non. 7,6 fois moins de calcul, mais seulement 1,4 fois plus rapide en
    inférence. Une convolution par FM fait très peu de calcul pour chaque valeur qu'elle
    lit en mémoire : le GPU passe son temps à **attendre la mémoire**, pas à calculer.
    Moins d'opérations ne veut pas dire plus rapide (souvenez-vous du cours d'architecture).
    Sur Colab, la durée d'une époque est en plus limitée par le chargement des images.

## 5. Le fil rouge : votre plus petit réseau

**Objectif du module : le moins de paramètres possible, avec au moins 90 % en validation.**

Vos leviers, à combiner :

| Levier | Dans le code | Effet sur les paramètres |
|---|---|---|
| la factorisation | `SepBasicBlock` au lieu de `BasicBlock` | environ ÷ 8 |
| la largeur | `width=64` (défaut), 32, 16… | environ ÷ 4 quand on divise par 2 |
| la profondeur | `[2, 2, 2, 2]`, `[1, 1, 1, 1]`… | moins de blocs par étage |
| la durée d'entraînement | `fit(model, epochs=...)` | aucun, mais les petits réseaux progressent longtemps |

??? tip "Repères mesurés (15 époques)"
    | | paramètres | MAC | précision |
    |---|---:|---:|---:|
    | ResNet18 | 11 173 962 | 555 M | 90,6 % |
    | factorisé | 1 439 626 | 73 M | 90,2 % |
    | factorisé, `width=32` | 371 658 | 20 M | 88,9 % |
    | **non** factorisé, `width=16` | 701 466 | 35 M | 88,0 % |
    | factorisé, `width=16` | 98 794 | 5,7 M | 84,4 % |

    À nombre de paramètres comparable, **factoriser vaut mieux que rétrécir** : 372 000
    paramètres factorisés font mieux que 701 000 non factorisés.

### Les règles de la mesure

Pour que les résultats de la promo soient comparables :

1. on note la précision **au dernier epoch**, pas la meilleure ;
2. on note le **nombre d'époques** ;
3. on note les **paramètres et les MAC** (`count_params`, `count_macs`) ;
4. on reporte **chaque** essai dans le tableau partagé, y compris ceux qui sont décevants.

!!! warning "Le piège de la validation"
    En choisissant votre meilleur réseau sur la validation, puis en annonçant sa précision
    sur cette même validation, vous surestimez un peu votre score : vous gardez les essais
    « chanceux ». À 0,1 point du seuil, relancez le même réseau avec une autre graine
    (`torch.manual_seed(1)` avant de le créer) avant de crier victoire.

### La référence à battre

L'agent lancé en démonstration a trouvé un réseau de **25 322 paramètres** à **90,01 %**
(moyenne sur 7 entraînements), avec 200 époques. Son architecture : des convolutions
séparables, comme les vôtres, et des raccourcis sans paramètre. Son rapport et son code
sont mis à votre disposition.

??? tip "Pour aller plus loin"
    - **Convolutions 1×1 groupées** : dans `SeparableConv2d`, ajoutez `groups=g` à la
      convolution 1×1 (avec $C_{in}$ et $C_{out}$ divisibles par $g$). Ses paramètres sont
      divisés par $g$. C'est l'idée de ShuffleNet, qui mélange ensuite les FM entre les groupes.
    - **Entraîner plus longtemps** : l'agent a gagné environ 2 points en passant de 50 à
      200 époques sur ses petits réseaux.
    - **Supprimer les raccourcis 1×1** : ils pèsent 12 % du réseau factorisé.
