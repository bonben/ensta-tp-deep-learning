# 5c. Quantification

> *Sur combien de bits peut-on écrire les poids de votre réseau ?*

## L'amorce

**[Ouvrir l'amorce dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/amorce_quantification.ipynb){ .md-button .md-button--primary }**

Puis *Fichier > Enregistrer une copie dans Drive*, et *GPU T4*.

Le notebook reprend la boîte à outils du fil rouge et y ajoute :

- `quantize(w, bits)` : la quantification uniforme symétrique, avec une échelle par filtre ;
- `quantize_model(model, bits)` : une **copie** du réseau dont les poids des convolutions et de
  la couche linéaire sont quantifiés ;
- `model_size_kb(model, bits)` : la taille mémoire, poids sur `bits` bits, BatchNorm et biais
  en 32 bits.

```python
q = quantize_model(model, 4)           # une copie, poids sur 4 bits
print(model_size_kb(q, 4), evaluate(q))
h = fit(q, epochs=5, lr=0.01)          # fine-tuner en quantifiant
```

Sur $b$ bits, chaque poids $w$ d'un filtre est remplacé par $s \cdot \mathrm{round}(w / s)$,
avec $s = \max |w| / (2^{b-1} - 1)$ : il ne prend plus que $2^b - 1$ valeurs.

Pour fine-tuner, l'arrondi pose un problème : sa dérivée est nulle presque partout. L'astuce
(*straight-through estimator*) : dans le calcul, les poids sont quantifiés, mais le gradient
traverse l'arrondi comme s'il n'existait pas, et met à jour une copie des poids en flottant.
Une ligne dans le notebook :

```python
return w + (quantize(w, self.bits) - w).detach()
```

Partez de **votre réseau entraîné** : rechargez ses poids dans la cellule « Votre réseau de
départ ».

## Repères mesurés

RTX 5090, réseau de démonstration : ResNet18 factorisé, `width=32`, 15 époques
(371 658 paramètres, 89,1 %).

| Bits par poids | Taille | Sans fine-tuning | + 5 époques (`lr=0.01`) |
|---:|---:|---:|---:|
| 32 | 1 452 Ko | 89,1 % | |
| 8 | 377 Ko | 89,2 % | |
| 6 | 287 Ko | 88,9 % | |
| 4 | 198 Ko | 87,1 % | 89,1 % |
| 3 | 153 Ko | 74,4 % | 87,9 % |
| 2 | 108 Ko | 11,8 % | 78,1 % |

À 2 bits, les poids sont ternaires : $-s$, $0$ ou $+s$.

## Pistes

- **Jusqu'où avec un fine-tuning ?** Plus d'époques à 2 bits ? Un autre `lr` ?
- **Précision mixte** : quelles couches sont les plus sensibles ? Gardez la première et la
  dernière sur 8 bits, le reste sur 2 ou 3. Il faut alors appliquer `FakeQuant` couche par
  couche, avec `parametrize.register_parametrization(module, "weight", FakeQuant(bits))`.
- **Une échelle par couche** au lieu d'une par filtre : dans `quantize`, remplacez le `max` par
  filtre par un `max` sur tout le tenseur. Qu'est-ce que ça change ?
- **Les activations** restent en flottant. Pour calculer en entiers, il faudrait les quantifier
  aussi : essayez sur la sortie de chaque bloc. Combien perd-on ?
- **Combiner** avec le pruning (partie 5b) : un poids nul ne coûte rien à quantifier.

## Pièges

- Ici on **simule** la quantification : les calculs restent en flottant, le réseau ne va pas
  plus vite. Le gain mesuré est un gain de **mémoire**. Pour gagner en temps et en énergie, il
  faut des activations en entiers et des kernels en entiers (TensorRT, TFLite,
  `torch.ao`).
- `quantize_model` fait une copie : pensez à évaluer et entraîner **la copie**.
- `model_size_kb` ne compte que les couches qu'il connaît (convolutions, linéaire, BatchNorm2d).
  Si votre réseau a d'autres couches à paramètres, vérifiez.

## À rapporter

Bits par couche, taille en Ko, époques de fine-tuning et `lr`, précision au dernier epoch.
