# 1. Prise en main de PyTorch

On part du tutoriel officiel de PyTorch, court et complet : il entraîne un petit
réseau convolutif sur **CIFAR-10**, le jeu de données de tout le module
(60 000 images couleur de 32 × 32 pixels, réparties en 10 classes).

**Tutoriel : [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)**

## Mise en place

1. En haut de la page du tutoriel, cliquez sur **Run in Google Colab**, puis
   *Fichier > Enregistrer une copie dans Drive*.
2. Passez sur GPU : *Exécution > Modifier le type d'exécution > GPU T4*.
3. **Avant tout le reste**, ajoutez une cellule tout en haut avec le
   téléchargement rapide des données (voir la [partie 0](00_colab.md#les-donnees-toujours-la-copie-rapide))
   et exécutez-la. Sinon, le tutoriel va chercher les données sur un serveur
   très lent et reste bloqué.
4. Exécutez ensuite le tutoriel **cellule par cellule**, en lisant le texte entre
   les cellules. Ne lancez pas tout d'un coup : l'intérêt est de comprendre
   chaque étape.

## Questions

Répondez dans des cellules Markdown de votre notebook. Ce sont exactement les
notions dont vous aurez besoin pour la partie 2.

**Q1. Les données.** Quelle est la forme (`.shape`) d'un batch d'images ? Que
représente chacune des quatre dimensions ? Et la forme du batch d'étiquettes ?

**Q2. Le réseau.** D'où vient le `16 * 5 * 5` de la couche `fc1` ? Suivez la
taille d'une image de 32 × 32 à travers `conv1`, `pool`, `conv2`, `pool`.

??? tip "Indice"
    Une convolution de noyau 5 × 5 sans *padding* fait perdre 4 pixels à chaque
    dimension. Un `MaxPool2d(2, 2)` divise chaque dimension par 2.

**Q3. Les paramètres.** Combien ce réseau a-t-il de paramètres ? Calculez-le de
deux façons : à la main, couche par couche, puis avec PyTorch.

??? tip "Indice"
    `net.parameters()` renvoie tous les tenseurs de poids et de biais ; `p.numel()`
    donne le nombre d'éléments d'un tenseur `p`.

!!! success "Résultat attendu"
    62 006 paramètres. Retenez l'ordre de grandeur : le ResNet18 de la partie 2
    en a 180 fois plus.

**Q4. L'apprentissage.** Que font les trois lignes `optimizer.zero_grad()`,
`loss.backward()` et `optimizer.step()` ? Que se passe-t-il si on supprime la
première ? Essayez.

**Q5. Le résultat.** Quelle précision obtenez-vous après 2 époques ? Quelles
classes sont les mieux et les moins bien reconnues ? Pourquoi, d'après vous ?

!!! success "Résultat attendu"
    Autour de 50 à 55 %. Le hasard donnerait 10 %.

## Passer sur GPU

Suivez la section *Training on GPU* du tutoriel : envoyez le réseau **et** chaque
batch sur le GPU avec `.to(device)`, puis relancez l'entraînement.

**Q6.** Mesurez la durée d'une époque sur CPU puis sur GPU (le module `time`
suffit). Le GPU est-il beaucoup plus rapide ? Pourquoi ?

??? tip "Indice"
    Regardez la taille des batchs (`batch_size = 4`) et la taille du réseau. Un
    GPU est rentable quand il a beaucoup de calculs à faire en parallèle.
    Recommencez avec `batch_size = 128` : qu'est-ce qui change ?

## Pour aller plus loin

Améliorez la précision de ce petit réseau : plus d'époques, plus de filtres
(c'est l'exercice proposé à la fin du tutoriel), un autre taux d'apprentissage.
Pour chaque essai, notez **le nombre de paramètres et la précision obtenue**.
Ce compromis entre taille et précision est le sujet de tout le module.

---

Quand vous avez répondu aux questions Q1 à Q5, passez à la
[partie 2](02_baseline.md).
