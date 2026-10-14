# 2. Votre baseline ResNet18

Vous allez entraîner un **ResNet18** sur CIFAR-10. C'est le réseau que vous
compresserez pendant tout le reste du module.

**[Ouvrir le notebook dans Colab](https://colab.research.google.com/github/bonben/ensta-tp-deep-learning/blob/main/docs/code/baseline.ipynb){ .md-button .md-button--primary }**

Puis, tout de suite : *Fichier > Enregistrer une copie dans Drive*, pour
travailler sur **votre** copie. Et passez sur GPU (*Exécution > Modifier le type
d'exécution > GPU T4*).

- Le notebook est **à trous** : chaque cellule `TODO` est à écrire par vous.
  Juste après, une cellule **Vérification** vous dit si votre code est juste :
  elle doit afficher `OK`.
- Sa première cellule télécharge les données et [`resnet.py`](code/resnet.py),
  la **description du réseau**. Ouvrez ce fichier (panneau *Fichiers* à gauche)
  et lisez-le : vous y retrouverez les blocs résiduels vus en cours.

Ce que vous écrivez ici ressemble beaucoup au tutoriel de la
[partie 1](01_pytorch.md). Gardez-le ouvert à côté : c'est votre meilleure aide.

!!! info "Jeux de données"
    Les **50 000 images d'entraînement** de CIFAR-10 servent à l'entraînement. Les
    **10 000 images de test** servent de **validation** : c'est sur elles qu'on
    mesure la précision du modèle.

## 1. Les données

Construisez les transformations, les deux jeux de données et les deux
`DataLoader`.

- La transformation d'entraînement contient de la **data augmentation** :
  un recadrage aléatoire (`RandomCrop(32, padding=4)`) et une symétrie
  horizontale aléatoire (`RandomHorizontalFlip()`). Vous verrez pourquoi en cours.
- La transformation de validation n'en contient **jamais** : on veut mesurer le
  modèle sur les vraies images.
- Les deux se terminent par `ToTensor()` puis `Normalize(MEAN, STD)`.

Documentation : [transforms](https://docs.pytorch.org/vision/stable/transforms.html),
[CIFAR10](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html),
[DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

??? tip "Indice"
    C'est la première cellule de code du tutoriel, avec deux transformations au
    lieu d'une, et `batch_size=BATCH_SIZE`. On mélange (`shuffle=True`) le jeu
    d'entraînement, pas celui de validation.

## 2. Le modèle

Créez un `ResNet18` (importé de `resnet.py`), envoyez-le sur le GPU, et comptez
ses paramètres comme à la question Q3 de la partie 1.

!!! success "Résultat attendu"
    11 173 962 paramètres, et une sortie de forme `(128, 10)` pour un batch de
    128 images : un score par classe et par image.

## 3. Perte, optimiseur, scheduler

| Élément | Choix |
|---|---|
| Perte | entropie croisée, [`nn.CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) |
| Optimiseur | [`torch.optim.SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html), `lr=LR`, `momentum=0.9`, `weight_decay=5e-4` |
| Scheduler | [`CosineAnnealingLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html), `T_max=EPOCHS` |

Le scheduler fait décroître le taux d'apprentissage de `LR` à 0 au fil des
époques, en suivant un demi-cosinus. Le *weight decay* est une régularisation :
vous le reverrez dans le cours sur la data augmentation.

## 4. Une époque d'entraînement

Complétez `train_one_epoch`. La structure est donnée ; il reste à écrire le
cœur de la boucle.

??? tip "Indice"
    C'est la boucle du tutoriel, avec trois différences :

    1. les données doivent aller sur le GPU, comme le modèle ;
    2. on accumule la perte et le nombre de bonnes réponses, pour renvoyer des
       moyennes sur toute l'époque ;
    3. la classe prédite est l'indice du score maximal : `outputs.argmax(dim=1)`.

    Attention : `loss` est la perte **moyenne** du batch. Pour la moyenne sur
    l'époque, multipliez-la par la taille du batch avant de l'accumuler.

## 5. L'évaluation

Écrivez `evaluate(model, loader, criterion)`, qui renvoie la perte moyenne et
la précision sur un jeu de données, **sans entraîner**.

??? tip "Indice"
    Partez de `train_one_epoch` et retirez tout ce qui sert à apprendre. Deux
    choses à ajouter :

    - `model.eval()` au lieu de `model.train()` : certaines couches (ici la
      BatchNorm) ne se comportent pas de la même façon en entraînement et en test ;
    - `with torch.no_grad():` autour de la boucle : PyTorch ne garde alors pas de
      quoi calculer les gradients, ce qui économise mémoire et temps.

!!! success "Résultat attendu"
    Sur le réseau **pas encore entraîné** : une perte proche de 2,3 et une
    précision proche de 10 %. Le réseau répond au hasard parmi 10 classes, et
    $\ln(10) \approx 2{,}3$.

## 6. La boucle d'entraînement

Pour chaque époque : entraîner, évaluer, faire avancer le scheduler
(`scheduler.step()`), afficher les résultats et les ranger dans `history`.

Sauvegardez aussi les poids **chaque fois que la précision de validation
s'améliore** :

```python
torch.save(model.state_dict(), "best.pt")
```

Avec `EPOCHS = 2`, comptez de l'ordre d'une minute par époque sur Colab. Après 2 époques, on attend environ 45 à 60 % en validation.

## 7. Les courbes

Tracez, en fonction de l'époque, la perte puis la précision, en entraînement et
en validation. Sauvegardez la figure dans `curves.png`.

??? tip "Indice"
    `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))` crée deux graphiques
    côte à côte ; `ax1.plot(liste, label="train")` trace une courbe ;
    `ax1.legend()` affiche la légende ; `fig.savefig("curves.png")` sauvegarde.

## 8. Recharger le modèle

Créez un **nouveau** ResNet18, chargez-y les poids de `best.pt`, et évaluez-le :
vous devez retrouver exactement la meilleure précision de l'entraînement.

C'est ce que vous ferez au début de chaque TP de compression : repartir de votre
baseline sans la réentraîner.

Documentation : [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

## 9. L'entraînement complet

Une fois que tout marche avec 2 époques :

1. passez à `EPOCHS = 30` ;
2. relancez tout : *Exécution > Tout exécuter* ;
3. **gardez l'onglet ouvert** : comptez environ 20 à 30 minutes sur le GPU de Colab.

!!! success "Résultat attendu"
    Environ **93 %** de précision en validation après 30 époques (et autour de 95 %
    si l'on entraîne beaucoup plus longtemps). À la fin, `best.pt` et `curves.png`
    sont dans votre dossier.

!!! warning "Gardez votre baseline"
    Les fichiers de Colab disparaissent à la fin de la session : **téléchargez
    `best.pt`** (panneau *Fichiers*, clic droit, *Télécharger*), renommez-le
    `baseline_resnet18.pt` et notez sa précision. C'est votre point de référence
    pour tout le module.

## Questions sur vos courbes

**Q1.** Pendant les premières époques, la précision de validation est souvent
**au-dessus** de celle d'entraînement. Comment est-ce possible ? Proposez au
moins deux raisons.

??? tip "Indice"
    Les images d'entraînement sont-elles les mêmes que celles de validation, une
    fois transformées ? Et à quel moment de l'époque mesure-t-on la précision
    d'entraînement ?

**Q2.** À la fin de l'entraînement, quel est l'écart entre précision
d'entraînement et de validation ? Le modèle surapprend-il ?

**Q3.** Le taux d'apprentissage diminue au fil des époques. Comment cela se
voit-il sur vos courbes, surtout à la fin ?

---

Ensuite : la [partie 3](03_data_augmentation.md).
