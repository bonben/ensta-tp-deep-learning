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

Presque tout ce que vous écrivez est dans le tutoriel de la
[partie 1](01_pytorch.md) : **chaque TODO indique la section du tutoriel où
aller chercher**, et donne les quelques lignes nouvelles. Gardez le tutoriel
ouvert dans un autre onglet.

!!! info "Jeux de données"
    Les **50 000 images d'entraînement** de CIFAR-10 servent à l'entraînement. Les
    **10 000 images de test** servent de **validation** : c'est sur elles qu'on
    mesure la précision du modèle.

Ci-dessous, ce que fait chaque étape et pourquoi. Le détail de ce qu'il faut
écrire est dans le notebook.

## 1. Les données

Comme dans le tutoriel, avec une différence importante : **deux
transformations**. Celle d'entraînement contient de la **data augmentation**
(un recadrage aléatoire et une symétrie horizontale aléatoire) ; celle de
validation n'en contient **jamais**, car on veut mesurer le modèle sur les
vraies images. Vous verrez pourquoi en cours.

## 2. Le modèle

Un `ResNet18`, sur le GPU. Comptez ses paramètres comme à la question Q3 de la
partie 1.

!!! success "Résultat attendu"
    11 173 962 paramètres, et une sortie de forme `(128, 10)` pour un batch de
    128 images : un score par classe et par image.

## 3. Perte, optimiseur, scheduler

La même perte que le tutoriel, le même optimiseur avec d'autres réglages. Le
*weight decay* est une régularisation, que vous verrez dans le cours sur la data
augmentation. Le scheduler (fourni) fait décroître le taux d'apprentissage au
fil des époques.

## 4. Une époque d'entraînement

La boucle d'entraînement du tutoriel, rangée dans une fonction
`train_one_epoch`, qui renvoie en plus la perte moyenne et la précision sur
l'époque.

## 5. L'évaluation

Le calcul de précision du tutoriel, rangé dans une fonction `evaluate`, qui
calcule en plus la perte. Deux lignes comptent ici : `model.eval()` (certaines
couches, comme la BatchNorm, ne se comportent pas pareil en entraînement et en
test) et `with torch.no_grad():` (inutile de garder de quoi calculer des
gradients quand on n'apprend pas).

!!! success "Résultat attendu"
    Sur le réseau **pas encore entraîné** : une perte proche de 2,3 et une
    précision proche de 10 %. Le réseau répond au hasard parmi 10 classes, et
    $\ln(10) \approx 2{,}3$.

## 6. La boucle d'entraînement

C'est la seule partie qui n'existe pas dans le tutoriel : à chaque époque, on
entraîne, on évalue, on fait avancer le scheduler, on garde l'historique, et on
**sauvegarde les poids chaque fois que la précision de validation
s'améliore**. Le squelette est dans le notebook.

Avec `EPOCHS = 2`, comptez de l'ordre d'une minute par époque sur Colab. Après
2 époques, on attend environ 50 à 60 % en validation.

La cellule suivante (fournie) trace les **courbes** de perte et de précision,
en entraînement et en validation, et les sauvegarde dans `curves.png`.

## 7. Recharger le modèle

On crée un **nouveau** ResNet18, on y charge les poids de `best.pt`, et on
l'évalue : on doit retrouver exactement la meilleure précision de
l'entraînement.

C'est ce que vous ferez au début de chaque TP de compression : repartir de votre
baseline sans la réentraîner.

## 8. L'entraînement complet

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
