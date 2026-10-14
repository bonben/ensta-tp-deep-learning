# Lecture d'articles

Chaque binôme lit **un article** fondateur de la vision par ordinateur et le
présente à la promotion. À la fin, vous aurez entendu les douze : c'est la
culture de base de quelqu'un qui conçoit un système d'IA aujourd'hui.

## Les douze articles

Les présentations ont lieu sur **deux séances**, dans l'ordre du tableau : la
séance A raconte l'histoire des réseaux convolutifs, la séance B celle des
Transformers et des modèles de fondation, qui s'appuie sur la première.

| | Article | Lien | La question à creuser |
|---|---|---|---|
| **Séance A** | | | |
| 1 | ResNet (He et al., 2015) | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | Pourquoi un réseau plus profond peut-il être *moins bon*, et comment le raccourci résiduel règle-t-il le problème ? |
| 2 | DenseNet (Huang et al., 2016) | [arXiv:1608.06993](https://arxiv.org/abs/1608.06993) | Comment fait-il aussi bien que ResNet avec moins de paramètres ? Qu'est-ce que ça coûte en mémoire ? |
| 3 | MobileNetV2 (Sandler et al., 2018) | [arXiv:1801.04381](https://arxiv.org/abs/1801.04381) | Combien d'opérations économise une convolution séparable en profondeur ? Faites le calcul. |
| 4 | EfficientNet (Tan & Le, 2019) | [arXiv:1905.11946](https://arxiv.org/abs/1905.11946) | Pour grossir un réseau, faut-il l'élargir, l'approfondir ou augmenter la résolution ? |
| 5 | YOLO (Redmon et al., 2015) | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) | Comment détecter des objets en une seule passe, en temps réel ? Une slide sur les versions actuelles. |
| 6 | ConvNeXt (Liu et al., 2022) | [arXiv:2201.03545](https://arxiv.org/abs/2201.03545) | En modernisant un ResNet étape par étape, qu'est-ce qui compte le plus : l'architecture ou la recette d'entraînement ? |
| **Séance B** | | | |
| 7 | ViT (Dosovitskiy et al., 2020) | [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | Une image découpée en mots : pourquoi faut-il autant de données pour que ça marche ? |
| 8 | DeiT (Touvron et al., 2020) | [arXiv:2012.12877](https://arxiv.org/abs/2012.12877) | Comment entraîner un ViT sans données géantes ? Rôle de la data augmentation et de la distillation. |
| 9 | DINO (Caron et al., 2021) | [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Apprendre sans étiquettes par auto-distillation. Une slide pour situer DINOv2 et DINOv3. |
| 10 | CLIP (Radford et al., 2021) | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Comment classer des images dans des catégories jamais vues à l'entraînement ? |
| 11 | SAM (Kirillov et al., 2023) | [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) | Un modèle qui segmente « n'importe quoi » : comment ont-ils obtenu les données ? |
| 12 | Stable Diffusion (Rombach et al., 2021) | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | Pourquoi diffuser dans un espace latent plutôt que sur les pixels ? Combien de calcul ça économise ? |

La répartition des articles entre binômes se fait en séance.

## Le format

**5 minutes de présentation, strictement, puis 2 minutes de questions.**
Trois ou quatre slides, qui répondent à ces questions :

1. **Le problème** : qu'est-ce qui ne marchait pas avant cet article ?
2. **L'idée clé**, en une figure : la vôtre ou celle de l'article, mais que vous
   savez expliquer.
3. **Le coût** : nombre de paramètres, opérations, données et calcul
   d'entraînement. C'est le fil rouge du module.
4. **La question à creuser** du tableau.

Déposez votre PDF dans le dossier partagé **la veille de votre séance** : les
présentations s'enchaînent sur une seule machine.

## Comment lire un article

On ne lit pas un article de recherche d'une traite, de la première à la dernière
ligne. Lisez-le en trois passes :

1. **Survol (10 min)** : titre, résumé, introduction, figures, conclusion.
   Vous savez alors de quoi il parle et ce qu'il prétend.
2. **Lecture (1 h)** : le corps de l'article, en sautant les détails
   techniques. Notez ce que vous ne comprenez pas.
3. **Approfondissement** : seulement pour la partie au cœur de votre
   présentation. Cherchez les références qui vous manquent.

Méthode décrite par S. Keshav, [*How to Read a Paper*](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf).

!!! note "Et les assistants IA ?"
    Vous pouvez vous en servir pour lire. Mais un résumé ne vous apprendra ni ce
    que l'article ne dit pas, ni ce qui cloche dans ses expériences, et les
    questions porteront justement là-dessus.
