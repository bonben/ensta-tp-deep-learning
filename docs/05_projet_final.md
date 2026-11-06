# 5. Projet fil rouge : partie 2


## La consigne

!!! abstract "Objectif"
    Choisissez **une** des trois techniques du cours 4 (**distillation**, **pruning** ou
    **quantification**) et appliquez-la à **votre réseau du fil rouge**. Jusqu'où peut-on le
    compresser, et à quel prix en précision ?

Vous présenterez votre méthode, vos résultats et vos conclusions devant toute la promo le
**vendredi 20 novembre**, et vous serez évalués par l'enseignant **et par vos camarades**, avec la
grille ci-dessous.

| Technique | La question | Amorce |
|---|---|---|
| [5a. Distillation](05a_distillation.md) | Un teacher fait-il gagner de la précision à votre réseau, ou lui permet-il de rétrécir ? | `amorce_distillation.ipynb` |
| [5b. Pruning](05b_elagage.md) | Combien de poids de votre réseau peut-on mettre à zéro ? | `amorce_elagage.ipynb` |
| [5c. Quantification](05c_quantification.md) | Sur combien de bits peut-on écrire ses poids ? | `amorce_quantification.ipynb` |

Chaque page fournit une **amorce** (un notebook qui fonctionne tel quel sur un réseau de
démonstration), des repères mesurés et des pistes.

## Contenu demandé

1. **Le réseau de départ** : votre réseau du fil rouge, son nombre de paramètres, sa précision.
   Tous les résultats sont comparés à ce réseau, entraîné avec le même nombre d'époques.
2. **La question étudiée**, et le résultat que vous en attendiez avant les mesures.
3. **Au moins une courbe de compromis** : précision en fonction de la taille (taux de pruning,
   nombre de bits, taille du student…), sur plusieurs points.
4. **Au moins deux réglages explorés**, un paramètre modifié à la fois (température,
   fine-tuning, couches non compressées, pruning en plusieurs étapes…).
5. **Les conclusions** : ce que montrent les mesures, leurs limites, et le gain attendu sur une
   cible embarquée (mémoire, temps de calcul).

Pour chaque essai, notez : paramètres (paramètres **non nuls** pour le pruning), nombre de
bits, **taille mémoire en Ko**, nombre d'époques, **précision au dernier epoch**.

Pour comparer des techniques différentes, l'unité commune est **la taille mémoire des poids**,
en kilo-octets. Un réseau de 371 658 paramètres pèse 1 452 Ko en 32 bits, 198 Ko en 4 bits.

## La présentation du 20 novembre

- **8 minutes par binôme**, puis 4 minutes de questions.
- Environ **5 slides** : réseau de départ, question et méthode, résultats (courbes),
  conclusions et limites.
- Chaque binôme présente **sur sa machine**.
- Pendant les présentations des autres, vous remplissez le **formulaire d'évaluation** en ligne
  (le lien sera donné en séance) : un seul formulaire pour toute la séance, avec une grille par
  binôme, sauf le vôtre. Il s'envoie à la fin des présentations.

## La grille d'évaluation

C'est la même pour l'enseignant et pour vous. Chaque critère est noté de 1 à 4.

| Critère | 4 : très bien | 1 : insuffisant |
|---|---|---|
| **1. Point de départ et question** | Le réseau de départ est chiffré (taille, précision), la question est précise. | On ne sait pas d'où l'on part ni ce que l'on cherche. |
| **2. Méthode** | La technique est expliquée et comprise, les réglages sont justifiés. | La technique est appliquée sans être comprise. |
| **3. Rigueur** | Une référence, une chose à la fois, même nombre d'époques ; les résultats douteux sont revérifiés. | Des chiffres incomparables entre eux. |
| **4. Résultats** | Une courbe de compromis lisible, des mesures complètes (taille, précision). | Un seul chiffre, ou des résultats illisibles. |
| **5. Conclusions** | Ce qu'on retient, les limites, ce que ça donnerait sur une cible réelle. | Pas de conclusion, ou une conclusion que les résultats ne montrent pas. |
| **6. Présentation** | Claire, dans le temps, réponses précises aux questions. | Confuse, hors du temps. |

Pour chaque binôme, le formulaire propose aussi un champ facultatif : un point fort et une
suggestion.

La grille note la démarche et non la valeur du résultat : un résultat négatif correctement
mesuré peut obtenir la note maximale.
