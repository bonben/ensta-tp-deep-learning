# 5. Projet fil rouge : partie 2

## La consigne

!!! abstract "Objectif"
    Choisissez une technique principale parmi celles du cours 4 (**distillation**, **pruning**
    ou **quantification**) et appliquez-la à **votre réseau du fil rouge**. Les combinaisons
    sont autorisées. Jusqu'où peut-on le compresser, et à quel prix en précision ?

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

1. **L'architecture du réseau** : une description claire du réseau de départ, et du réseau
   final s'il diffère. Blocs, nombre d'étages, largeurs (nombre de canaux), types de
   convolution, nombre de paramètres. Un schéma est le plus lisible.
2. **La métrique de taille** : choisissez ce que vous cherchez à réduire (nombre de paramètres,
   paramètres non nuls, taille mémoire en Ko, MAC…) et expliquez pourquoi elle est pertinente
   pour votre technique.
3. **La stratégie** : les essais dans l'ordre où vous les avez faits, avec pour chacun ce qui a
   changé, le résultat obtenu et la décision prise (garder, abandonner, et pourquoi).
4. **Les résultats** : au moins une courbe de compromis, précision en fonction de votre
   métrique, comparée au réseau de départ entraîné avec le même nombre d'époques.
5. **Les conclusions** : ce qui a fait gagner le plus, ce qui n'a pas marché, les limites de vos
   mesures.

Facultatif :

- **le budget de calcul** (nombre d'essais, époques, temps GPU) : apprécié ;
- **l'intérêt sur une cible embarquée** (mémoire, temps de calcul, énergie).

Pour chaque essai, notez : paramètres (paramètres **non nuls** pour le pruning), nombre de
bits, taille mémoire en Ko, nombre d'époques, **précision au dernier epoch**.

Repère : un réseau de 371 658 paramètres pèse 1 452 Ko en 32 bits, 198 Ko en 4 bits.

## La présentation du 20 novembre

- **8 minutes par binôme**, puis 4 minutes de questions.
- Environ **5 slides** : architecture, métrique et stratégie, résultats (courbes),
  conclusions.
- Chaque binôme présente **sur sa machine**.
- Pendant les présentations des autres, vous remplissez le **formulaire d'évaluation** en ligne
  (le lien sera donné en séance) : un seul formulaire pour toute la séance, avec une grille par
  binôme, sauf le vôtre. Il s'envoie à la fin des présentations.

## La grille d'évaluation

C'est la même pour l'enseignant et pour vous. Chaque critère est noté de 1 à 4. La note finale
compte pour **70 % la note de l'enseignant** et **30 % la moyenne des évaluations des pairs**.

| Critère | 4 : très bien | 1 : insuffisant |
|---|---|---|
| **1. Architecture et métrique** | L'architecture est décrite clairement ; la métrique de taille est choisie et justifiée. | On ne sait pas quel réseau est compressé ni ce qui est mesuré. |
| **2. Stratégie** | Les essais s'enchaînent, chaque décision s'appuie sur les mesures précédentes. | Des essais sans fil conducteur. |
| **3. Rigueur** | Une référence, une chose à la fois, même nombre d'époques ; les résultats douteux sont revérifiés. | Des chiffres incomparables entre eux. |
| **4. Résultats** | Une courbe de compromis lisible, des mesures complètes (taille, précision). | Un seul chiffre, ou des résultats illisibles. |
| **5. Conclusions** | Ce qui a marché, ce qui n'a pas marché, les limites. | Pas de conclusion, ou une conclusion que les résultats ne montrent pas. |
| **6. Présentation** | Claire, dans le temps, réponses précises aux questions. | Confuse, hors du temps. |

Pour chaque binôme, le formulaire propose aussi un champ facultatif : un point fort et une
suggestion.

La grille note la démarche et non la valeur du résultat : un résultat négatif correctement
mesuré peut obtenir la note maximale.
