# 5. Projet fil rouge : partie 2

## La consigne

!!! abstract "Objectif"
    Choisissez une technique principale parmi celles du cours 4 (**distillation**, **pruning**
    ou **quantification**) et appliquez-la à **votre réseau du fil rouge**. Les combinaisons
    sont autorisées. Jusqu'où peut-on le compresser, et à quel prix en précision ?

Vous présenterez votre méthode, vos résultats et vos conclusions devant toute la promo le
**vendredi 20 novembre**, et vous serez évalués par l'enseignant **et par vos camarades**, avec la
grille de la page [7. Présentations du 20 novembre](07_presentations.md).

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

## La présentation et l'évaluation

Le déroulé de la séance du 20 novembre et la grille d'évaluation sont sur une page à part :
[7. Présentations du 20 novembre](07_presentations.md).
