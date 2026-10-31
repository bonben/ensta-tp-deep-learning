# 5. Le projet final


## La consigne

!!! abstract "Objectif"
    Choisissez **une** des trois techniques du cours 4 (**distillation**, **élagage** ou
    **quantification**) et appliquez-la à **votre réseau du fil rouge**. Jusqu'où peut-on le
    compresser, et à quel prix en précision ?

Vous présenterez votre méthode, vos résultats et vos conclusions devant toute la promo le
**vendredi 20 novembre**, et vous serez évalués par l'enseignant **et par vos camarades**, avec la
grille ci-dessous.

| Technique | La question | Amorce |
|---|---|---|
| [5a. Distillation](05a_distillation.md) | Un enseignant fait-il gagner de la précision à votre réseau, ou lui permet-il de rétrécir ? | `amorce_distillation.ipynb` |
| [5b. Élagage](05b_elagage.md) | Combien de poids de votre réseau peut-on mettre à zéro ? | `amorce_elagage.ipynb` |
| [5c. Quantification](05c_quantification.md) | Sur combien de bits peut-on écrire ses poids ? | `amorce_quantification.ipynb` |

Chaque page fournit une **amorce** : un notebook qui fonctionne tel quel sur un réseau de
démonstration, des repères mesurés et des pistes. Votre travail commence là où l'amorce s'arrête.

## Ce qu'on attend

1. **Un point de départ chiffré** : votre réseau du fil rouge, ses paramètres, sa précision.
   C'est votre référence : chaque résultat se compare à lui, **au même nombre d'époques**.
2. **Une question précise**, et une prédiction écrite avant de lancer.
3. **Au moins une courbe de compromis** : la précision en fonction de la taille (taux
   d'élagage, nombre de bits, taille de l'élève…), et non un seul point.
4. **Au moins deux réglages explorés**, une chose à la fois (température, réentraînement,
   couches épargnées, élagage en plusieurs étapes…).
5. **Des conclusions honnêtes** : ce qui marche, ce qui ne marche pas, les limites de vos
   mesures, et ce que votre compression ferait gagner **vraiment** sur une carte embarquée.

Pour chaque essai, notez : paramètres (et paramètres **non nuls** pour l'élagage), nombre de
bits, **taille mémoire en Ko**, nombre d'époques, **précision au dernier epoch**.

Pour comparer des techniques différentes, une même unité : **la taille mémoire des poids**,
en kilo-octets. Un réseau de 371 658 paramètres en 32 bits pèse 1 452 Ko ; en 4 bits, 198 Ko.

## La présentation du 20 novembre

- **7 minutes par binôme**, puis 3 minutes de questions. Le temps est tenu.
- Environ **5 slides** : le point de départ, la question et la méthode, les résultats (vos
  courbes), les conclusions et les limites.
- Chacun présente **sur sa machine**.
- Pendant les présentations des autres, vous remplissez le **formulaire d'évaluation** en ligne
  (le lien sera donné en séance) : une réponse par binôme présenté, sauf le vôtre.

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

Le formulaire demande aussi, pour chaque binôme, **un point fort** et **une suggestion** :
une phrase chacun, précise. « Bien » ou « rien à dire » ne compte pas.

!!! tip "Évaluer ses camarades"
    Notez ce que vous avez vu, pas ce que vous imaginez. Un résultat négatif bien mesuré et bien
    expliqué vaut mieux qu'un résultat spectaculaire sans référence : la grille récompense la
    **démarche**, pas le chiffre final.
