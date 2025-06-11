# Veille scolaire et protocolaire sur la régression logistique

## Introduction
La régression logistique est une méthode statistique et d'apprentissage automatique utilisée pour modéliser la relation entre un ensemble de variables indépendantes, appelées prédicteurs, et une variable dépendante binaire, qui ne prend que deux valeurs, comme zéro ou un. Elle est particulièrement adaptée aux problèmes de classification, où l'objectif est de prédire la probabilité qu'une observation appartienne à une catégorie spécifique. Cette veille explore ses principes, son fonctionnement, ses applications et les bonnes pratiques.

## 1. Principes fondamentaux
La régression logistique s'appuie sur une fonction spéciale, appelée sigmoïde, qui transforme une combinaison linéaire des prédicteurs en une probabilité comprise entre zéro et un. Contrairement à la régression linéaire, qui prédit une valeur continue, elle estime la chance qu'un événement se produise, comme un risque médical.

## 2. Fonction de coût
Pour entraîner le modèle, on utilise une fonction de coût, souvent appelée entropie croisée binaire, qui mesure l'écart entre les prédictions du modèle et les vraies étiquettes. Cette mesure est calculée pour chaque observation, puis moyennée sur toutes les données. Elle aide à ajuster le modèle pour qu'il devienne plus précis en minimisant cet écart.

## 3. Entraînement et optimisation
L'entraînement consiste à ajuster les paramètres du modèle pour réduire l'erreur mesurée par la fonction de coût. On utilise une technique appelée descente de gradient, qui modifie progressivement ces paramètres en fonction de la direction où l'erreur diminue le plus. La vitesse de ces ajustements dépend d'un paramètre appelé taux d'apprentissage.

## 4. Interprétation des résultats
Pour prendre une décision, on compare la probabilité prédite à un seuil, souvent fixé à la moitié, pour classer l'observation dans une catégorie ou une autre. On peut aussi analyser l'impact d'une variable en calculant un indicateur appelé odds ratio, qui montre comment une augmentation d'une variable influence la probabilité, en tenant les autres constantes.

## 5. Applications
- **Médecine** : Prédire des risques, comme les maladies cardiovasculaires.
- **Marketing** : Identifier les clients susceptibles d'acheter.
- **Finance** : Détecter les fraudes.

## 6. Bonnes pratiques
- **Préparation des données** : Normaliser les variables pour améliorer la convergence du modèle.
- **Régularisation** : Ajouter une pénalité pour limiter la complexité et éviter le surapprentissage.
- **Validation croisée** : Diviser les données en ensembles d'entraînement et de test, ou utiliser une méthode comme k-fold, pour évaluer la performance.
- **Vérification des hypothèses** : Vérifier qu'il n'y a pas de forte corrélation entre les prédicteurs.

## 7. Limites
- Sensible aux données déséquilibrées, où une classe domine l'autre, nécessitant des techniques comme des poids de classe ou SMOTE.
- Suppose une relation spécifique entre les prédicteurs et la probabilité, qui peut ne pas convenir à des données très complexes.
- Moins efficace pour des données très non linéaires, où d'autres modèles comme les SVM ou les réseaux de neurones pourraient être préférés.

## Conclusion
La régression logistique est un outil puissant et interprétable pour la classification binaire. Sa capacité à estimer des probabilités et son optimisation via une mesure d'erreur en font une méthode de choix dans de nombreux domaines, à condition de suivre les bonnes pratiques et de connaître ses limites.