# cardio-vasc-risks


## Environnement virtuel
Créé avec l’outil [uv](https://docs.astral.sh/uv/) développé par Astral (une alternative ultra-rapide à pip et venv).

1. **Installer uv :**
`pip install uv`

2. **Activer l'environnement virtuel et installer les dépendances :**
`uv sync`

## Jeu de données

**variable indépendantes**
|Anglais|Français|Signification|besoin métier|
|:-|:-|:-|-|
| AGE     | ÂGE | âge en nombre de jours (entier)||
| HEIGHT  | TAILLE | taille en cm (entier)||
| WEIGHT  | POIDS | poids en kg (entier)||
| GENDER  | SEXE | sexe, catégoriel (1 : femme, 2 : homme)|valeur cardinal|
| AP_HIGH | Tension artérielle systolique | lorsque le cœur se contracte pour éjecter le sang<br> (entier)||
| AP_LOW  | Tension artérielle diastolique | entre deux battements de coeur <br> (entier)||
| CHOLESTEROL | CHOLESTÉROL | taux de cholestérol <br> catégoriel (1 : normal, 2 : supérieur à la normale, 3 : largement supérieur à la normale)|valeur élevé = incidence négative|
| GLUCOSE     | GLUCOSE | taux de glucose <br> catégoriel (1 : normal, 2 : supérieur à la normale, 3 : largement supérieur à la normale)|valeur élevé = incidence négative|
| SMOKE       | TABAGISME | si le patient fume ou non <br> catégoriel (0 : non, 1 : oui)|valeur élevé = incidence négative|
| ALCOHOL     | ALCOOL | si le patient consomme ou non de l'alcool <br> catégoriel (0 : non, 1 : oui)|valeur élevé = incidence négative|
| PHYSICAL_ACTIVITY | ACTIVITÉ_PHYSIQUE | si le patient est actif ou non <br> catégoriel (0 : non, 1 : oui)|il faut inverser les valeurs|

**variable cible**
|Anglais|Français|Signification|
|:-|:-|:-|
| CARDIO_DISEASE | MALADIE_CARDIOLOGIQUE | si le patient contracté la maladie ou non, catégorique (0 : non, 1 : oui)
