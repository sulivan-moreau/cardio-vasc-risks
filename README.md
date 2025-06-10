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


---

### Activité physique

|Variable|interpretation| valeur élevé =|
|:-|:-|:-|
| CHOLESTEROL | 1 : normal<br> 2 : supérieur à la normale<br> 3 : largement supérieur à la normale| incidence négative|
| GLUCOSE     | 1 : normal<br> 2 : supérieur à la normale<br> 3 : largement supérieur à la normale| incidence négative|
| SMOKE       | 0 : non fumeur<br>1 : fumeur| incidence négative|
| ALCOHOL     | 0 : non consomateur<br> 1 : consomateur| incidence négative|
| PHYSICAL_ACTIVITY | 0 : non, 1 : oui| incidence positive|



Pour l'activité physique, la logique est inversée<br>
Nous alons donc intervertir les 0 et 1 de `active` pour respecter la même logique que sur les autre variables.


----
## gender
2 = homme<br>
1 = femme<br>
Mais il s'agit d'une donnée Cardinale, pas ordinale><br>
Nous utilisons OneHotEncoder pour remplacer `gender` par `male`et `female`.


---
## IMC

### Indice de Masse Corporelle

Calcul de l'IMC :

$$
IMC = \frac{\text{poids (kg)}}{\left( \frac{\text{taille (cm)}}{100} \right)^2}
$$

​
Catégorisation médicale simplifiée :

|IMC	|Catégorie|	Valeur|
|-|-|-|
|< 25	|Normal	|1|
|25 ≤ IMC < 30	|Surpoids |	2|
|≥ 30	|Obésité |	3|



---
## pressure

### Tension artérielle

| Diastolique (`ap_lo`) | Systolique (`ap_hi`) | Interprétation        | Code |
| -------------------- | --------------------- | --------------------- | ---- |
| < 80                | < 120                | Normale               | 1    |
| 80–89             | 120–139               | Élevée (à surveiller) | 2    |
| ≥ 90                | ≥ 140                  | Hypertension          | 3    |


Si la systolique ou la diastolique dépasse un seuil, on prend la catégorie la plus élevée.