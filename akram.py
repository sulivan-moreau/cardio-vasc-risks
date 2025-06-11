# === Simulation d’un dataset santé simplifié 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

np.random.seed(42)
n = 1000

# Variables explicatives
age = np.random.randint(30, 65, size=n)
gender = np.random.choice([0, 1], size=n)  # 0 = femme, 1 = homme
smoker = np.random.choice([0, 1], size=n, p=[0.9, 0.1])  # déséquilibré
alcohol = np.random.choice([0, 1], size=n, p=[0.92, 0.08])  # déséquilibré

# Variable cible équilibrée
cardio_risk = np.random.choice([0, 1], size=n, p=[0.5, 0.5])

# Dataset initial équilibré
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'smoker': smoker,
    'alcohol': alcohol,
    'cardio_risk': cardio_risk
})

# === FEATURE ENGINEERING : on crée une nouvelle variable combinée ===
df['risk_behavior'] = df['smoker'] + df['alcohol']

# QUESTION : Pourquoi "risk_behavior" est utile ?
# RÉPONSE :
# Cette variable agrège deux comportements à risque (fumer, boire).
# Elle capte mieux l'effet cumulé, souvent plus informatif pour prédire un problème cardiovasculaire.

# === Cas 1 : variable cible ÉQUILIBRÉE ===
X_eq = df[['age', 'gender', 'smoker', 'alcohol', 'risk_behavior']]
y_eq = df['cardio_risk']

# Régression sans pénalisation
model_none = LogisticRegression(penalty='None', max_iter=1000)
model_none.fit(X_eq, y_eq)
print("=== Coefficients (pas de pénalisation, cible équilibrée) ===")
print(dict(zip(X_eq.columns, model_none.coef_[0])))

# Régression avec pénalisation L2
model_l2 = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
model_l2.fit(X_eq, y_eq)
print("=== Coefficients (L2 - Ridge) ===")
print(dict(zip(X_eq.columns, model_l2.coef_[0])))

# Régression avec pénalisation L1
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)
model_l1.fit(X_eq, y_eq)
print("=== Coefficients (L1 - Lasso) ===")
print(dict(zip(X_eq.columns, model_l1.coef_[0])))

# QUESTION : Que fait L1 ici ?




# RÉPONSE :
# La pénalisation L1 pousse certains coefficients vers zéro → sélection automatique de variables pertinentes.
# Utile quand on a trop de variables ou du bruit dans les données.




# === Cas 2 : variable cible DÉSÉQUILIBRÉE (90% non à risque) ===
# Simulation d’un déséquilibre
df_imb = pd.concat([
    df[df['cardio_risk'] == 0].sample(frac=0.9, random_state=1),
    df[df['cardio_risk'] == 1].sample(frac=0.1, random_state=1)
])

X_imb = df_imb[['age', 'gender', 'smoker', 'alcohol', 'risk_behavior']]
y_imb = df_imb['cardio_risk']

# Régression sans class_weight → biaisé par la classe majoritaire
model_imb_naive = LogisticRegression(max_iter=1000)
model_imb_naive.fit(X_imb, y_imb)
print("=== Déséquilibré, sans class_weight ===")
print(dict(zip(X_imb.columns, model_imb_naive.coef_[0])))
print(classification_report(y_imb, model_imb_naive.predict(X_imb)))

# Régression avec class_weight='balanced' → corrige le déséquilibre
model_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
model_balanced.fit(X_imb, y_imb)
print("=== Déséquilibré, avec class_weight='balanced' ===")
print(dict(zip(X_imb.columns, model_balanced.coef_[0])))
print(classification_report(y_imb, model_balanced.predict(X_imb)))

# QUESTION : Pourquoi le modèle naïf est-il mauvais ?
# RÉPONSE :
# Il apprend surtout à prédire la classe majoritaire (0).
# Il a une précision globale élevée, mais ignore la minorité (1).

# QUESTION : Que fait class_weight='balanced' ?
# RÉPONSE :
# Il donne plus de poids aux erreurs sur la classe minoritaire pour forcer l’algorithme à en tenir compte.
# Cela permet de mieux détecter les cas de risque malgré leur rareté.

# QUESTION : Est-ce utile d’utiliser L1 ou L2 en plus ?
# RÉPONSE :
# Oui, on peut combiner penalty='l1' ou 'l2' avec class_weight='balanced' pour faire à la fois :
# - pondération des classes
# - sélection ou régularisation des variables

# === Conclusion pour les apprenants ===
# - Toujours inspecter l’équilibre des classes cible.
# - Faire du feature engineering métier (ex : risk_behavior) améliore les performances.
# - Adapter le modèle (class_weight, pénalisation) selon la distribution des données.