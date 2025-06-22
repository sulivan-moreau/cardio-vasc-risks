# --- Imports ---
import os
import sys
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, make_scorer, accuracy_score, classification_report, precision_recall_curve, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import matplotlib.pyplot as plt

# --- Ajout du chemin vers utils.py ---
sys.path.append(os.path.abspath(".."))
from utils import execute_model, save_results, calculate_scores, ThresholdedClassifier



# --- Chargement des données ---
DF_PATH = "../data/balanced_3.csv"
df = pd.read_csv(DF_PATH, sep=";")

# --- Séparation features / cible ---
X = df.drop(columns=["cardio"])
y = df["cardio"]

# --- Calcul poids manuel pour déséquilibre ---
counter = Counter(y)
n0, n1 = counter[0], counter[1]
manual_weight = {0: 1, 1: round(n0 / n1, 2)}
print("Poids manuel calculé :", manual_weight)

# --- Création d'un scorer personnalisé (recall classe 1) ---
recall_scorer = make_scorer(recall_score, pos_label=1)

# --- Définition modèle de base ---
model = LogisticRegression()

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# --- Grille d'hyperparamètres pour GridSearchCV ---
param_grid = [
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'C': [28, 29],
        'l1_ratio': [0.47, 0.49],
        'class_weight': ['balanced', manual_weight],
        'max_iter': [1000]
    }
]


# --- GridSearchCV ---
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring=recall_scorer,
    n_jobs=-1,
    verbose=2
)

if "best_model_grid" not in globals():
    grid_search.fit(X_train, y_train)
    best_model_grid = grid_search.best_estimator_
    print("✅ Grid search terminée.")
else:
    print("✅ Modèle déjà entraîné, utilisation de best_model_grid.")

print("Meilleurs hyperparamètres :", grid_search.best_params_)
print("Meilleur score CV (recall) :", grid_search.best_score_)


# --- Sauvegarde ou Chargement du meilleur modèle ---
MODEL_PATH = "best_model_logreg.joblib"

if not os.path.exists(MODEL_PATH):
    joblib.dump(best_model_grid, MODEL_PATH)
    print("✅ Modèle sauvegardé.")
else:
    best_model_grid = joblib.load(MODEL_PATH)
    print("📦 Modèle chargé depuis le disque.")

# --- Obtenir les probabilités sur le test set ---
y_proba = best_model_grid.predict_proba(X_test)[:, 1]

# --- Trouver le meilleur seuil ---
def find_best_threshold_for_recall(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    best_index = recall.argmax()
    best_threshold = thresholds[best_index - 1] if best_index > 0 else 0.5
    print(f"🔍 Meilleur seuil trouvé : {best_threshold:.4f} avec recall = {recall[best_index]:.4f}")
    return best_threshold

best_threshold = find_best_threshold_for_recall(y_test, y_proba)

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)



# --- Modèle enveloppé avec seuil ---
final_model = ThresholdedClassifier(model=best_model_grid, threshold=best_threshold)

# --- Prédictions avec seuil personnalisé ---
y_pred = final_model.predict(X_test)
y_proba_full = final_model.predict_proba(X_test)

# --- Scores ---
scores = calculate_scores(y_test, y_pred, y_proba=y_proba_full[:, 1], prefix="metric_")
scores["threshold"] = best_threshold

# --- Enregistrement des résultats ---
save_results(
    model=best_model_grid,
    scores=scores,
    data_path=DF_PATH,
    output_path="../results_2.csv"
)

# --- (Optionnel) Affichage de la matrice de confusion ---
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matrice de confusion avec seuil personnalisé")
plt.show()


plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.show()

plt.plot(thresholds, precision[:-1], label='Précision')
plt.plot(thresholds, recall[:-1], label='Rappel')
plt.xlabel('Seuil de décision')
plt.ylabel('Score')
plt.legend()
plt.title('Courbe Précision-Rappel selon le seuil')
plt.grid()
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)



def find_best_threshold_for_recall(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # On veut maximiser le recall
    best_index = recall.argmax()
    best_threshold = thresholds[best_index - 1] if best_index > 0 else 0.5
    
    print(f"Meilleur seuil trouvé : {best_threshold:.4f} avec recall = {recall[best_index]:.4f}")
    return best_threshold

# Exemple d’utilisation
y_proba = best_model_grid.predict_proba(X_test)[:, 1]
best_threshold = find_best_threshold_for_recall(y_test, y_proba)

# Prédiction avec ce seuil personnalisé
y_pred_custom = (y_proba >= best_threshold).astype(int)

# Évaluation avec ce nouveau seuil
custom_recall = recall_score(y_test, y_pred_custom)
print(f"Recall avec seuil personnalisé : {custom_recall:.4f}")


def find_best_threshold_for_recall(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # On veut maximiser le recall
    best_index = recall.argmax()
    best_threshold = thresholds[best_index - 1] if best_index > 0 else 0.5
    
    print(f"Meilleur seuil trouvé : {best_threshold:.4f} avec recall = {recall[best_index]:.4f}")
    return best_threshold

# Exemple d’utilisation
y_proba = best_model_grid.predict_proba(X_test)[:, 1]
best_threshold = find_best_threshold_for_recall(y_test, y_proba)

# Prédiction avec ce seuil personnalisé
y_pred_custom = (y_proba >= best_threshold).astype(int)

# Évaluation avec ce nouveau seuil
custom_recall = recall_score(y_test, y_pred_custom)
print(f"Recall avec seuil personnalisé : {custom_recall:.4f}")
