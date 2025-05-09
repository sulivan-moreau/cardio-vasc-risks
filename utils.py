"""
utils.py - Fonctions pour entraîner un modèle de régression logistique
et enregistrer les résultats dans un fichier CSV.

 --- Exemple d'utilisation dans le notebook modelisation.ipynb ---

DF_PATH ="data/cardio_optimized.csv"

df = pd.read_csv(DF_PATH, sep=";")
X = df.drop(columns=["cardio"])
y = df["cardio"]
execute_model_and_save_score(X, y, max_iter=100)

"""


# Pour enregistrer les résultat
import os
import csv
from datetime import datetime

# Pour la modèlisation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# --- Paramètres par défaut (remplis avec None pour override possible) ---
MAX_ITER = None
MAX_ITER = None
PENALTY = None
SOLVER = None
C = None
FIT_INTERCEPT = None
CLASS_WEIGHT = None

# Output
RESULT_PATH = "../results.csv"

# --- En-têtes du .csv de sortie ---
EN_TETES = [
    'Timestamp', 'FichierDonnees',
    'max_iter', 'penalty', 'solver', 'C',
    'Accuracy', 'Precision', 'Recall', 'F1-Score',
    'True Negative', 'False Positive', 'False Negative', 'True Positive'
]



def results(result_file, ligne, entetes=None):
    """Enregistre une ligne de résultats dans un fichier CSV (avec en-têtes si fichier inexistant)."""
    try:
        file_exists = os.path.exists(result_file)
        with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=entetes or ligne.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(ligne)
        print(f"Résultat enregistré dans {result_file}")
    except Exception as e:
        print(f"Erreur d'enregistrement CSV : {e}")

def to_str_or_default(value):
    return "default" if value is None else value

def execute_model_and_save_score(X, y, df_path,
                                 max_iter=MAX_ITER,
                                 penalty=PENALTY,
                                 solver=SOLVER,
                                 C=C,
                                 fit_intercept=FIT_INTERCEPT,
                                 class_weight=CLASS_WEIGHT):
    """Entraîne un modèle, affiche les scores et sauvegarde les résultats dans un CSV."""

    data_name = os.path.basename(df_path)

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Création dynamique des arguments non None de LogisticRegression
    model_args = {
        k: v for k, v in {
            'max_iter': max_iter,
            'penalty': penalty,
            'solver': solver,
            'C': C,
            'fit_intercept': fit_intercept,
            'class_weight': class_weight
        }.items() if v is not None
    }

    model = LogisticRegression(**model_args)
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)

    # Évaluation
    accuracy = accuracy_score(y_test, y_pred)
    rapport = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    precision = round(rapport['macro avg']['precision'], 4)
    recall = round(rapport['macro avg']['recall'], 4)
    f1 = round(rapport['macro avg']['f1-score'], 4)


    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (None, None, None, None)

    # Affichage console
    print("✅ Accuracy :", round(accuracy, 4))
    print("📊 Rapport de classification :\n", classification_report(y_test, y_pred))
    print("🧱 Matrice de confusion :\n", cm)

    # Sauvegarde .csv
    result_line = {
        'Timestamp': datetime.now().strftime('%m-%d %H:%M:%S'),
        'FichierDonnees': data_name,
        'max_iter': to_str_or_default(max_iter),
        'penalty': to_str_or_default(penalty),
        'solver': to_str_or_default(solver),
        'C': to_str_or_default(C),
        'Accuracy': round(accuracy, 4),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }

    results(RESULT_PATH, result_line, entetes=EN_TETES )