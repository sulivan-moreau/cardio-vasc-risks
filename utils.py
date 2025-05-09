from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# Pour enregistrer les résultat
import os
import csv
from datetime import datetime

MAX_ITER = 1000
DF_PATH = "data/cardio_optimized.csv"
RESULT_PATH = "../results.csv"

"""
 --- Pour utiliser dans le notebook ---

DF_PATH ="data/cardio_optimized.csv"

df = pd.read_csv(DF_PATH, sep=";")
X = df.drop(columns=["cardio"])
y = df["cardio"]
execute_model_and_save_score(X, y, max_iter=100)

"""

def results(result_file, ligne, entetes=None):
    try:
        file_exists = os.path.exists(result_file)
        with open(result_file, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=entetes or ligne.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(ligne)
        print(f"✅ Résultat enregistré dans {result_file}")
    except Exception as e:
        print(f"❌ Erreur d'enregistrement CSV : {e}")

def execute_model_and_save_score(X, y):
    data_name = os.path.basename(DF_PATH)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    rapport = classification_report(y_test, y_pred, output_dict=True)
    precision = round(rapport['macro avg']['precision'], 4)
    recall = round(rapport['macro avg']['recall'], 4)
    f1 = round(rapport['macro avg']['f1-score'], 4)

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = None

    print("✅ Accuracy :", round(accuracy, 4))
    print("📊 Rapport de classification :\n", classification_report(y_test, y_pred))
    print("🧱 Matrice de confusion :\n", cm)

    result_line = {
        'Timestamp': datetime.now().strftime('%m-%d %H:%M:%S'),
        'FichierDonnees': data_name,
        'Accuracy': round(accuracy, 4),
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'True Negative': tn,
        'False Positive': fp,
        'False Negative': fn,
        'True Positive': tp
    }
    results(RESULT_PATH, result_line, entetes=list(result_line.keys()))