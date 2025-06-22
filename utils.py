from datetime import datetime
import os, csv, joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, average_precision_score



def save_and_compare_grid_result(model, params, scores, result_file="results_3.csv", model_dir="models", prefix="model_logreg"):
    """
    Sauvegarde le modèle, les paramètres et les métriques dans un CSV.
    Compare le recall_1 aux anciennes versions et retourne le meilleur.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Convertir score en dict simple
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recall_1": scores.get("metric_recall_1", None),
        **params,
        **scores
    }

    # Lire l'existant
    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
    else:
        df = pd.DataFrame()

    # Ajouter la ligne actuelle
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(result_file, index=False)

    # Sauvegarde du modèle avec recall_1
    recall_val = row["recall_1"]
    if recall_val is not None:
        model_path = os.path.join(model_dir, f"{prefix}_recall1_{recall_val:.4f}.joblib")
        joblib.dump(model, model_path)
        print(f"✅ Modèle sauvegardé : {model_path}")

    # Trouver la meilleure ligne
    best_row = df.loc[df["recall_1"].idxmax()]
    best_params = {k: best_row[k] for k in params.keys()}
    best_scores = {k: best_row[k] for k in scores.keys()}
    
    print("🔍 Meilleurs paramètres selon recall_1 :", best_params)
    print("📊 Meilleures métriques :", best_scores)

    return best_params, best_scores

def load_best_model_from_results(results_file, model_dir):
    if not os.path.exists(results_file):
        return None

    df = pd.read_csv(results_file)
    if 'metric_recall_1' not in df.columns:
        return None

    best_row = df.sort_values("metric_recall_1", ascending=False).iloc[0]
    model_name = best_row.get("model_file", None)
    if model_name and os.path.exists(os.path.join(model_dir, model_name)):
        return joblib.load(os.path.join(model_dir, model_name))
    return None




def execute_model(X, y, model=None, test_size=0.2, random_state=42, stratify=True):
    from sklearn.model_selection import train_test_split

    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    if model is None:
        raise ValueError("Un modèle entraîné doit être passé à execute_model.")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    return y_test, y_pred, y_proba, X_test, model


def calculate_scores(y_true, y_pred, y_proba=None, prefix="metric_"):
    report = classification_report(y_true, y_pred, output_dict=True)

    scores = {
        f"{prefix}Accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}Precision": report["macro avg"]["precision"],
        f"{prefix}Recall": report["macro avg"]["recall"],
        f"{prefix}Recall_1": report.get("1", {}).get("recall", None),
        f"{prefix}F1_Score": report["macro avg"]["f1-score"],
    }

    if y_proba is not None:
        scores[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba)
        scores[f"{prefix}average_precision"] = average_precision_score(y_true, y_proba)


    return scores




def save_results(model, scores, data_path=None, output_path="../results_2.csv"):
    import json

    # Récupération des paramètres
    if hasattr(model, "best_estimator_"):  # GridSearchCV
        params = model.best_params_
    elif hasattr(model, "get_params"):     # LogisticRegression ou autre modèle sklearn
        params = model.get_params()
    else:
        params = {}

    # Convertir les paramètres complexes (dict, list) en JSON string pour CSV
    params_clean = {}
    for k, v in params.items():
        if isinstance(v, (dict, list)):
            params_clean[f"param_{k}"] = json.dumps(v)
        else:
            params_clean[f"param_{k}"] = v

    data_name = os.path.basename(data_path) if data_path else "unknown"

    result_line = {
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "Dataset": data_name,
        **params_clean,
        **scores
    }

    headers = list(result_line.keys())

    try:
        file_exists = os.path.exists(output_path)
        with open(output_path, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(result_line)
        print(f"✅ Résultat enregistré dans {output_path}")
    except Exception as e:
        print(f"❌ Erreur d'enregistrement CSV : {e}")



class ThresholdedClassifier:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        
    def predict(self, X):
        proba = self.model.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def set_threshold(self, threshold):
        self.threshold = threshold
