import os
import pandas as pd
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_confusion_matrix(conf_matrix, labels, file_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()

def evaluate_model(base_dir):
    try:
        # Define paths
        preprocess_dir = os.path.join(base_dir, 'data', 'processed')
        models_dir = os.path.join(base_dir, 'models')
        report_dir = os.path.join(base_dir, 'reports')
        os.makedirs(report_dir, exist_ok=True)

        # Load vectorizer and model
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')

        if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
            logging.error("Vectorizer or model file not found.")
            return

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Evaluate on train, test, and validate splits
        for split in ['imdb_train', 'imdb_test', 'imdb_validation']:
            # Load data
            file_path = os.path.join(preprocess_dir, f"{split}.csv")
            if not os.path.exists(file_path):
                logging.warning(f"File '{file_path}' not found. Skipping.")
                continue

            data = pd.read_csv(file_path)

            if 'text' not in data.columns or 'label' not in data.columns:
                logging.warning(f"'text' or 'label' column not found in '{split}'. Skipping.")
                continue

            X = data['text']
            y = data['label']

            # Transform X using vectorizer
            X_tfidf = vectorizer.transform(X)

            # Predict using model
            y_pred = model.predict(X_tfidf)

            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred)
            conf_matrix = confusion_matrix(y, y_pred)

            # Save metrics to report folder
            report_path = os.path.join(report_dir, f"{split}_report.txt")
            with open(report_path, 'w') as f:
                f.write(f"Accuracy: {accuracy}\n\n")
                f.write("Classification Report:\n")
                f.write(f"{report}\n\n")

            # Save confusion matrix as a figure
            conf_matrix_path = os.path.join(report_dir, f"{split}_confusion_matrix.png")
            save_confusion_matrix(conf_matrix, labels=[0, 1], file_path=conf_matrix_path)

            logging.info(f"Report and confusion matrix for '{split}' saved to '{report_dir}'")

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")

# Example usage
if __name__ == "__main__":
    base_dir = "."  # Set base directory
    evaluate_model(base_dir)
