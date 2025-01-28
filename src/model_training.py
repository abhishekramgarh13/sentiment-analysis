import os
import pandas as pd
import logging
import pickle
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_model(base_dir):
    try:
        # Define paths
        preprocess_dir = os.path.join(base_dir, 'data', 'processed')
        models_dir = os.path.join(base_dir, 'models')
        params_path = os.path.join(base_dir, 'params.yaml')
        os.makedirs(models_dir, exist_ok=True)

        # Load parameters from params.yaml
        if not os.path.exists(params_path):
            logging.error(f"Parameters file '{params_path}' not found.")
            return

        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)

        # Extract training parameters
        vectorizer_params = params.get('Train', {}).get('Vectorizer', {})
        lr_params = params.get('Train', {}).get('LR', {})

        # Load Processed train data
        train_path = os.path.join(preprocess_dir, 'imdb_train.csv')
        if not os.path.exists(train_path):
            logging.error(f"Processed train data '{train_path}' not found.")
            return

        train_data = pd.read_csv(train_path)

        # Divide into X_train and y_train
        if 'text' not in train_data.columns or 'label' not in train_data.columns:
            logging.error("'text' or 'label' column not found in train data.")
            return

        X_train = train_data['text']
        y_train = train_data['label']

        # Apply TF-IDF Vectorization with parameters from params.yaml
        vectorizer = TfidfVectorizer(**vectorizer_params)
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Train Logistic Regression model with parameters from params.yaml
        model = LogisticRegression(**lr_params)
        model.fit(X_train_tfidf, y_train)
        logging.info("Logistic Regression model trained successfully.")

        # Save vectorizer
        vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        logging.info(f"TF-IDF Vectorizer saved to '{vectorizer_path}'")

        # Save model
        model_path = os.path.join(models_dir, 'logistic_regression_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Logistic Regression model saved to '{model_path}'")

    except Exception as e:
        logging.error(f"Error during model training and saving: {e}")

# Example usage
if __name__ == "__main__":
    base_dir = "."  # Set base directory
    train_and_save_model(base_dir)
