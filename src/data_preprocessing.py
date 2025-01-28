import os
import pandas as pd
import logging
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the preprocessing function
def preprocess_comment(comment):
    # Convert to lowercase
    comment = comment.lower()

    # Remove trailing and leading whitespaces
    comment = comment.strip()

    # Remove URLs
    comment = re.sub(r'https?://\S+|www\.\S+', '', comment)

    # Remove HTML tags
    comment = re.sub(r'<.*?>', '', comment)

    # Remove newline characters
    comment = re.sub(r'\n', ' ', comment)

    # Remove non-alphanumeric characters and punctuation
    comment = re.sub(r'[^\w\s]', '', comment)

    # Remove stopwords but retain important ones for sentiment analysis
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    comment = ' '.join([word for word in comment.split() if word not in stop_words])

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

    return comment

# Load, preprocess, and save data
def load_preprocess_save(base_dir):
    try:
        # Define paths
        raw_dir = os.path.join(base_dir, 'data', 'raw')
        preprocess_dir = os.path.join(base_dir, 'data', 'processed')
        os.makedirs(preprocess_dir, exist_ok=True)

        # Process each dataset
        for split in ['imdb_train', 'imdb_validation', 'imdb_test']:
            # Load data
            file_path = os.path.join(raw_dir, f"{split}.csv")
            if not os.path.exists(file_path):
                logging.warning(f"File '{file_path}' not found. Skipping.")
                continue

            df = pd.read_csv(file_path)

            # Remove duplicates
            df.drop_duplicates(inplace=True)

            # Apply preprocessing to the 'text' column
            if 'text' in df.columns:
                df['text'] = df['text'].apply(preprocess_comment)
            else:
                logging.warning(f"'text' column not found in '{split}'. Skipping preprocessing.")

            # Save preprocessed data
            preprocess_path = os.path.join(preprocess_dir, f"{split}.csv")
            df.to_csv(preprocess_path, index=False)
            logging.info(f"Preprocessed data saved to '{preprocess_path}'")

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")

# Example usage
if __name__ == "__main__":
    base_dir = "."  # Set base directory
    load_preprocess_save(base_dir)