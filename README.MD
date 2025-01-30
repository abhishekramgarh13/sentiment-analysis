# Sentiment Analysis

This repository contains a sentiment analysis project that predicts whether a movie review is positive or negative. The data is stored in a MySQL database, and the project includes a DVC pipeline for training and evaluating the model, as well as a Flask app for serving predictions. The model is deployed on Render, and the live application can be accessed [here](https://sentiment-analysis-abhishek-ramgarh.onrender.com/).

## Project Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/abhishekramgarh13/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up the MySQL database:

   - Ensure your MySQL server is running.
   - Configure your MySQL host, username, and password in `data_setup.py`.
   - Run the script to create the database and insert the data:
     ```bash
     python data_setup.py
     ```

## Data Acquisition

The dataset used for this project is `IMDb_movie_reviews` from the `jahjinx` dataset on Hugging Face.

Steps to load and store the data:

1. The script `data_setup.py` downloads the dataset using the `datasets` library.
2. The dataset is split into `train`, `validation`, and `test` sets, which are stored in corresponding MySQL tables (`imdb_train`, `imdb_validation`, `imdb_test`).

## Run Instructions

### 1. Run the DVC Pipeline

The DVC pipeline consists of the following stages:

- **data\_ingestion**: Loads the data into MySQL.
- **data\_preprocessing**: Preprocesses the text data.
- **model\_training**: Trains the sentiment analysis model.
- **model\_evaluation**: Evaluates the model's performance.

Ensure your MySQL server is running before executing the pipeline. To execute the entire pipeline, run:

```bash
dvc repro
```

### 2. Start the Flask Server

To start the Flask server for serving predictions, use:

```bash
python app.py
```

The server will run on `http://127.0.0.1:5000/` by default.

### 3. Test the Endpoint

To test the prediction endpoint, you can use `curl` or a tool like Postman. Example `curl` command:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text": "This is an amazing movie!"}'
```

The response will contain the predicted sentiment.

### 4. Access the Deployed Model

The model is deployed on Render. You can access the live application here: [Sentiment Analysis - Render Deployment](https://sentiment-analysis-abhishek-ramgarh.onrender.com/).

## Model Info

### Chosen Model

- **Model Architecture:** Logistic Regression
- **Libraries Used:** Scikit-learn

### Key Results

- **Training Accuracy:** 91%
- **Validation Accuracy:** 89%
- **Test Accuracy:** 89%

Additionally, various models were tested during experimentation using MLflow. You can explore the experiments and results [here](https://dagshub.com/abhishekramgarh13/sentiment-analysis.mlflow).

## Pipeline Visualization

Below is the DVC pipeline used in this project:

![Pipeline Visualization](./Screenshot%202025-01-30%20143624.png)

## Repository Link

Find the complete code and resources on GitHub: [Sentiment Analysis Repository](https://github.com/abhishekramgarh13/sentiment-analysis)

---

Feel free to open an issue or submit a pull request if you have suggestions or improvements!

