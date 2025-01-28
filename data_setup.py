from datasets import load_dataset
import pandas as pd
import mysql.connector
from mysql.connector import errorcode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to connect to MySQL and create a database
def connect_and_create_database(host, user, password, db_name):
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password)
        cursor = conn.cursor()

        # Create database if it doesn't exist
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        logging.info(f"Database '{db_name}' created successfully!")

        # Use the database
        cursor.execute(f"USE {db_name}")
        return conn, cursor

    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            logging.error("Error: Invalid username or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            logging.error("Error: Database does not exist")
        else:
            logging.error(err)
        exit()

# Function to create tables
def create_tables(cursor):
    table_definitions = {
        "imdb_train": """
            CREATE TABLE IF NOT EXISTS imdb_train (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                label INT NOT NULL
            )
        """,
        "imdb_validation": """
            CREATE TABLE IF NOT EXISTS imdb_validation (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                label INT NOT NULL
            )
        """,
        "imdb_test": """
            CREATE TABLE IF NOT EXISTS imdb_test (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT NOT NULL,
                label INT NOT NULL
            )
        """
    }

    for table_name, create_query in table_definitions.items():
        cursor.execute(create_query)
        logging.info(f"Table '{table_name}' created successfully!")

# Function to insert data into tables
def insert_data_into_table(cursor, conn, table_name, df):
    insert_query = f"""
        INSERT INTO {table_name} (text, label)
        VALUES (%s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (row["text"], row["label"]))
    conn.commit()
    logging.info(f"Data inserted into table '{table_name}' successfully!")

# Main function to load dataset and store in MySQL
def load_and_store_dataset(host, user, password, db_name):
    try:
        # Step 1: Connect to the database
        conn, cursor = connect_and_create_database(host, user, password, db_name)

        # Step 2: Create tables
        create_tables(cursor)

        # Step 3: Load dataset
        ds = load_dataset("jahjinx/IMDb_movie_reviews")

        # Step 4: Process and insert data into each table
        for split in ['train', 'validation', 'test']:
            # Convert dataset split to DataFrame
            df_split = pd.DataFrame(ds[split])

            # Insert data into the corresponding table
            table_name = f"imdb_{split}"
            insert_data_into_table(cursor, conn, table_name, df_split)

        # Close the connection
        cursor.close()
        conn.close()
        logging.info("All data successfully stored in MySQL!")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    host = "localhost"  # Change to your MySQL host
    user = "root"  # Replace with your MySQL username
    password = "Abhishek"  # Replace with your MySQL password
    db_name = "IMDB"

    load_and_store_dataset(host, user, password, db_name)