import os
import pandas as pd
import mysql.connector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to connect to MySQL and fetch data
def fetch_data_from_table(host, user, password, db_name, table_name):
    try:
        # Connect to the MySQL database
        conn = mysql.connector.connect(host=host, user=user, password=password, database=db_name)
        cursor = conn.cursor()

        # Query to fetch all data from the table
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)

        # Fetch all rows and column names
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

        # Convert to Pandas DataFrame
        df = pd.DataFrame(rows, columns=columns)
        logging.info(f"Data fetched successfully from table '{table_name}'")
        
        # Drop the 'id' column
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

        # Close the connection
        cursor.close()
        conn.close()

        return df

    except mysql.connector.Error as err:
        logging.error(f"Error: {err}")
        return None

# Fetch all tables' data and drop the 'id' column
def fetch_all_data(host, user, password, db_name):
    datasets = {}
    for table_name in ['imdb_train', 'imdb_validation', 'imdb_test']:
        df = fetch_data_from_table(host, user, password, db_name, table_name)
        if df is not None:
            datasets[table_name] = df
    return datasets

# Save data to the specified directory
def save_data_to_directory(data, base_dir):
    try:
        # Create directories if they don't exist
        raw_dir = os.path.join(base_dir, 'data', 'raw')
        os.makedirs(raw_dir, exist_ok=True)

        # Save each dataset to a CSV file
        for table, df in data.items():
            file_path = os.path.join(raw_dir, f"{table}.csv")
            df.to_csv(file_path, index=False)
            logging.info(f"Data from '{table}' saved to '{file_path}'")

    except Exception as e:
        logging.error(f"Error saving data: {e}")

# Example usage
if __name__ == "__main__":
    host = "localhost"  # Change to your MySQL host
    user = "root"  # Replace with your MySQL username
    password = "Abhishek"  # Replace with your MySQL password
    db_name = "IMDB"

    # Fetch data from all tables
    data = fetch_all_data(host, user, password, db_name)
    
    if data:
        # Save data to the 'data/raw' directory
        save_data_to_directory(data, base_dir=".")

