import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load your CSV data
# df = pd.read_csv('data/2020to2024injuries.csv')

# Get DATABASE_URL from environment variables
DATABASE_URL = os.getenv('DATABASE_URL')

engine = create_engine(DATABASE_URL)

# Write the dataframe to PostgreSQL
# df.to_sql('injuries', engine, if_exists='replace', index=False)

directory = "../data/"
list_of_dataframes = []
for file in os.listdir(directory):
    if file.endswith("log.csv"):
        df = pd.read_csv(os.path.join(directory, file))
        list_of_dataframes.append(df)
combined_dataframe = pd.concat(list_of_dataframes, ignore_index=True)

combined_dataframe.to_sql('pitches', engine, if_exists='replace', index=False)