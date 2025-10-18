import sys
import os
from src.forest.exception import ForestExpection
from src.forest.constants.database import DATABASE_NAME, COLLECTION_NAME
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()

ca = certifi.where()

username = os.getenv("MONGO_DB_USERNAME")
password = os.getenv("MONGO_DB_PASSWORD")

class MongoDBClient:
    client = None

    def __init__(self, database_name = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                uri = f"mongodb+srv://{username}:{password}@cluster0.2gvphkf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
                if uri is None:
                    raise ForestExpection("MongoDB URI is not set. Please check your environment variables.")
                MongoDBClient.client = MongoClient(uri, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[DATABASE_NAME]
            self.collection = self.database[COLLECTION_NAME]
        except Exception as e:
            raise ForestExpection(e, sys)