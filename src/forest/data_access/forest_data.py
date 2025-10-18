from src.forest.configurations.mogo_db_connection import MongoDBClient
from src.forest.constants.database import DATABASE_NAME, COLLECTION_NAME
from src.forest.exception import ForestExpection
from src.forest.logger import logging
import pandas as pd
import sys
from typing import Optional
import numpy as np
        

class ForestData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise ForestExpection(e, sys)
        
    def export_collection_as_dataframe(self, collection_name: Optional[str] = COLLECTION_NAME, database_name: Optional[str] = DATABASE_NAME) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]
                
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df.drop('_id', axis=1, inplace=True)
                
            df.replace({"na": np.nan}, inplace=True)
            return df
        
        except Exception as e:
            raise ForestExpection(e, sys)
