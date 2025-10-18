DATABASE_NAME = "green_vision"
COLLECTION_NAME = "forest"


from dotenv import load_dotenv
import os
load_dotenv()

username = os.getenv("MONGO_DB_USERNAME")
password = os.getenv("MONGO_DB_PASSWORD")