from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from src.forest.entity.s3_estimator import SensorEstimator
from src.forest.entity.config_entity import PredictionPipelineConfig
from src.forest.pipeline.prediction_pipeline import PredictionPipeline

import pandas as pd


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Forest Cover Prediction API",
    description="An API to predict forest cover type based on land attributes.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows your HTML frontend (running on a different origin) to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Prediction Endpoint ---
@app.post("/predict")
async def predict(request: Request):
    """
    Accepts land attribute data, validates it, and uses the real
    PredictionPipeline to return a forest cover type prediction.
    """
    try:
        data = await request.json()   # This should be a dict

        # Convert into a pandas DataFrame
        df = pd.DataFrame(data, index=[0])   # ✅ wrap in list to make it a single-row DataFrame
        
        pipeline = PredictionPipeline()
        prediction_result = pipeline.predict(dataframe = df)
        prediction = int(prediction_result[0])
        
        print("Prediction result:", prediction)
        
        return {"prediction": prediction}

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
        


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
