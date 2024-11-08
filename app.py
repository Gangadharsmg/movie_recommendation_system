from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from prometheus_fastapi_instrumentator import Instrumentator
import redis
import time
import numpy as np

# Initialize FastAPI, Redis, and model
app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0)  # Set Redis configuration

# Initialize metrics and model
Instrumentator().instrument(app).expose(app)
model = LogisticRegression()
model.fit(np.array([[1], [2], [3]]), [0, 1, 1])  # Train a dummy model for demonstration

# Feature store helper functions
def set_user_feature(user_id: str, feature: dict):
    redis_client.hset(user_id, mapping=feature)

def get_user_feature(user_id: str):
    feature = redis_client.hgetall(user_id)
    if not feature:
        raise HTTPException(status_code=404, detail="User not found in feature store")
    return {k.decode(): float(v.decode()) for k, v in feature.items()}

# Data Model
class UserActivity(BaseModel):
    user_id: str
    activity: List[float]  # Example feature: User preferences

# Endpoint to update user features in the feature store
@app.post("/update_feature/")
async def update_user_feature(data: UserActivity):
    # Simulate processing and updating features
    avg_activity = sum(data.activity) / len(data.activity)
    set_user_feature(data.user_id, {"average_activity": avg_activity})
    return {"message": "User feature updated"}

# Endpoint to predict recommendation and track metrics
@app.get("/recommend/{user_id}")
async def recommend(user_id: str):
    start_time = time.time()
    
    # Retrieve user feature and make prediction
    try:
        feature = get_user_feature(user_id)
        feature_array = np.array([[feature["average_activity"]]])
        prediction = model.predict(feature_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Calculate latency
    latency = time.time() - start_time
    Instrumentator().metrics(f'latency', latency)

    return {
        "user_id": user_id,
        "recommendation": int(prediction[0]),
        "latency": latency
    }

# Endpoint to check model accuracy (for demonstration purposes)
@app.get("/model_accuracy/")
async def model_accuracy():
    y_true = [0, 1, 1]  # True labels for demo
    y_pred = model.predict(np.array([[1], [2], [3]]))
    accuracy = accuracy_score(y_true, y_pred)
    return {"model_accuracy": accuracy}
