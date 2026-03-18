from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="E-commerce Hybrid Recommender API",
    description="ALS + LightGBM hybrid recommender system",
    version="1.0.0"
)

# Pydantic models
class RecommendRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "A2SUAM1J3GNN3B",
                "n_recommendations": 5
            }
        }

class Item(BaseModel):
    asin: str
    title: str
    score: float
    category: Optional[int] = None

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Item]
    processing_time_ms: float

# Global variables for models (loaded at startup)
cf_model = None
reranker = None
scaler = None
product_data = None
user_to_idx = None
item_to_idx = None
idx_to_item = None

@app.on_event("startup")
async def load_models():
    """Load all models at startup."""
    global cf_model, reranker, scaler, product_data
    global user_to_idx, item_to_idx, idx_to_item
    
    try:
        model_dir = Path("models")
        data_dir = Path("data/processed")
        
        # Load CF model
        with open(model_dir / "cf_model.pkl", "rb") as f:
            cf_data = pickle.load(f)
            cf_model = cf_data['model']
            user_to_idx = cf_data['user_to_idx']
            item_to_idx = cf_data['item_to_idx']
            idx_to_item = {v: k for k, v in item_to_idx.items()}
        
        # Load reranker if available
        reranker_path = model_dir / "full_reranker_model.pkl"
        if reranker_path.exists():
            with open(reranker_path, "rb") as f:
                reranker_data = pickle.load(f)
                reranker = reranker_data['model']
                scaler = reranker_data['scaler']
        
        # Load product data for titles
        product_data = pd.read_csv("data/raw/amazon_products.csv")
        product_data = product_data.set_index('asin')[['title', 'category_id']]
        
        logger.info(f"✅ Models loaded successfully")
        logger.info(f"   Users in model: {len(user_to_idx):,}")
        logger.info(f"   Items in model: {len(item_to_idx):,}")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "Hybrid Recommender API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": cf_model is not None
    }

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """Get personalized recommendations for a user."""
    import time
    start_time = time.time()
    
    # Check if user exists
    if request.user_id not in user_to_idx:
        # Return popular items for new users
        popular_items = product_data.head(request.n_recommendations)
        recommendations = [
            Item(
                asin=idx,
                title=row['title'],
                score=0.5,
                category=row['category_id']
            )
            for idx, row in popular_items.iterrows()
        ]
        
        return RecommendResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    # Get user index
    user_idx = user_to_idx[request.user_id]
    
    # Get recommendations from CF model
    user_factors = cf_model.user_factors[user_idx]
    item_factors = cf_model.item_factors
    
    # Calculate scores for all items
    scores = np.dot(item_factors, user_factors)
    
    # Get top N items
    top_indices = np.argsort(scores)[-request.n_recommendations:][::-1]
    
    # Convert to items
    recommendations = []
    for idx in top_indices:
        asin = idx_to_item[idx]
        score = float(scores[idx])
        
        # Get product info
        if asin in product_data.index:
            title = product_data.loc[asin, 'title']
            category = int(product_data.loc[asin, 'category_id']) if not pd.isna(product_data.loc[asin, 'category_id']) else None
        else:
            title = "Unknown Product"
            category = None
        
        recommendations.append(Item(
            asin=asin,
            title=title,
            score=score,
            category=category
        ))
    
    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        processing_time_ms=(time.time() - start_time) * 1000
    )

@app.get("/user/{user_id}/info")
async def get_user_info(user_id: str):
    """Get information about a user."""
    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = user_to_idx[user_id]
    return {
        "user_id": user_id,
        "user_index": user_idx,
        "in_model": True
    }

@app.get("/item/{asin}")
async def get_item_info(asin: str):
    """Get information about an item."""
    if asin in product_data.index:
        row = product_data.loc[asin]
        return {
            "asin": asin,
            "title": row['title'],
            "category": row['category_id'],
            "in_model": asin in item_to_idx
        }
    else:
        raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Set to False in production
    )