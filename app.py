from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import uvicorn
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="E-commerce Hybrid Recommender API",
    description="ALS + LightGBM hybrid recommender system for Amazon products",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models (updated for V2 compatibility)
class RecommendRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "A2SUAM1J3GNN3B",
                "n_recommendations": 5
            }
        }
    )

class Item(BaseModel):
    asin: str
    title: str
    score: float
    category: Optional[int] = None
    price: Optional[float] = None
    stars: Optional[float] = None

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Item]
    processing_time_ms: float
    model_version: str = "1.0.0"
    timestamp: str = ""

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    users_in_model: int
    items_in_model: int
    version: str = "1.0.0"

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
    """Load all models at startup from data/processed directory."""
    global cf_model, reranker, scaler, product_data
    global user_to_idx, item_to_idx, idx_to_item
    
    try:
        # ✅ CORRECT PATH: models are in data/processed, not /models
        data_dir = Path("data/processed")
        raw_dir = Path("data/raw")
        
        # Check if directory exists
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return
        
        # Load CF model from data/processed
        cf_path = data_dir / "cf_model.pkl"
        if not cf_path.exists():
            logger.error(f"CF model not found at {cf_path}")
            logger.info("Please run scripts/02_train_cf.py first")
            return
        
        logger.info(f"Loading CF model from {cf_path}")
        with open(cf_path, "rb") as f:
            cf_data = pickle.load(f)
            cf_model = cf_data['model']
            user_to_idx = cf_data['user_to_idx']
            item_to_idx = cf_data['item_to_idx']
            idx_to_item = {v: k for k, v in item_to_idx.items()}
        
        logger.info(f"✅ CF Model loaded: {cf_model.factors} factors")
        logger.info(f"   Users: {len(user_to_idx):,}")
        logger.info(f"   Items: {len(item_to_idx):,}")
        
        # Load reranker from data/processed if available
        reranker_path = data_dir / "full_reranker_model.pkl"
        if reranker_path.exists():
            logger.info(f"Loading reranker from {reranker_path}")
            with open(reranker_path, "rb") as f:
                reranker_data = pickle.load(f)
                reranker = reranker_data['model']
                scaler = reranker_data['scaler']
            logger.info("✅ Reranker model loaded")
        else:
            logger.warning("Reranker model not found, using CF only")
        
        # Load product data for titles and metadata
        product_path = raw_dir / "amazon_products.csv"
        if product_path.exists():
            logger.info(f"Loading product data from {product_path}")
            product_data = pd.read_csv(product_path)
            product_data = product_data.set_index('asin')[['title', 'category_id', 'price', 'stars']]
            logger.info(f"✅ Product data loaded: {len(product_data):,} products")
        else:
            logger.warning(f"Product data not found at {product_path}")
            product_data = pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

def get_cf_recommendations(user_id: str, n: int = 10):
    """Get recommendations using collaborative filtering."""
    if user_id not in user_to_idx:
        return []
    
    user_idx = user_to_idx[user_id]
    
    # Calculate scores for all items using dot product
    user_factors = cf_model.user_factors[user_idx]
    item_factors = cf_model.item_factors
    
    scores = np.dot(item_factors, user_factors)
    
    # Get top N items
    top_indices = np.argsort(scores)[-n:][::-1]
    
    recommendations = []
    for idx in top_indices:
        asin = idx_to_item[idx]
        score = float(scores[idx])
        
        # Get product info if available
        if asin in product_data.index:
            row = product_data.loc[asin]
            recommendations.append({
                'asin': asin,
                'title': row['title'] if pd.notna(row['title']) else "Unknown",
                'score': score,
                'category': int(row['category_id']) if pd.notna(row['category_id']) else None,
                'price': float(row['price']) if pd.notna(row['price']) else None,
                'stars': float(row['stars']) if pd.notna(row['stars']) else None
            })
        else:
            recommendations.append({
                'asin': asin,
                'title': "Unknown Product",
                'score': score,
                'category': None,
                'price': None,
                'stars': None
            })
    
    return recommendations

@app.get("/")
async def root():
    return {
        "message": "E-commerce Hybrid Recommender API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if cf_model is not None else "degraded",
        models_loaded=cf_model is not None,
        users_in_model=len(user_to_idx) if user_to_idx else 0,
        items_in_model=len(item_to_idx) if item_to_idx else 0
    )

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """Get personalized recommendations for a user."""
    start_time = time.time()
    
    # Check if models are loaded
    if cf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    # Check if user exists
    if request.user_id not in user_to_idx:
        # For new users, return popular items
        logger.info(f"User {request.user_id} not found, returning popular items")
        
        if not product_data.empty:
            # Get popular items by stars
            popular_items = product_data.nlargest(request.n_recommendations, 'stars')
            recommendations = []
            for asin, row in popular_items.iterrows():
                recommendations.append(Item(
                    asin=asin,
                    title=row['title'] if pd.notna(row['title']) else "Unknown",
                    score=0.5,
                    category=int(row['category_id']) if pd.notna(row['category_id']) else None,
                    price=float(row['price']) if pd.notna(row['price']) else None,
                    stars=float(row['stars']) if pd.notna(row['stars']) else None
                ))
        else:
            recommendations = []
    
    else:
        # Get CF recommendations
        cf_recs = get_cf_recommendations(
            request.user_id, 
            request.n_recommendations
        )
        
        # Convert to Item models
        recommendations = [Item(**rec) for rec in cf_recs]
    
    processing_time = (time.time() - start_time) * 1000
    
    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        processing_time_ms=round(processing_time, 2),
        timestamp=datetime.now().isoformat()
    )

@app.get("/user/{user_id}/info")
async def get_user_info(user_id: str):
    """Get information about a user."""
    if user_id not in user_to_idx:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = user_to_idx[user_id]
    
    # Get user's interactions (if available)
    train_path = Path("data/processed/train_interactions.parquet")
    if train_path.exists():
        train_df = pd.read_parquet(train_path)
        user_history = train_df[train_df['user_id'] == user_id]
        n_interactions = len(user_history)
        avg_rating = user_history['rating'].mean() if n_interactions > 0 else None
    else:
        n_interactions = None
        avg_rating = None
    
    return {
        "user_id": user_id,
        "user_index": user_idx,
        "in_model": True,
        "n_interactions": n_interactions,
        "avg_rating": round(avg_rating, 2) if avg_rating else None
    }

@app.get("/item/{asin}")
async def get_item_info(asin: str):
    """Get information about an item."""
    if asin in product_data.index:
        row = product_data.loc[asin]
        return {
            "asin": asin,
            "title": row['title'] if pd.notna(row['title']) else "Unknown",
            "category": int(row['category_id']) if pd.notna(row['category_id']) else None,
            "price": float(row['price']) if pd.notna(row['price']) else None,
            "stars": float(row['stars']) if pd.notna(row['stars']) else None,
            "in_model": asin in item_to_idx if item_to_idx else False,
            "item_index": item_to_idx.get(asin) if item_to_idx else None
        }
    else:
        raise HTTPException(status_code=404, detail="Item not found")

@app.get("/stats")
async def get_stats():
    """Get model statistics."""
    return {
        "users_in_model": len(user_to_idx) if user_to_idx else 0,
        "items_in_model": len(item_to_idx) if item_to_idx else 0,
        "products_in_catalog": len(product_data) if not product_data.empty else 0,
        "has_reranker": reranker is not None,
        "model_version": "1.0.0"
    }

@app.get("/debug/paths")
async def debug_paths():
    """Debug endpoint to check file paths."""
    data_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    return {
        "data_processed_exists": data_dir.exists(),
        "cf_model_exists": (data_dir / "cf_model.pkl").exists(),
        "reranker_exists": (data_dir / "full_reranker_model.pkl").exists(),
        "product_data_exists": (raw_dir / "amazon_products.csv").exists(),
        "data_processed_files": [f.name for f in data_dir.glob("*")] if data_dir.exists() else [],
        "current_directory": str(Path.cwd())
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )